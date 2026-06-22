import urllib.request
import json
import os
import math
import ssl
from pathlib import Path
import numpy as np
import config

ssl_context = ssl._create_unverified_context()

class RAGEngine:
    def __init__(self):
        self.cache_path = config.EMBEDDING_CACHE_PATH
        self.matrix_cache_path = config.MATRIX_CACHE_PATH
        self.chunks_cache_path = config.CHUNKS_CACHE_PATH
        self.documents_dir = config.DOCUMENTS_DIR
        self.api_key = config.OPENROUTER_API_KEY
        self.embedding_url = config.OPENROUTER_EMBEDDING_URL
        self.model = config.EMBEDDING_MODEL
        
        # Lazy load JSON cache to save memory and parsing time
        self.cache = None
        self.chunks = []
        self.vectors_matrix = np.empty((0, 0), dtype=np.float32)
        
        # Attempt to load binary cache for query acceleration
        if self.matrix_cache_path.exists() and self.chunks_cache_path.exists():
            try:
                with open(self.chunks_cache_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.vectors_matrix = np.load(self.matrix_cache_path, mmap_mode='r')
            except Exception as e:
                print(f"[RAG] Warning: Failed to load numpy binary cache: {e}. Rebuilding...")
                self._build_binary_cache_from_json()
        else:
            self._build_binary_cache_from_json()

    def _build_binary_cache_from_json(self):
        """Construct binary cache from original JSON file"""
        if self.cache is None:
            self.cache = self._load_cache()
        self._save_binary_cache()

    def _load_cache(self):
        """Load cached embeddings from JSON file"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[RAG] Warning: Failed to load embedding cache: {e}")
        return {}

    def _save_cache(self):
        """Save updated embeddings cache to JSON file"""
        if self.cache is None:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[RAG] Warning: Failed to save embedding cache: {e}")

    def _save_binary_cache(self):
        """Save the current chunks list and a float32 NumPy matrix of embeddings"""
        if self.cache is None:
            return
        try:
            self.chunks = list(self.cache.keys())
            vectors = [self.cache[chunk] for chunk in self.chunks]
            if vectors:
                # Ensure parents exist
                self.matrix_cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.vectors_matrix = np.array(vectors, dtype=np.float32)
                np.save(self.matrix_cache_path, self.vectors_matrix)
                with open(self.chunks_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.chunks, f, ensure_ascii=False, indent=2)
                # Re-load as memory-mapped
                self.vectors_matrix = np.load(self.matrix_cache_path, mmap_mode='r')
            else:
                self.vectors_matrix = np.empty((0, 0), dtype=np.float32)
        except Exception as e:
            print(f"[RAG] Warning: Failed to save binary cache: {e}")

    def _get_embeddings_batch(self, texts: list[str], input_type: str = "passage") -> list[list]:
        """Fetch embeddings for a batch of texts in one API call"""
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return []

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment.")

        payload = {
            "model": self.model,
            "input": texts,          # list instead of single string
            "input_type": input_type
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000"
        }

        req = urllib.request.Request(
            self.embedding_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            response = urllib.request.urlopen(req, context=ssl_context, timeout=60)
            res_data = json.loads(response.read().decode("utf-8"))
            if "data" in res_data and len(res_data["data"]) > 0:
                # API returns embeddings in order, sorted by index
                return [item["embedding"] for item in sorted(res_data["data"], key=lambda x: x["index"])]
            else:
                raise RuntimeError(f"OpenRouter embedding error: {res_data}")
        except Exception as e:
            print(f"[RAG] Batch embedding failed: {e}")
            raise e

    def _get_embedding(self, text: str, input_type: str = "passage") -> list:
        """Fetch single embedding (checks JSON cache first for passages, then calls batch endpoint)"""
        # Clean text
        text = text.strip()
        if not text:
            return []

        # Check JSON cache (load it first if needed)
        if input_type == "passage":
            if self.cache is None:
                self.cache = self._load_cache()
            if text in self.cache:
                return self.cache[text]

        # Fetch from batch endpoint
        vectors = self._get_embeddings_batch([text], input_type=input_type)
        vector = vectors[0] if vectors else []

        # Cache single passage embedding if retrieved successfully
        if input_type == "passage" and vector:
            if self.cache is None:
                self.cache = self._load_cache()
            self.cache[text] = vector
            self._save_cache()

        return vector

    def _chunk_text(self, text: str, max_words: int = 150) -> list:
        """Split text into manageable paragraph chunks"""
        # Clean text: replace zero-width spaces with a regular space to prevent word concatenation
        text = text.replace("\u200b", " ")
        # Remove zero-width joiners and byte order marks
        for zw in ["\u200c", "\u200d", "\ufeff"]:
            text = text.replace(zw, "")

        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            words = p.split()
            # If paragraph is too long, chunk it by word count
            if len(words) > max_words:
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i:i + max_words]))
            else:
                chunks.append(p)
        return chunks

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using pypdf if available, otherwise fallback"""
        text = ""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except ImportError:
            print("[RAG] Warning: 'pypdf' package not installed. Cannot parse PDF files.")
            text = f"PDF parsing failed: Please install pypdf to read {pdf_path.name}"
        except Exception as e:
            print(f"[RAG] Error parsing PDF {pdf_path.name}: {e}")
        return text

    def ingest_documents(self, batch_size: int = 32) -> int:
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return 0

        if self.cache is None:
            self.cache = self._load_cache()

        all_chunks = []
        for file_path in self.documents_dir.iterdir():
            if file_path.suffix == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        all_chunks.extend(self._chunk_text(f.read()))
                except Exception as e:
                    print(f"[RAG] Error reading {file_path.name}: {e}")
            elif file_path.suffix == ".pdf":
                all_chunks.extend(self._chunk_text(self.extract_text_from_pdf(file_path)))

        # Filter only chunks not already cached
        new_chunks = [c for c in all_chunks if c not in self.cache]
        print(f"[RAG] {len(all_chunks)} total chunks, {len(new_chunks)} new to embed.")

        new_count = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            try:
                vectors = self._get_embeddings_batch(batch, input_type="passage")
                for chunk, vector in zip(batch, vectors):
                    self.cache[chunk] = vector
                    new_count += 1
                print(f"[RAG] Embedded {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} chunks...")
            except Exception as e:
                print(f"[RAG] Batch {i // batch_size + 1} failed: {e}")

        if new_count > 0 or not self.matrix_cache_path.exists() or len(self.chunks) == 0:
            print("[RAG] Rebuilding NumPy matrix cache...")
            self._save_cache()
            self._save_binary_cache()
        else:
            print("[RAG] Finished scan. No new chunks added.")

        return new_count

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve most relevant document chunks for the query using NumPy matrix operations"""
        if len(self.chunks) == 0 or self.vectors_matrix.size == 0:
            return ""

        try:
            # Get embedding of the query using input_type='query'
            query_vector = self._get_embedding(query, input_type="query")
        except Exception as e:
            print(f"[RAG] Query embedding failed: {e}. Falling back to empty context.")
            return ""

        if not query_vector:
            return ""

        try:
            # Convert query vector to NumPy array
            q_arr = np.array(query_vector, dtype=np.float32)
            
            # Compute dot products
            dot_products = np.dot(self.vectors_matrix, q_arr)
            
            # Compute norms
            mags = np.linalg.norm(self.vectors_matrix, axis=1)
            q_mag = np.linalg.norm(q_arr)
            
            # Safe divide to calculate cosine similarities
            denominators = mags * q_mag
            with np.errstate(invalid='ignore', divide='ignore'):
                sims = np.where(denominators > 0, dot_products / denominators, 0.0)
            
            # Sort by similarity score descending
            indices = np.argsort(sims)[::-1]
            
            # Build context from top_k results with a lower threshold (0.15) to prevent false negatives
            top_results = []
            for idx in indices[:top_k]:
                if sims[idx] > 0.15:
                    top_results.append(self.chunks[idx])
            
            return "\n\n---\n\n".join(top_results)
        except Exception as e:
            print(f"[RAG] Vector similarity search failed: {e}. Falling back to empty context.")
            return ""
