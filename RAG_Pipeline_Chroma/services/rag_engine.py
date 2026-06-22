import urllib.request
import json
import os
import ssl
import hashlib
from pathlib import Path
import chromadb
import config

ssl_context = ssl._create_unverified_context()

class RAGEngine:
    def __init__(self):
        self.documents_dir = config.DOCUMENTS_DIR
        self.api_key = config.OPENROUTER_API_KEY
        self.embedding_url = config.OPENROUTER_EMBEDDING_URL
        self.model = config.EMBEDDING_MODEL
        
        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        
        # Get or create collection with cosine similarity
        self.collection = self.chroma_client.get_or_create_collection(
            name="campus_documents",
            metadata={"hnsw:space": "cosine"}
        )

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
        """Fetch single embedding (delegates to batch embedding)"""
        # Clean text
        text = text.strip()
        if not text:
            return []

        # Fetch from batch endpoint
        vectors = self._get_embeddings_batch([text], input_type=input_type)
        return vectors[0] if vectors else []

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

        all_chunks_with_sources = []
        for file_path in self.documents_dir.iterdir():
            if file_path.suffix == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for chunk in self._chunk_text(f.read()):
                            all_chunks_with_sources.append((chunk, file_path.name))
                except Exception as e:
                    print(f"[RAG] Error reading {file_path.name}: {e}")
            elif file_path.suffix == ".pdf":
                for chunk in self._chunk_text(self.extract_text_from_pdf(file_path)):
                    all_chunks_with_sources.append((chunk, file_path.name))

        # Deduplicate against existing ChromaDB entries
        try:
            existing_ids = set(self.collection.get()["ids"])
        except Exception:
            existing_ids = set()

        new_chunks = [
            (chunk, src) for chunk, src in all_chunks_with_sources
            if hashlib.sha256(chunk.encode()).hexdigest() not in existing_ids
        ]
        print(f"[RAG] {len(all_chunks_with_sources)} total chunks, {len(new_chunks)} new to embed.")

        new_count = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            texts = [c for c, _ in batch]
            sources = [s for _, s in batch]
            ids = [hashlib.sha256(c.encode()).hexdigest() for c in texts]
            try:
                vectors = self._get_embeddings_batch(texts, input_type="passage")
                self.collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=[{"source": s} for s in sources]
                )
                new_count += len(batch)
                print(f"[RAG] Embedded {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} chunks...")
            except Exception as e:
                print(f"[RAG] Batch {i // batch_size + 1} failed: {e}")

        return new_count

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve most relevant document chunks for the query using ChromaDB"""
        try:
            # Get embedding of the query using input_type='query'
            query_vector = self._get_embedding(query, input_type="query")
        except Exception as e:
            print(f"[RAG] Query embedding failed: {e}. Falling back to empty context.")
            return ""

        try:
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            # ChromaDB cosine space returns cosine distance = 1 - cosine_similarity
            # So similarity > 0.15 matches distance < 0.85
            top_results = []
            if results and "documents" in results and len(results["documents"]) > 0:
                documents = results["documents"][0]
                distances = results["distances"][0]
                for doc, dist in zip(documents, distances):
                    if dist < 0.85:  # matches cosine similarity > 0.15
                        top_results.append(doc)
            
            return "\n\n---\n\n".join(top_results)
        except Exception as e:
            print(f"[RAG] ChromaDB query failed: {e}. Falling back to empty context.")
            return ""
