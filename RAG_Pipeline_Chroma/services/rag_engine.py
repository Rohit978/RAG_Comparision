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

    def _get_embedding(self, text: str, input_type: str = "passage") -> list:
        """Fetch embedding from OpenRouter using nvidia/llama-nemotron-embed-vl-1b-v2:free"""
        # Clean text
        text = text.strip()
        if not text:
            return []

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment.")

        # Request payload
        payload = {
            "model": self.model,
            "input": text,
            "input_type": input_type  # 'passage' or 'query'
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
            response = urllib.request.urlopen(req, context=ssl_context, timeout=20)
            res_data = json.loads(response.read().decode("utf-8"))
            if "data" in res_data and len(res_data["data"]) > 0:
                vector = res_data["data"][0]["embedding"]
                return vector
            else:
                raise RuntimeError(f"OpenRouter embedding error: {res_data}")
        except Exception as e:
            print(f"[RAG] Error fetching embedding: {e}")
            if hasattr(e, "read"):
                print(f"[RAG] Error details: {e.read().decode('utf-8')}")
            raise e

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

    def ingest_documents(self) -> int:
        """Re-scan the documents folder and index new document chunks in ChromaDB"""
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return 0
        
        all_chunks_with_sources = []
        for file_path in self.documents_dir.iterdir():
            if file_path.suffix == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = self._chunk_text(content)
                        for chunk in chunks:
                            all_chunks_with_sources.append((chunk, file_path.name))
                except Exception as e:
                    print(f"[RAG] Error reading txt file {file_path.name}: {e}")
            elif file_path.suffix == ".pdf":
                pdf_content = self.extract_text_from_pdf(file_path)
                chunks = self._chunk_text(pdf_content)
                for chunk in chunks:
                    all_chunks_with_sources.append((chunk, file_path.name))

        # Get existing IDs from collection to avoid duplicate embeds
        try:
            existing_results = self.collection.get()
            existing_ids = set(existing_results["ids"]) if existing_results and "ids" in existing_results else set()
        except Exception as e:
            print(f"[RAG] Error fetching existing IDs from ChromaDB: {e}")
            existing_ids = set()

        print(f"[RAG] Found {len(all_chunks_with_sources)} chunks in total. Ingesting new chunks...")
        new_count = 0
        for chunk, filename in all_chunks_with_sources:
            # Generate a stable ID based on chunk hash
            chunk_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_id not in existing_ids:
                try:
                    embedding = self._get_embedding(chunk, input_type="passage")
                    if embedding:
                        self.collection.add(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{"source": filename}]
                        )
                        new_count += 1
                        # Add to local existing_ids set so we don't duplicate within same loop
                        existing_ids.add(chunk_id)
                except Exception as e:
                    print(f"[RAG] Failed to index chunk from {filename}: {e}")
        
        print(f"[RAG] Finished ingestion. Added {new_count} new chunks to ChromaDB.")
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
