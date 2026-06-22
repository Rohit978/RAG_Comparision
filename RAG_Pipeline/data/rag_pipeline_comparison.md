# RAG Pipeline Comparison Report

We conducted a benchmark comparing the original **In-Memory JSON Cache RAG** with the new **ChromaDB-backed RAG** pipeline. The benchmark was run under identical conditions, querying the exact same event documents and utilizing the free OpenRouter embedding model (`nvidia/llama-nemotron-embed-vl-1b-v2:free`).

Below is the side-by-side analysis of performance, resource usage, latency, and retrieval accuracy.

---

## 1. Side-by-Side Metrics Table

| Metric | In-Memory RAG (Original) | ChromaDB RAG (New) | Measurement Methodology | Key Takeaway / Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Module Import Time** | **0.15 s** | **4.01 s** | Measured using `time.perf_counter()` before and after importing `RAGEngine` and `config`. | ChromaDB is significantly slower to import because it loads SQLite, `pydantic`, `onnxruntime`, `numpy`, and other database binaries. |
| **Engine Initialization** | **0.05 s** | **0.50 s** | Measured using `time.perf_counter()` around the `RAGEngine()` constructor call. | Original loads a single JSON file; ChromaDB establishes a connection to a local SQLite instance and HNSW index. |
| **Scan & Ingest (Cached)** | **0.001 s** | **0.07 s** | Measured using `time.perf_counter()` around the `ingest_documents()` function call. | Check for existing chunks is a local dictionary lookup vs. local SQLite index queries. Both are sub-100ms. |
| **Average Query Latency** | **1148.55 ms** | **956.78 ms** | Average duration of 5 distinct `retrieve_context()` queries measured using `time.perf_counter()`. | Both are dominated by the OpenRouter network API call (~900–1000ms) to embed the search query. ChromaDB's search is slightly faster. |
| **Python Allocated RAM** | **4.61 MB** | **45.50 MB** | Tracked via Python's built-in `tracemalloc` to calculate memory delta before and after execution. | ChromaDB consumes ~10x more memory allocations due to loading HNSW and database client libraries. |
| **Storage Footprint** | **1.10 MB** (`.json`) | **1.34 MB** (`.db`) | Calculated using `os.path.getsize()` on the cache files or persistent database directory. | Both files are compact. ChromaDB stores raw data in SQLite alongside indexing files. |

---

## 2. Retrieval Accuracy & Overlap

To check for parity in retrieval quality, we compared the exact document chunks returned by both pipelines across all 5 test queries. 

> [!NOTE]
> **Overlap Score: 100.0%** for all queries.
> The distance calculation in ChromaDB using `cosine` space ($d = 1 - S$, with $d < 0.85$) behaves exactly identical to the pure-Python cosine similarity check ($S > 0.15$), ensuring no regression in match quality.

### Query Matching Overlap Details:
- **"What is the early bird registration fee..."**: `100.0% overlap` (10/10 matching lines)
- **"When and where is Dr. Marcus Vance's keynote..."**: `100.0% overlap` (16/16 matching lines)
- **"What is the decryption key leaked..."**: `100.0% overlap` (16/16 matching lines)
- **"What events are scheduled for October 14th?"**: `100.0% overlap` (17/17 matching lines)
- **"Are attendees who do not have a Level 3 clearance..."**: `100.0% overlap` (16/16 matching lines)

---

## 3. Architecture Tradeoffs

### 🟢 In-Memory RAG (Original)
* **Pros:** Extremely lightweight, near-instant import and initialization, virtually zero overhead. Ideal for resource-constrained systems (like Raspberry Pi 5) running other background tasks.
* **Cons:** Hard to scale. If the document corpus grows to tens of thousands of chunks, the pure-Python cosine similarity loop ($O(N)$) will become a CPU bottleneck.

### 🔵 ChromaDB RAG (New)
* **Pros:** Highly scalable. Uses indexing (HNSW) to look up neighbors in $O(\log N)$ time. Provides database features such as metadata filtering and stable multi-process concurrency.
* **Cons:** Larger memory footprint (~45MB vs ~4.6MB) and slow imports due to the weight of dependencies.
