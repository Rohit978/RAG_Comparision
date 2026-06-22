import os
import sys
import io
import time
import tracemalloc
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure config modules can be resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_clean_import_time():
    """Runs a separate process to time importing the RAG engine without caching."""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import time; t0=time.perf_counter(); import config, services.rag_engine; print(round(time.perf_counter()-t0, 3))"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    return float(proc.stdout.strip()) if proc.returncode == 0 else 0.0

def main():
    print("Running performance benchmarks...")
    import_time = get_clean_import_time()
    
    # Track RAM usage changes during load
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()
    
    # Time engine startup
    t0 = time.perf_counter()
    import config
    from services.rag_engine import RAGEngine
    rag = RAGEngine()
    init_time = time.perf_counter() - t0
    
    # Peak heap allocation
    peak_mem, _ = tracemalloc.get_traced_memory()
    ram_usage_mb = (peak_mem - mem_start) / (1024 * 1024)
    
    # Time cached ingestion scan
    t0 = time.perf_counter()
    rag.ingest_documents()
    ingest_time = time.perf_counter() - t0
    
    # Time query retrieval latency
    test_queries = [
        "What is the early bird registration fee?",
        "When is Dr. Marcus Vance keynote session?",
        "What events are scheduled for October 14th?",
        "Are Level 3 clearance required for the server room?",
        "What is the registration deadline?"
    ]
    
    query_times = []
    for query in test_queries:
        t_start = time.perf_counter()
        rag.retrieve_context(query, top_k=2)
        query_times.append(time.perf_counter() - t_start)
        
    avg_latency_ms = (sum(query_times) / len(query_times)) * 1000
    
    # Calculate storage footprint of active indexes/caches
    total_bytes = 0
    cache_json = Path(config.DATA_DIR) / "embedding_cache.json"
    matrix_npy = Path(config.DATA_DIR) / "vectors_matrix.npy"
    chroma_db_dir = getattr(config, "CHROMA_DB_PATH", None)
    
    if cache_json.exists():
        total_bytes += cache_json.stat().st_size
    if matrix_npy.exists():
        total_bytes += matrix_npy.stat().st_size
    if chroma_db_dir and Path(chroma_db_dir).exists():
        for file in Path(chroma_db_dir).rglob("*"):
            if file.is_file():
                total_bytes += file.stat().st_size
                
    storage_mb = total_bytes / (1024 * 1024)
    tracemalloc.stop()
    
    # Log results
    print("\n--- Performance Metrics ---")
    print(f"Module Import:         {import_time:.2f}s")
    print(f"Engine Init:           {init_time:.2f}s")
    print(f"Scan & Ingest (Cache): {ingest_time:.3f}s")
    print(f"Avg Query Latency:     {avg_latency_ms:.2f}ms")
    print(f"Allocated RAM:         {ram_usage_mb:.2f}MB")
    print(f"Storage Size:          {storage_mb:.2f}MB")
    print("-" * 27)

if __name__ == "__main__":
    main()
