import os
import sys
import io
import time
import tracemalloc
from pathlib import Path

# Force standard output to UTF-8 to handle Unicode characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def measure_import_time():
    """Run a quick python subprocess to measure import overhead cleanly without caching"""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import time; t0=time.perf_counter(); import config, services.rag_engine; print(round(time.perf_counter()-t0, 3))"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if proc.returncode == 0:
        return float(proc.stdout.strip())
    return 0.0

def main():
    print("=" * 60)
    print("           PIPELINE PROFILE & METRICS RUNNER")
    print("=" * 60)
    print("Measuring module import time...")
    import_time = measure_import_time()
    
    # Start tracing memory
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()
    
    # 1. Measure Engine Init
    t0 = time.perf_counter()
    import config
    from services.rag_engine import RAGEngine
    rag = RAGEngine()
    init_time = time.perf_counter() - t0
    
    # 2. Measure RAM Allocation
    mem_peak_initial, _ = tracemalloc.get_traced_memory()
    allocated_ram_mb = (mem_peak_initial - mem_start) / (1024 * 1024)
    
    # 3. Measure Scan & Ingest (Cached)
    t0 = time.perf_counter()
    rag.ingest_documents()
    ingest_time = time.perf_counter() - t0
    
    # 4. Measure Query Latency (Average of 5 queries)
    queries = [
        "What is the early bird registration fee?",
        "When is Dr. Marcus Vance keynote session?",
        "What events are scheduled for October 14th?",
        "Are Level 3 clearance required for the server room?",
        "What is the registration deadline?"
    ]
    
    latencies = []
    print("Running query benchmark (5 queries)...")
    for q in queries:
        t0 = time.perf_counter()
        rag.retrieve_context(q, top_k=2)
        latencies.append(time.perf_counter() - t0)
        
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    
    # 5. Measure Storage Footprint
    storage_bytes = 0
    cache_json = Path(config.DATA_DIR) / "embedding_cache.json"
    matrix_npy = Path(config.DATA_DIR) / "vectors_matrix.npy"
    
    # Try loading ChromaDB path if defined in config
    chroma_db_path = getattr(config, "CHROMA_DB_PATH", None)
    
    if cache_json.exists():
        storage_bytes += cache_json.stat().st_size
    if matrix_npy.exists():
        storage_bytes += matrix_npy.stat().st_size
    if chroma_db_path and Path(chroma_db_path).exists():
        chroma_db = Path(chroma_db_path)
        for f in chroma_db.rglob("*"):
            if f.is_file():
                storage_bytes += f.stat().st_size
                
    storage_mb = storage_bytes / (1024 * 1024)
    
    tracemalloc.stop()
    
    print("\n" + "=" * 60)
    print("                     METRICS SUMMARY")
    print("=" * 60)
    print(f" • Module Import Time:     {import_time:.2f} s")
    print(f" • Engine Initialization:  {init_time:.2f} s")
    print(f" • Scan & Ingest (Cached): {ingest_time:.3f} s")
    print(f" • Average Query Latency:  {avg_latency_ms:.2f} ms")
    print(f" • Python Allocated RAM:   {allocated_ram_mb:.2f} MB")
    print(f" • Storage Footprint:      {storage_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
