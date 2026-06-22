# Receptionist Core Backend: RAG Pipeline (NumPy / In-Memory)

This folder contains the lightweight, in-memory NumPy implementation of the receptionist RAG backend.

## Performance Metrics
To profile and measure the performance metrics of this RAG engine (including import speed, initialization latency, RAM utilization, query latency, and storage footprint), run the profiler:

```bash
# Navigate to this folder
cd RAG_Pipeline

# Run the profiler
python profile_pipeline.py
```

## Running the Chat CLI
To interactively chat and test queries with this engine:
```bash
python cli_chat.py
```
