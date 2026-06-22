import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import config
from services.rag_engine import RAGEngine
from services.map_tool import MapTool
from services.agent import ReceptionistAgent

app = FastAPI(
    title="Receptionist Core Backend",
    description="Core Backend with RAG Knowledge and OSRM Navigation Tools"
)

# Initialize Services
rag = RAGEngine()
map_tool = MapTool()
agent = ReceptionistAgent(rag_engine=rag, map_tool=map_tool)

# Run Document Ingestion on startup
@app.on_event("startup")
def startup_event():
    print("[SERVER] Starting Receptionist Core Backend...")
    print("[SERVER] Running initial document indexing...")
    try:
        new_chunks = rag.ingest_documents()
        print(f"[SERVER] Document indexing finished. Added {new_chunks} new chunks.")
    except Exception as e:
        print(f"[SERVER] Startup document indexing warning: {e}")

# Request and Response schemas
class QueryRequest(BaseModel):
    text_query: str
    bot_lat: Optional[float] = None
    bot_lon: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    route_meta: Optional[dict] = None

@app.post("/query", response_model=QueryResponse)
def handle_query(payload: QueryRequest):
    query_text = payload.text_query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    # Use request coordinates if supplied, otherwise fallback to configured default coordinates
    lat = payload.bot_lat if payload.bot_lat is not None else config.DEFAULT_BOT_LAT
    lon = payload.bot_lon if payload.bot_lon is not None else config.DEFAULT_BOT_LON
    
    try:
        result = agent.query(
            user_query=query_text,
            bot_lat=lat,
            bot_lon=lon
        )
        return QueryResponse(
            response=result["response"],
            route_meta=result["route_meta"]
        )
    except Exception as e:
        print(f"[SERVER] Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Agent Error: {str(e)}")

@app.post("/reindex")
def trigger_reindex():
    """Endpoint to trigger re-indexing of PDFs and text files inside data/documents/"""
    try:
        new_chunks = rag.ingest_documents()
        return {
            "status": "success",
            "message": f"Ingestion complete. Indexed {new_chunks} new document chunks."
        }
    except Exception as e:
        print(f"[SERVER] Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"[SERVER] Starting API server on {config.HOST}:{config.PORT}...")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
