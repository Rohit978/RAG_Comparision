import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PLACES_JSON_PATH = DATA_DIR / "places.json"
EMBEDDING_CACHE_PATH = DATA_DIR / "embedding_cache.json"
MATRIX_CACHE_PATH = DATA_DIR / "vectors_matrix.npy"
CHUNKS_CACHE_PATH = DATA_DIR / "chunks_list.json"
ENV_PATH = BASE_DIR / ".env"

# Zero-dependency, lightweight .env loader
def load_env(env_file):
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        # Set environment variables if not already set externally
                        os.environ.setdefault(key.strip(), val.strip())
        except Exception as e:
            print(f"[CONFIG] Warning: Failed to parse .env file: {e}")

# Load variables
load_env(ENV_PATH)

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# API Configurations
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

# Modular Model Selection
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b:free")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")

# Geocoding & Campus Customization
DEFAULT_CAMPUS_CITY = os.environ.get("DEFAULT_CAMPUS_CITY", "Noida")

# Fallback Bot Coordinates
DEFAULT_BOT_LAT = float(os.environ.get("DEFAULT_BOT_LAT", "28.61460"))
DEFAULT_BOT_LON = float(os.environ.get("DEFAULT_BOT_LON", "77.35820"))

# OSM & OSRM API URLs
OSRM_FOOT_URL = "https://router.project-osrm.org/route/v1/foot"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RohitReceptionist/1.0 (contact@example.com)"

# Server Config
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
