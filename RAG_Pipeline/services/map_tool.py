import urllib.request
import urllib.parse
import json
from pathlib import Path
import config

class MapTool:
    def __init__(self):
        self.places_path = config.PLACES_JSON_PATH
        self.osrm_url = config.OSRM_FOOT_URL
        self.nominatim_url = config.NOMINATIM_URL
        self.user_agent = config.USER_AGENT
        self.places = self._load_places()

    def _load_places(self) -> dict:
        """Load landmarks registry from JSON file"""
        if self.places_path.exists():
            try:
                with open(self.places_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MAP] Error loading places JSON: {e}")
        return {}

    def resolve_landmark(self, query: str) -> dict:
        """Resolve query to a landmark coordinate from places.json or Nominatim"""
        if not query:
            return None

        clean_query = query.strip().lower().replace(" ", "")

        # 1. Exact/Cleaned Match in places.json
        for key, details in self.places.items():
            if key.replace(" ", "") == clean_query:
                return {
                    "name": details["name"],
                    "lat": details["lat"],
                    "lon": details["lon"],
                    "source": "local"
                }

        # 2. Substring Match in places.json
        for key, details in self.places.items():
            if clean_query in key.replace(" ", "") or key.replace(" ", "") in clean_query:
                return {
                    "name": details["name"],
                    "lat": details["lat"],
                    "lon": details["lon"],
                    "source": "local_fuzzy"
                }

        # 3. Fallback: Query Nominatim (OpenStreetMap Search)
        print(f"[MAP] Landmark '{query}' not found locally. Querying Nominatim...")
        try:
            url = f"{self.nominatim_url}?q={urllib.parse.quote(query + ', ' + config.DEFAULT_CAMPUS_CITY)}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            response = urllib.request.urlopen(req, timeout=10)
            data = json.loads(response.read().decode("utf-8"))
            if data:
                short_name = data[0]["display_name"].split(",")[0]
                return {
                    "name": short_name,
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"]),
                    "source": "nominatim"
                }
        except Exception as e:
            print(f"[MAP] Nominatim failed: {e}")

        return None

    def get_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> dict:
        """Calculate walking route between coordinates using OSRM"""
        url = f"{self.osrm_url}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&steps=true"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        
        try:
            response = urllib.request.urlopen(req, timeout=10)
            data = json.loads(response.read().decode("utf-8"))
            
            if data.get("code") == "Ok":
                route = data["routes"][0]
                distance = route["distance"]
                duration = route["duration"]
                steps = route["legs"][0]["steps"]
                
                formatted_steps = []
                for step in steps:
                    maneuver = step.get("maneuver", {})
                    instruction = maneuver.get("instruction")
                    
                    if not instruction:
                        m_type = maneuver.get("type", "go")
                        m_mod = maneuver.get("modifier", "")
                        instruction = f"{m_type} {m_mod}".strip()
                        
                    step_dist = step.get("distance", 0)
                    formatted_steps.append({
                        "instruction": instruction,
                        "distance_m": round(step_dist, 1)
                    })
                
                return {
                    "success": True,
                    "distance_m": round(distance, 1),
                    "duration_s": round(duration, 1),
                    "steps": formatted_steps
                }
            else:
                return {"success": False, "error": f"OSRM routing code: {data.get('code')}"}
        except Exception as e:
            return {"success": False, "error": f"OSRM API request failed: {e}"}
