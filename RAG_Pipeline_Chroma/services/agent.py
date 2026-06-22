import urllib.request
import json
import ssl
import re
import config
from services.rag_engine import RAGEngine
from services.map_tool import MapTool

ssl_context = ssl._create_unverified_context()

class ReceptionistAgent:
    def __init__(self, rag_engine: RAGEngine, map_tool: MapTool):
        self.rag = rag_engine
        self.map = map_tool
        self.api_key = config.OPENROUTER_API_KEY
        self.chat_url = config.OPENROUTER_CHAT_URL
        self.model = config.LLM_MODEL

    def _call_llm(self, messages: list, tools: list = None) -> dict:
        """Call OpenRouter chat completions API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000"
        }

        req = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            response = urllib.request.urlopen(req, context=ssl_context, timeout=25)
            res_data = json.loads(response.read().decode("utf-8"))
            if "choices" in res_data and len(res_data["choices"]) > 0:
                return res_data["choices"][0]["message"]
            else:
                raise RuntimeError(f"OpenRouter empty response: {res_data}")
        except Exception as e:
            print(f"[AGENT] OpenRouter API call failed: {e}")
            if hasattr(e, "read"):
                print(f"[AGENT] API error details: {e.read().decode('utf-8')}")
            raise e

    def _get_map_tool_definition(self) -> list:
        """Return schema for the map routing tool"""
        return [{
            "type": "function",
            "function": {
                "name": "get_directions",
                "description": "Get precise walking directions between locations on campus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The target room, block, kiosk, or entrance (e.g. 'canteen', 'Academic Block 3', 'Central Library')."
                        },
                        "start": {
                            "type": "string",
                            "description": "The starting point. If not specified by the user, keep it null so we route from the bot's current location."
                        }
                    },
                    "required": ["destination"]
                }
            }
        }]

    def query(self, user_query: str, bot_lat: float, bot_lon: float) -> dict:
        """Process a user query, running RAG, OSRM mapping, and LLM orchestration"""
        # 1. Retrieve text context from local RAG engine
        rag_context = self.rag.retrieve_context(user_query, top_k=2)

        # 2. Formulate the System Prompt
        system_prompt = (
            "You are a friendly, helpful, and professional Receptionist. "
            "Your job is to greet visitors and answer questions about campus, schedules, events, and venues.\n\n"
            "=== CAMPUS INFORMATION (RAG) ===\n"
            f"{rag_context if rag_context else 'No specific event/policy documents found in RAG.'}\n\n"
            "=== OPERATIONAL INSTRUCTIONS ===\n"
            "1. Answer questions based on the RAG context. If you do not know the answer, politely say so. Do not make up info.\n"
            f"2. You are running physically on a receptionist kiosk. Your current coordinates are: Lat {bot_lat}, Lon {bot_lon}.\n"
            "3. If the user asks for directions or how to go somewhere, USE the 'get_directions' tool to calculate the route. "
            "Do NOT try to guess or describe coordinates yourself; call the tool to get the real path.\n"
            "4. Keep your responses concise and polite (ideal for Text-to-Speech voice responses, so avoid long lists or complex markdown tables in final speech output).\n"
            "5. If the retrieved context contains contradictory or conflicting information (e.g., an event is scheduled for a date/room but another update says it is rescheduled or cancelled), explicitly state these contradictions or updates to the user."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print(f"[AGENT] Processing query: '{user_query}'...")
        
        # 3. Call LLM with tool definitions enabled
        tools = self._get_map_tool_definition()
        response_msg = self._call_llm(messages, tools=tools)

        # 4. Check for Tool Call (Function Calling)
        tool_calls = response_msg.get("tool_calls")
        
        # Fallback check: in case tool call isn't natively structured but model writes it in text
        if not tool_calls and "get_directions" in response_msg.get("content", ""):
            # Try parsing a manual tool output if the model outputted text function call
            print("[AGENT] Native tool-call not found, checking text fallback...")
            match = re.search(r'get_directions\((.*?)\)', response_msg.get("content", ""))
            if match:
                try:
                    # Mock tool call structure
                    params_str = match.group(1)
                    dest_match = re.search(r'destination=["\'](.*?)["\']', params_str)
                    if dest_match:
                        tool_calls = [{
                            "id": "call_fallback",
                            "type": "function",
                            "function": {
                                "name": "get_directions",
                                "arguments": json.dumps({
                                    "destination": dest_match.group(1),
                                    "start": None
                                })
                            }
                        }]
                except Exception as e:
                    print(f"[AGENT] Text fallback parsing failed: {e}")

        # If a tool call is triggered
        if tool_calls:
            tool_call = tool_calls[0]
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            print(f"[AGENT] Tool Triggered: {func_name}({arguments})")
            
            if func_name == "get_directions":
                dest_query = arguments.get("destination")
                start_query = arguments.get("start")
                
                # Resolve destination coordinates
                end_landmark = self.map.resolve_landmark(dest_query)
                if not end_landmark:
                    return {
                        "response": f"I'd love to guide you, but I couldn't find the location of '{dest_query}' in our campus records.",
                        "route_meta": None
                    }

                # Resolve start coordinates
                start_lat, start_lon = bot_lat, bot_lon
                start_name = "Reception Bot Location"
                
                if start_query:
                    start_landmark = self.map.resolve_landmark(start_query)
                    if start_landmark:
                        start_lat = start_landmark["lat"]
                        start_lon = start_landmark["lon"]
                        start_name = start_landmark["name"]
                    else:
                        print(f"[AGENT] Warning: Start landmark '{start_query}' not resolved. Defaulting to bot coordinates.")

                # Fetch walking route from OSRM
                route_result = self.map.get_route(
                    start_lat=start_lat, start_lon=start_lon,
                    end_lat=end_landmark["lat"], end_lon=end_landmark["lon"]
                )

                if not route_result or not route_result.get("success"):
                    err_msg = route_result.get("error", "Routing error")
                    return {
                        "response": f"I resolved the coordinates for {end_landmark['name']}, but OSRM routing failed: {err_msg}.",
                        "route_meta": None
                    }

                # 5. Send route back to LLM for conversational narration
                print("[AGENT] Routing successful. Prompting LLM to narrate directions...")
                
                # Append assistant tool request and tool response to chat history
                messages.append(response_msg)
                
                tool_response_text = (
                    f"OSRM Route computed successfully:\n"
                    f"Start: {start_name} (Lat: {start_lat}, Lon: {start_lon})\n"
                    f"Destination: {end_landmark['name']} (Lat: {end_landmark['lat']}, Lon: {end_landmark['lon']})\n"
                    f"Distance: {route_result['distance_m']} meters\n"
                    f"Steps:\n" + "\n".join(
                        f"- {step['instruction']} ({step['distance_m']}m)" 
                        for step in route_result["steps"]
                    )
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "call_fallback"),
                    "name": "get_directions",
                    "content": tool_response_text
                })

                # Call LLM again to get natural language narration
                final_response = self._call_llm(messages)
                
                return {
                    "response": final_response.get("content"),
                    "route_meta": {
                        "start_name": start_name,
                        "destination_name": end_landmark["name"],
                        "distance_m": route_result["distance_m"],
                        "duration_s": route_result["duration_s"],
                        "steps": route_result["steps"]
                    }
                }

        # Normal Response (No tools triggered)
        return {
            "response": response_msg.get("content", ""),
            "route_meta": None
        }
