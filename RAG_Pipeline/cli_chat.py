import os
import sys
import io

# Force standard output and input to UTF-8 to handle Unicode characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from services.rag_engine import RAGEngine
from services.map_tool import MapTool
from services.agent import ReceptionistAgent

def main():
    print("=" * 60)
    print("           RECEPTIONIST CLI INTERACTIVE CHAT")
    print("=" * 60)
    print("Initializing services, please wait...")
    
    rag = RAGEngine()
    map_tool = MapTool()
    agent = ReceptionistAgent(rag_engine=rag, map_tool=map_tool)
    
    # Ingest documents on start
    print("Scanning document directory...")
    new_chunks = rag.ingest_documents()
    if new_chunks > 0:
        print(f"Indexed {new_chunks} new document chunks successfully.")
    
    print("\nInitialization complete! Type 'exit' or 'quit' to end.")
    print("Ask about campus facilities, timings, directions, etc.")
    print("-" * 60)
    
    # Default bot coordinates
    bot_lat = config.DEFAULT_BOT_LAT
    bot_lon = config.DEFAULT_BOT_LON
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("\nThinking...")
            result = agent.query(user_input, bot_lat, bot_lon)
            
            print("\nReceptionist:")
            print(result["response"])
            
            # Print routing metadata if directions tool was called
            if result.get("route_meta"):
                route = result["route_meta"]
                print("\n📍 Navigation Route Summary:")
                print(f"  • Start: {route['start_name']}")
                print(f"  • Destination: {route['destination_name']}")
                print(f"  • Total Distance: {route['distance_m']} meters")
                print(f"  • Walking Time: {round(route['duration_s'] / 60, 1)} minutes")
                print("  • Path Steps:")
                for step in route["steps"]:
                    print(f"    - {step['instruction']} ({step['distance_m']}m)")
                    
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
