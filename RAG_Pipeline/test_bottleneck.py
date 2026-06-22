import os
import sys
import io

# Force standard output to UTF-8 to handle Unicode characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Append RAG_Pipeline path to import our packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_engine import RAGEngine
from services.map_tool import MapTool
from services.agent import ReceptionistAgent

def test_query(agent, query_text, rag):
    print(f"\nQUERY: \"{query_text}\"")
    # Coordinates of default bot position (JSS Noida main gate)
    bot_lat = 28.61460
    bot_lon = 77.35820
    context = rag.retrieve_context(query_text, top_k=2)
    print(f"RETRIEVED CONTEXT:\n{context}\n")
    res = agent.query(query_text, bot_lat, bot_lon)
    print("RESPONSE:")
    print(res["response"])
    print("-" * 60)

def main():
    print("=== STARTING RAG PIPELINE BOTTLENECK TESTS ===")
    
    # Initialize Engines
    rag = RAGEngine()
    map_tool = MapTool()
    agent = ReceptionistAgent(rag_engine=rag, map_tool=map_tool)
    
    # Ingest documents
    print("\nIngesting documents...")
    rag.ingest_documents()
    
    # Run the tests
    print("\n[Test 1: Chunk Boundary Splitting]")
    test_query(agent, "What is the early bird registration fee for the Aegis Prime Symposium and what is the exact deadline date for it?", rag)
    
    print("\n[Test 2: Temporal Contradictions]")
    test_query(agent, "When and where is Dr. Marcus Vance's keynote presentation scheduled, and is it cancelled?", rag)
    
    print("\n[Test 3: Anaphoric Pronoun / Entity Reference]")
    test_query(agent, "What is the decryption key leaked by Dr. Marcus Vance during the afternoon session?", rag)
    
    print("\n[Test 4: Acronym Collision]")
    test_query(agent, "Explain the two different systems referred to as C.O.L.D.S.T.A.R. in the documents.", rag)
    
    print("\n[Test 5: Unicode Word Splitting]")
    test_query(agent, "What events are scheduled for October 14th?", rag)

    print("\n[Test 6: Logical Double Negations]")
    test_query(agent, "Are attendees who do not have a Level 3 clearance permitted or prohibited from entering the server room?", rag)

    print("\n[Test 7: Noisy Log Retrieval]")
    test_query(agent, "What thermal temperature sensor issue was reported for Sat-01 telemetry in the system diagnostic logs?", rag)

    print("\n[Test 8: Decryption of Base64 Passcode]")
    test_query(agent, "What is the plaintext secret passkey for the VIP lounge?", rag)

    print("=== BOTTLENECK TESTS COMPLETED ===")

if __name__ == "__main__":
    main()
