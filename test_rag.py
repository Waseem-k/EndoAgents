"""
Quick test script for the RAG agent.
Run: python test_rag.py
"""

from agents.rag_agent import RAGAgent

agent = RAGAgent()

print("\nTest 1: General query")
result = agent.retrieve("junctional zone thickness adenomyosis")
print(f"Query: {result.query}")
print(f"Found: {len(result.passages)} passages\n")
for i, (p, s, sc) in enumerate(zip(result.passages, result.sources, result.relevance_scores), 1):
    print(f"[{i}] Source: {s} | Score: {sc:.3f}")
    print(p[:300])
    print()

print("\nTest 2: Retrieve by pathology - Adenomyosis")
result2 = agent.retrieve_by_pathology("Adenomyosis")
print(f"Found: {len(result2.passages)} passages\n")
for i, (p, s, sc) in enumerate(zip(result2.passages, result2.sources, result2.relevance_scores), 1):
    print(f"[{i}] Source: {s} | Score: {sc:.3f}")
    print(p[:300])
    print()

print("\nTest 3: Section-specific retrieval - myometrial_assessment")
result3 = agent.retrieve_for_section("myometrial_assessment", "heterogeneous myometrium with cystic spaces")
print(f"Found: {len(result3.passages)} passages\n")
for i, (p, s, sc) in enumerate(zip(result3.passages, result3.sources, result3.relevance_scores), 1):
    print(f"[{i}] Source: {s} | Score: {sc:.3f}")
    print(p[:300])
    print()

print("\nTest 4: Context block (what gets injected into prompts)")
print(result.context_block[:800])
