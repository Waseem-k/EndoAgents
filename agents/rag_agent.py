"""
agents/rag_agent.py  — Medical Literature RAG Agent (Day 5-7)
─────────────────────────────────────────────────────────────────
Retrieves relevant passages from ISUOG guidelines and MUSA criteria
stored in ChromaDB. Grounds the Narrator's language in established
clinical terminology.

TODO (Day 5-7):
    - Implement ChromaDB retrieval
    - Implement query construction from Narrator output
    - Implement self-reflection (relevance scoring)
    - Implement summarisation of retrieved passages
"""

from loguru import logger


class RAGAgent:
    """Medical Literature RAG Agent — Day 5-7 implementation placeholder."""

    def __init__(self) -> None:
        logger.info("RAGAgent initialised (stub — full impl Day 5-7)")

    def retrieve(self, query: str, top_k: int = 4) -> list:
        raise NotImplementedError(
            "RAGAgent.retrieve() — implementation scheduled for Day 5-7."
        )

    def run(self, narrator_output) -> dict:
        raise NotImplementedError(
            "RAGAgent.run() — implementation scheduled for Day 5-7."
        )
