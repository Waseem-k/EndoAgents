"""
agents/rag_agent.py — Medical Literature RAG Agent
Retrieves relevant passages from MUSA/ISUOG guidelines stored in ChromaDB
and grounds the Narrator's caption in established clinical terminology.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings
from rag.ingest import build_index, COLLECTION_NAME


@dataclass
class RAGOutput:
    query: str
    passages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    relevance_scores: list[float] = field(default_factory=list)
    summary: str = ""

    @property
    def context_block(self) -> str:
        """Formatted context block ready to inject into a prompt."""
        if not self.passages:
            return ""
        lines = ["CLINICAL GUIDELINE CONTEXT"]
        for i, (passage, source, score) in enumerate(
            zip(self.passages, self.sources, self.relevance_scores), 1
        ):
            lines.append(f"\n[{i}] Source: {source} (relevance: {score:.2f})")
            lines.append(passage.strip())
        lines.append("END CONTEXT")
        return "\n".join(lines)


class RAGAgent:
    """
    Retrieves MUSA guideline passages from ChromaDB given a clinical query.
    Builds the index on first use if it doesn't exist yet.
    """

    # MUSA sections mapped to focused sub-queries for better retrieval
    SECTION_QUERIES = {
        "image_type": "transvaginal ultrasound TVUS imaging technique acquisition",
        "uterine_morphology": "uterine size shape contour morphology measurement",
        "myometrial_assessment": "myometrium echogenicity heterogeneity asymmetry adenomyosis fibroid",
        "junctional_zone": "junctional zone thickness disruption regularity adenomyosis",
        "endometrium": "endometrium thickness echogenicity interface irregularity",
        "annotations": "cysts lacunae hyperechoic islands fan-shaped shadowing fibroid features",
        "impression": "adenomyosis fibroid classification diagnosis MUSA criteria",
    }

    def __init__(self) -> None:
        self._embedder: Optional[SentenceTransformer] = None
        self._collection: Optional[chromadb.Collection] = None
        self._client: Optional[chromadb.PersistentClient] = None
        logger.info("RAGAgent initialised (lazy-loads index on first query)")

    def _ensure_loaded(self) -> None:
        if self._collection is not None:
            return

        db_dir = Path(settings.chroma_db_dir)
        self._client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Build index if not present
        try:
            col = self._client.get_collection(COLLECTION_NAME)
            if col.count() == 0:
                raise ValueError("Empty collection")
            self._collection = col
            logger.info(f"RAGAgent: loaded existing index ({col.count()} chunks)")
        except Exception:
            logger.info("RAGAgent: index not found — building now...")
            self._collection = build_index(force=False)

        self._embedder = SentenceTransformer(settings.embedding_model)

    def retrieve(self, query: str, top_k: int | None = None) -> RAGOutput:
        """Retrieve top-k passages most relevant to the query."""
        self._ensure_loaded()
        k = top_k or settings.rag_top_k

        embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        passages = results["documents"][0]
        metadatas = results["metadatas"][0]
        # ChromaDB cosine distance → similarity score
        scores = [round(1 - d, 4) for d in results["distances"][0]]
        sources = [m.get("source", "unknown") for m in metadatas]

        # Filter low-relevance passages (below 0.25 similarity)
        filtered = [
            (p, s, sc) for p, s, sc in zip(passages, sources, scores) if sc >= 0.25
        ]
        if not filtered:
            logger.warning(f"RAGAgent: no relevant passages found for query: '{query[:80]}'")
            return RAGOutput(query=query)

        passages, sources, scores = zip(*filtered)
        logger.debug(f"RAGAgent: retrieved {len(passages)} passages (top score: {scores[0]:.3f})")

        return RAGOutput(
            query=query,
            passages=list(passages),
            sources=list(sources),
            relevance_scores=list(scores),
        )

    def retrieve_for_section(self, section: str, narrator_text: str = "") -> RAGOutput:
        """
        Retrieve passages for a specific MUSA section.
        Combines the section-specific sub-query with the narrator's own text.
        """
        base_query = self.SECTION_QUERIES.get(section, section)
        query = f"{base_query}. {narrator_text[:200]}" if narrator_text else base_query
        return self.retrieve(query)

    def run(self, narrator_output) -> dict[str, RAGOutput]:
        """
        Main entry point. Given a NarratorOutput, retrieves guideline context
        for each MUSA section and returns a dict keyed by section name.
        """
        self._ensure_loaded()
        sections = getattr(narrator_output, "sections", {})
        results: dict[str, RAGOutput] = {}

        if sections:
            for section, text in sections.items():
                results[section] = self.retrieve_for_section(section, narrator_text=text)
                logger.debug(f"RAGAgent: section '{section}' → {len(results[section].passages)} passages")
        else:
            # Fallback: retrieve using raw caption
            raw = getattr(narrator_output, "raw_caption", str(narrator_output))
            results["general"] = self.retrieve(raw[:500])

        logger.info(f"RAGAgent: completed retrieval for {len(results)} sections")
        return results

    def retrieve_by_pathology(self, pathology: str) -> RAGOutput:
        """Convenience method — retrieve guidelines specific to a pathology class."""
        queries = {
            "Adenomyosis": "adenomyosis MUSA criteria junctional zone myometrial features diagnosis",
            "Fibroid": "uterine fibroid leiomyoma classification location size echogenicity",
            "Normal": "normal uterus morphology MUSA sonographic appearance",
        }
        query = queries.get(pathology, f"{pathology} ultrasound diagnosis criteria")
        return self.retrieve(query, top_k=6)
