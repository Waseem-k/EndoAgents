"""
orchestrator.py — EndoAgents Pipeline Orchestrator (Day 8-9)
─────────────────────────────────────────────────────────────────────────────
Coordinates the full agent pipeline:
  GemmaVision → NarratorAgent → RAGAgent → CaptionSynthesiser → JudgeAgent

Stub for now. Full implementation in Day 8-9 once all agents are built.

Architecture:
    Image
      │
      ▼
  [GemmaVision]          ← Pre-trained / fine-tuned Gemma 4
      │  raw caption
      ▼
  [NarratorAgent]        ← Enforces 7-section format + self-reflects
      │  structured sections + confidence scores
      ├──────────────────────────────────────────┐
      ▼                                          ▼
  [RAGAgent]                              (parallel)
  Retrieves ISUOG/MUSA passages
      │  relevant clinical context
      └──────────────────────────────────────────┘
                          │
                          ▼
              [CaptionSynthesiser]
              Confidence-weighted merge
                          │ draft caption
                          ▼
                    [JudgeAgent]
              Groundedness / Completeness /
              Clinical Consistency scoring
                          │
              ┌───────────┴────────────┐
              │ pass                   │ fail (≤2 retries)
              ▼                        ▼
        Final Caption          Structured feedback
                                → back to Synthesiser
"""

from loguru import logger


class EndoAgentsOrchestrator:
    """Full pipeline orchestrator — Day 8-9 implementation placeholder."""

    def __init__(self) -> None:
        logger.info("EndoAgentsOrchestrator initialised (stub — full impl Day 8-9)")

    def run(self, image, pathology_class: str = "Adenomyosis") -> dict:
        raise NotImplementedError(
            "EndoAgentsOrchestrator.run() — implementation scheduled for Day 8-9."
        )
