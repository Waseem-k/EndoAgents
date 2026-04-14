"""
agents/judge.py  — Judge Agent (Day 10-11)
──────────────────────────────────────────────────────────────────
Evaluates a synthesised caption on three dimensions:
  1. Groundedness  — every claim traceable to vision output or RAG
  2. Completeness  — all required sections present
  3. Clinical Consistency — findings align with known patterns

TODO (Day 10-11):
    - Implement three-dimension scoring
    - Implement structured feedback generation
    - Implement feedback loop trigger logic
"""

from loguru import logger


class JudgeAgent:
    """Judge Agent — Day 10-11 implementation placeholder."""

    def __init__(self) -> None:
        logger.info("JudgeAgent initialised (stub — full impl Day 10-11)")

    def evaluate(self, caption: str, context: dict) -> dict:
        raise NotImplementedError(
            "JudgeAgent.evaluate() — implementation scheduled for Day 10-11."
        )
