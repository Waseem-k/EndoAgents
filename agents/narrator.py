"""
agents/narrator.py  — Radiological Narrator Agent (Day 3-4)
─────────────────────────────────────────────────────────────
Wraps GemmaVision, enforces the 7-section structured caption
format, and produces a per-section confidence score via
self-reflection.

TODO (Day 3-4):
    - Implement section parser (split raw caption into 7 sections)
    - Implement self-reflection pass (confidence per section)
    - Implement retry logic if a section is missing
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger


@dataclass
class NarratorOutput:
    raw_caption: str
    sections: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    missing_sections: list = field(default_factory=list)
    reflection_note: str = ""

    @property
    def structured_caption(self) -> str:
        if not self.sections:
            return self.raw_caption
        return "\n\n".join(
            f"[{k.upper()}]\n{v}" for k, v in self.sections.items()
        )


class NarratorAgent:
    """Radiological Narrator Agent — Day 3-4 implementation placeholder."""

    REQUIRED_SECTIONS = [
        "image_type",
        "uterine_morphology",
        "myometrial_assessment",
        "junctional_zone",
        "endometrium",
        "annotations",
        "impression",
    ]

    def __init__(self, vision_model=None) -> None:
        self.vision_model = vision_model
        logger.info("NarratorAgent initialised (stub — full impl Day 3-4)")

    def run(self, image, pathology_class: str = "Adenomyosis") -> NarratorOutput:
        """
        Generate and structure a caption for the given image.
        Full implementation in Day 3-4.
        """
        raise NotImplementedError(
            "NarratorAgent.run() — implementation scheduled for Day 3-4. "
            "Use models/vision.py GemmaVision directly for Day 1-2 baseline."
        )
