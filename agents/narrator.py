"""
agents/narrator.py  — Radiological Narrator Agent (Day 3-4)
─────────────────────────────────────────────────────────────
Wraps GemmaVision, enforces the 7-section structured caption
format, and produces a per-section confidence score via
self-reflection.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from loguru import logger
from PIL import Image

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
            f"{k.replace('_', ' ').upper()}:\n{v}" for k, v in self.sections.items()
        )


class NarratorAgent:
    """Radiological Narrator Agent for generating structured TVUS captions."""

    # Map internal keys to the expected headers in the LLM output
    SECTION_MAPPING = {
        "image_type": ["1.", "IMAGE TYPE"],
        "uterine_morphology": ["2.", "UTERINE MORPHOLOGY"],
        "myometrial_assessment": ["3.", "MYOMETRIAL ASSESSMENT"],
        "junctional_zone": ["4.", "JUNCTIONAL ZONE"],
        "endometrium": ["5.", "ENDOMETRIUM"],
        "annotations": ["6.", "VISIBLE ANNOTATIONS"],
        "impression": ["7.", "IMPRESSION"],
    }

    def __init__(self, vision_model, max_retries: int = 2) -> None:
        """
        Args:
            vision_model: Instantiated GemmaVision wrapper.
            max_retries: How many times to retry if sections are missing.
        """
        if vision_model is None:
            raise ValueError("NarratorAgent requires a loaded vision_model instance.")
        
        self.vision_model = vision_model
        self.max_retries = max_retries
        logger.info(f"NarratorAgent initialised with max_retries={self.max_retries}")

    def run(self, image: Image.Image, pathology_class: str = "Adenomyosis") -> NarratorOutput:
        """
        Generate and structure a caption for the given image, including self-reflection.
        """
        logger.info("NarratorAgent run initiated.")
        
        raw_caption = ""
        sections = {}
        missing_sections = list(self.SECTION_MAPPING.keys())
        
        user_prompt = (
            "Please generate a complete structured clinical caption for this uterine "
            "ultrasound image following the 7-section format specified."
        )

        # Step 1 & 2: Generation with Retry Logic
        for attempt in range(self.max_retries + 1):
            logger.debug(f"Caption generation attempt {attempt + 1}/{self.max_retries + 1}")
            result = self.vision_model.generate_caption(
                image=image, 
                user_prompt=user_prompt
            )
            
            if not result.success:
                logger.error(f"Vision model failed: {result.error}")
                break
                
            raw_caption = result.caption
            sections, missing_sections = self._parse_sections(raw_caption)
            
            if not missing_sections:
                logger.success("All required sections successfully generated and parsed.")
                break
                
            logger.warning(f"Missing sections detected: {missing_sections}")
            if attempt < self.max_retries:
                # Update user prompt to strictly request the missing sections
                user_prompt = (
                    "In your previous attempt, you missed the following required sections: "
                    f"{', '.join([s.replace('_', ' ').upper() for s in missing_sections])}. "
                    "Please generate the caption again, ensuring absolutely ALL 7 sections are included."
                )

        # Step 3: Self-Reflection Pass
        confidence_scores, reflection_note = self._reflect(image, sections)

        return NarratorOutput(
            raw_caption=raw_caption,
            sections=sections,
            confidence=confidence_scores,
            missing_sections=missing_sections,
            reflection_note=reflection_note
        )

    def _parse_sections(self, raw_text: str) -> tuple[Dict[str, str], List[str]]:
        """Splits the raw LLM output into the defined dictionary mapping."""
        sections = {}
        missing = []
        
        # Simple heuristic parsing: finding the headers and extracting text between them
        text_lower = raw_text.lower()
        
        # Identify indices of headers
        header_indices = {}
        for key, markers in self.SECTION_MAPPING.items():
            # Try to find the word marker first (e.g. "IMAGE TYPE")
            marker = markers[1].lower()
            idx = text_lower.find(marker)
            if idx != -1:
                header_indices[key] = idx
                
        # Sort headers by their appearance in the text
        sorted_headers = sorted(header_indices.items(), key=lambda x: x[1])
        
        # Extract text between headers
        for i, (key, start_idx) in enumerate(sorted_headers):
            # Find the end of the header string itself to start the content
            marker = self.SECTION_MAPPING[key][1].lower()
            content_start = start_idx + len(marker)
            
            # Find where the next section begins, or use end of string
            if i + 1 < len(sorted_headers):
                content_end = sorted_headers[i + 1][1]
            else:
                content_end = len(raw_text)
                
            # Clean up the extracted text (remove colons, dashes, leading/trailing whitespace)
            content = raw_text[content_start:content_end]
            content = re.sub(r"^[\s:\-]+", "", content) 
            sections[key] = content.strip()
            
        # Determine missing sections
        for key in self.SECTION_MAPPING.keys():
            if key not in sections or not sections[key]:
                missing.append(key)
                
        return sections, missing

    def _reflect(self, image: Image.Image, sections: Dict[str, str]) -> tuple[Dict[str, float], str]:
        """
        Asks the vision model to reflect on its own generated sections and provide a confidence score.
        """
        if not sections:
            return {}, "Reflection skipped due to empty sections."

        logger.info("Running self-reflection pass for confidence scoring...")
        
        # Format current findings for the prompt
        findings_text = "\n".join([f"- {k.replace('_', ' ').upper()}: {v}" for k, v in sections.items()])
        
        reflection_prompt = (
            "You are a senior sonographer reviewing an AI draft caption. "
            "Based on the provided ultrasound image and the drafted findings below, "
            "assign a confidence score from 0.1 to 1.0 for each section. "
            "Format your output EXACTLY as 'SECTION NAME: [Score]'. Do not add any other text.\n\n"
            f"Draft Findings:\n{findings_text}"
        )

        result = self.vision_model.generate_caption(
            image=image,
            user_prompt=reflection_prompt,
            max_new_tokens=150 # Keep it short, we only need scores
        )
        
        confidence_scores = {}
        reflection_note = ""
        
        if result.success:
            reflection_note = result.caption
            # Parse the scores (e.g. "IMAGE TYPE: 0.9")
            for key, markers in self.SECTION_MAPPING.items():
                header = markers[1]
                # Regex to find the header followed by a colon/space and a float
                match = re.search(rf"{header}.*?(0\.\d+|1\.0)", result.caption, re.IGNORECASE)
                if match:
                    try:
                        confidence_scores[key] = float(match.group(1))
                    except ValueError:
                        confidence_scores[key] = 0.5
                else:
                    confidence_scores[key] = 0.5 # Default fallback
        else:
            logger.warning("Self-reflection failed.")
            confidence_scores = {k: 0.5 for k in sections.keys()}
            reflection_note = "Failed to generate reflection."

        return confidence_scores, reflection_note