"""
agents/synthesiser.py  — Caption Synthesiser Agent (Day 8-9)
─────────────────────────────────────────────────────────────────
Merges the structured visual findings from the Narrator Agent with
the clinical guidelines retrieved by the RAG Agent. Applies a
confidence-weighted approach and incorporates Judge feedback 
during retry loops.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

from loguru import logger
from PIL import Image


@dataclass
class SynthesiserOutput:
    draft_caption: str
    synthesis_notes: str


class CaptionSynthesiser:
    """Caption Synthesiser Agent for merging Narrator and RAG outputs."""

    def __init__(self, vision_model: Any) -> None:
        """
        Args:
            vision_model: Instantiated GemmaVision wrapper. 
                          Reused from the main orchestrator to save VRAM.
        """
        if vision_model is None:
            raise ValueError("CaptionSynthesiser requires a loaded vision_model instance.")
        
        self.vision_model = vision_model
        logger.info("CaptionSynthesiser initialised.")

    def run(
        self, 
        image: Image.Image, 
        narrator_output: Any, 
        rag_context: List[str], 
        judge_feedback: Optional[str] = None
    ) -> SynthesiserOutput:
        """
        Synthesise the final clinical caption.
        
        Args:
            image: The original ultrasound image (passed to keep the VLM grounded).
            narrator_output: The NarratorOutput dataclass containing sections and confidence.
            rag_context: List of retrieved clinical guidelines strings from the RAGAgent.
            judge_feedback: Optional feedback from the JudgeAgent if this is a retry loop.
        """
        logger.info("CaptionSynthesiser run initiated.")

        # 1. Format the Narrator findings with confidence weights
        findings_str = self._format_narrator_findings(narrator_output)
        
        # 2. Format the RAG context
        context_str = "\n".join([f"- {c}" for c in rag_context]) if rag_context else "No external guidelines provided."

        # 3. Construct the Synthesiser Prompt
        system_instruction = (
            "You are an expert clinical summariser and sonographer. Your task is to write "
            "the final, cohesive 7-section ultrasound caption. "
            "You will be provided with initial visual findings (with confidence scores) and "
            "established clinical guidelines (ISUOG/MUSA)."
        )

        user_prompt = (
            "Please synthesize the final clinical caption based on the following inputs.\n\n"
            "INSTRUCTIONS:\n"
            "1. Maintain the strict 7-section format.\n"
            "2. If a visual finding has LOW confidence (< 0.6), rely heavily on the clinical guidelines "
            "to frame the uncertainty correctly (e.g., 'Findings are indeterminate, but suggestive of...').\n"
            "3. If a visual finding has HIGH confidence (>= 0.6), ensure its terminology aligns with the guidelines.\n\n"
            f"--- VISUAL FINDINGS ---\n{findings_str}\n\n"
            f"--- CLINICAL GUIDELINES ---\n{context_str}\n\n"
        )

        # 4. Inject Judge Feedback if this is a correction loop
        if judge_feedback:
            logger.warning("Applying Judge feedback for caption revision.")
            user_prompt += (
                f"--- CRITICAL REVISION FEEDBACK ---\n"
                f"Your previous draft failed the evaluation. You MUST address the following issues:\n"
                f"{judge_feedback}\n\n"
            )

        user_prompt += "Generate the final updated 7-section caption now. Do not include any conversational filler."

        # 5. Generate the synthesis
        # We pass the image alongside the prompt so the VLM retains its visual grounding
        result = self.vision_model.generate_caption(
            image=image,
            system_prompt=system_instruction,
            user_prompt=user_prompt,
            max_new_tokens=600
        )

        if not result.success:
            logger.error(f"Synthesiser generation failed: {result.error}")
            return SynthesiserOutput(
                draft_caption=narrator_output.structured_caption, # Fallback to narrator
                synthesis_notes="Model failed. Falling back to raw narrator output."
            )

        logger.success("Caption successfully synthesised.")
        return SynthesiserOutput(
            draft_caption=result.caption.strip(),
            synthesis_notes="Synthesis complete."
        )

    def _format_narrator_findings(self, narrator_output: Any) -> str:
        """Helper to format the sections and append their confidence scores for the prompt."""
        formatted_lines = []
        for section, text in narrator_output.sections.items():
            conf = narrator_output.confidence.get(section, 0.5)
            # Tag low confidence explicitly so the LLM pays attention
            conf_tag = f"[Confidence: {conf:.2f} - LOW]" if conf < 0.6 else f"[Confidence: {conf:.2f} - HIGH]"
            formatted_lines.append(f"{section.replace('_', ' ').upper()} {conf_tag}:\n{text}\n")
        
        return "\n".join(formatted_lines)