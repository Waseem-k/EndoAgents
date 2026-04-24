"""
agents/judge.py  — Judge Agent (Day 10-11)
──────────────────────────────────────────────────────────────────
Evaluates a synthesised caption on three dimensions:
  1. Completeness  — uses evaluation/feature_audit.py (Deterministic)
  2. Groundedness  — uses TruLens (Semantic)
  3. Clinical Consistency — uses TruLens (Semantic)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from loguru import logger

# Import your custom feature audit script
try:
    from evaluation.feature_audit import audit_caption, AuditResult
except ImportError:
    logger.warning("Could not import feature_audit. Ensure you are running from the project root.")

# Import TruLens for standalone evaluation
try:
    from trulens.providers.litellm import LiteLLM
    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    logger.warning("trulens-eval not installed. Semantic judging will use fallback logic.")


@dataclass
class JudgeOutput:
    passed: bool
    feedback: str
    scores: Dict[str, float] = field(default_factory=dict)
    
    @property
    def needs_retry(self) -> bool:
        return not self.passed


class JudgeAgent:
    """Judge Agent for validating the synthesised clinical caption."""

    def __init__(self, threshold: float = 0.7) -> None:
        """
        Args:
            threshold: Minimum score required to pass TruLens semantic checks (0.0 to 1.0).
        """
        self.threshold = threshold
        
        if TRULENS_AVAILABLE:
            # We initialize the provider. LiteLLM acts as a router. 
            # You can set OPENAI_API_KEY in your .env, or map it to a local endpoint.
            self.llm_provider = LiteLLM(model_engine="gpt-4o-mini") 
            logger.info(f"JudgeAgent initialised with TruLens provider (threshold={self.threshold})")
        else:
            self.llm_provider = None
            logger.info("JudgeAgent initialised (Completeness check ONLY).")

    def run(
        self, 
        draft_caption: str, 
        pathology_class: str, 
        narrator_output: Any, 
        rag_context: List[str]
    ) -> JudgeOutput:
        """
        Evaluate the draft caption.
        """
        logger.info("JudgeAgent evaluation initiated.")
        scores = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Deterministic Completeness Check (Fast Fail)
        # ─────────────────────────────────────────────────────────────────────
        audit: AuditResult = audit_caption(draft_caption, pathology_class)
        scores["completeness_pct"] = audit.coverage_pct / 100.0  # scale to 0-1
        
        if not audit.passed:
            missing_str = ", ".join(audit.missing)
            logger.warning(f"Judge failed: Missing required sections -> {missing_str}")
            feedback = (
                f"Your previous caption missed mandatory sections for {pathology_class}. "
                f"You MUST include the following missing clinical features: {missing_str}."
            )
            return JudgeOutput(passed=False, feedback=feedback, scores=scores)

        logger.success("Completeness check passed. Proceeding to semantic evaluation.")

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Semantic TruLens Evaluation (Groundedness & Consistency)
        # ─────────────────────────────────────────────────────────────────────
        if not TRULENS_AVAILABLE:
            logger.info("Skipping TruLens checks. Caption passes by default based on format.")
            return JudgeOutput(passed=True, feedback="", scores=scores)

        # 2a. Groundedness (Are the claims in the caption supported by the visual findings?)
        visual_context = narrator_output.raw_caption 
        groundedness_score, groundedness_reasons = self.llm_provider.groundedness_measure_with_cot_reasons(
            source=visual_context, 
            statement=draft_caption
        )
        scores["groundedness"] = groundedness_score

        # 2b. Clinical Consistency (Does the caption logically align with RAG guidelines?)
        guideline_context = "\n".join(rag_context) if rag_context else "No guidelines provided."
        consistency_score, consistency_reasons = self.llm_provider.relevance_with_cot_reasons(
            prompt=guideline_context, 
            response=draft_caption
        )
        scores["consistency"] = consistency_score

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Threshold Decision & Feedback Generation
        # ─────────────────────────────────────────────────────────────────────
        failed_dimensions = []
        feedback_lines = []

        if groundedness_score < self.threshold:
            failed_dimensions.append("Groundedness")
            feedback_lines.append(f"- Visual Grounding Error: {groundedness_reasons}")
            
        if consistency_score < self.threshold:
            failed_dimensions.append("Clinical Consistency")
            feedback_lines.append(f"- Guideline Violation: {consistency_reasons}")

        if failed_dimensions:
            logger.warning(f"Judge failed on semantic checks: {failed_dimensions}")
            feedback = (
                "Your draft failed semantic clinical validation. Please fix the following:\n" +
                "\n".join(feedback_lines)
            )
            return JudgeOutput(passed=False, feedback=feedback, scores=scores)

        logger.success("All Judge evaluation dimensions passed!")
        return JudgeOutput(
            passed=True, 
            feedback="Perfect. No changes needed.", 
            scores=scores
        )