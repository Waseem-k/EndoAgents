import pytest
from agents.judge import JudgeAgent
from agents.narrator import NarratorOutput

# --- Mocks ---

class MockTruLensProvider:
    """Mocks the TruLens LiteLLM provider to prevent real API calls during tests."""
    def groundedness_measure_with_cot_reasons(self, source, statement):
        # Return a high score and dummy reason
        return 0.9, {"reason": "The statement perfectly aligns with the visual source."}
        
    def relevance_with_cot_reasons(self, prompt, response):
        # Return a high score and dummy reason
        return 0.85, {"reason": "The response perfectly follows the clinical guidelines."}


@pytest.fixture
def mock_narrator():
    return NarratorOutput(
        raw_caption="Visible globular shape and cysts.",
        sections={"uterine_morphology": "Globular shape."}
    )

# --- Tests ---

def test_judge_fails_completeness(mock_narrator):
    """Test that missing MUSA sections trigger an immediate fail and feedback loop."""
    agent = JudgeAgent(threshold=0.7)
    
    # Intentionally missing "Image Type", "Junctional Zone", etc.
    bad_draft = "2. UTERINE MORPHOLOGY: Globular shape.\n7. IMPRESSION: Adenomyosis."
    
    result = agent.run(
        draft_caption=bad_draft,
        pathology_class="Adenomyosis",
        narrator_output=mock_narrator,
        rag_context=["MUSA Guidelines..."]
    )
    
    assert result.passed is False
    assert result.needs_retry is True
    assert "Image Type" in result.feedback
    # TruLens is never called here because it fails the format check first

def test_judge_passes_format():
    """Test a perfectly formatted caption bypasses Completeness and passes Semantic checks."""
    agent = JudgeAgent(threshold=0.7)
    
    # Inject the mock provider to intercept the real API call
    agent.llm_provider = MockTruLensProvider()
    
    perfect_draft = (
        "1. IMAGE TYPE: TVUS.\n"
        "2. UTERINE MORPHOLOGY: Normal.\n"
        "3. MYOMETRIAL ASSESSMENT: Homogeneous.\n"
        "4. JUNCTIONAL ZONE: Regular.\n"
        "5. ENDOMETRIUM: Normal.\n"
        "6. VISIBLE ANNOTATIONS: None.\n"
        "7. IMPRESSION: Normal uterus."
    )
    
    result = agent.run(
        draft_caption=perfect_draft,
        pathology_class="Normal",
        narrator_output=NarratorOutput(raw_caption="Normal"),
        rag_context=["Guideline 1", "Guideline 2"]
    )
    
    assert result.passed is True
    assert result.scores["groundedness"] == 0.9
    assert result.scores["consistency"] == 0.85