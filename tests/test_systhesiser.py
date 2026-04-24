import pytest
from PIL import Image

from agents.synthesiser import CaptionSynthesiser
from agents.narrator import NarratorOutput
from models.vision import CaptionResult

# --- Mocks ---

class MockVisionModelSuccess:
    """Mocks a successful Gemma 4 response."""
    def generate_caption(self, image, system_prompt, user_prompt, **kwargs):
        fake_synthesis = (
            "1. IMAGE TYPE: Transvaginal ultrasound.\n"
            "2. UTERINE MORPHOLOGY: Globular shape.\n"
            "3. MYOMETRIAL ASSESSMENT: Heterogeneous echogenicity.\n"
            "4. JUNCTIONAL ZONE: Poorly visualised.\n"
            "5. ENDOMETRIUM: Regular.\n"
            "6. VISIBLE ANNOTATIONS: None.\n"
            "7. IMPRESSION: Findings suggestive of adenomyosis based on MUSA criteria."
        )
        return CaptionResult(
            caption=fake_synthesis, model_id="mock", 
            inference_time_s=0.1, approx_tokens=50, 
            quantisation="none", visual_token_budget=560
        )

class MockVisionModelFailure:
    """Mocks a failed Gemma 4 response (e.g., OOM error)."""
    def generate_caption(self, image, system_prompt, user_prompt, **kwargs):
        return CaptionResult(
            caption="", model_id="mock", 
            inference_time_s=0.1, approx_tokens=0, 
            quantisation="none", visual_token_budget=560,
            error="CUDA out of memory"
        )

# --- Fixtures ---

@pytest.fixture
def dummy_image():
    """Provides a fast, blank PIL image for testing."""
    return Image.new("RGB", (100, 100))

@pytest.fixture
def mock_narrator_output():
    """Simulates the input from the NarratorAgent."""
    return NarratorOutput(
        raw_caption="Raw output",
        sections={
            "image_type": "Transvaginal ultrasound.",
            "uterine_morphology": "Globular shape.",
            "echogenic_islands": "Some bright spots seen."
        },
        confidence={
            "image_type": 0.95,
            "uterine_morphology": 0.80,
            "echogenic_islands": 0.40  # Low confidence
        }
    )

# --- Tests ---

def test_synthesiser_success(dummy_image, mock_narrator_output):
    """Tests the standard synthesis path."""
    agent = CaptionSynthesiser(vision_model=MockVisionModelSuccess())
    mock_rag = ["MUSA Criteria: Echogenic islands should be described as discrete hyperechoic foci."]
    
    result = agent.run(
        image=dummy_image, 
        narrator_output=mock_narrator_output, 
        rag_context=mock_rag
    )
    
    assert result.synthesis_notes == "Synthesis complete."
    assert "MUSA criteria" in result.draft_caption
    assert "1. IMAGE TYPE" in result.draft_caption

def test_synthesiser_fallback(dummy_image, mock_narrator_output):
    """Tests the fallback logic if the vision model crashes."""
    agent = CaptionSynthesiser(vision_model=MockVisionModelFailure())
    mock_rag = []
    
    result = agent.run(
        image=dummy_image, 
        narrator_output=mock_narrator_output, 
        rag_context=mock_rag
    )
    
    # It should catch the failure and fallback to the structured narrator output
    assert "Falling back" in result.synthesis_notes
    assert "IMAGE TYPE:\nTransvaginal ultrasound." in result.draft_caption