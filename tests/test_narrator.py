import pytest
from PIL import Image
from agents.narrator import NarratorAgent
from models.vision import CaptionResult

class MockVisionModel:
    def generate_caption(self, image, user_prompt, **kwargs):
        # Return a perfect fake response to test the parser
        fake_caption = """
        1. IMAGE TYPE: Transvaginal ultrasound, sagittal view.
        2. UTERINE MORPHOLOGY: Globular and enlarged.
        3. MYOMETRIAL ASSESSMENT: Heterogeneous echogenicity.
        4. JUNCTIONAL ZONE: Poorly visualised and irregular.
        5. ENDOMETRIUM: Regular, 5mm thick.
        6. VISIBLE ANNOTATIONS: Calipers measuring the posterior wall.
        7. IMPRESSION: Findings highly suggestive of diffuse adenomyosis.
        """
        return CaptionResult(
            caption=fake_caption, model_id="mock", 
            inference_time_s=0.1, approx_tokens=50, 
            quantisation="none", visual_token_budget=560
        )

def test_narrator_parsing():
    dummy_img = Image.new("RGB", (100, 100))
    agent = NarratorAgent(vision_model=MockVisionModel())
    
    result = agent.run(dummy_img)
    
    assert "image_type" in result.sections
    assert "Transvaginal" in result.sections["image_type"]
    assert len(result.missing_sections) == 0