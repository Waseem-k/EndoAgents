"""
models/vision.py
─────────────────────────────────────────────────────────────────────────────
Gemma 4 vision-language model wrapper for EndoAgents.

Handles:
  - Model loading with optional 4-bit / 8-bit quantisation
  - Correct Gemma 4 chat template formatting
  - Configurable visual token budget
  - Structured caption generation

Usage:
    from models.vision import GemmaVision
    model = GemmaVision()
    result = model.generate_caption(image)
    print(result.caption)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
#  Adenomyosis-specific system prompt
#  This is the primary clinical knowledge injected into every inference call.
#  Refine this collaboratively with Dr. Gennari as annotation progresses.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert gynaecological sonographer and clinician \
specialising in adenomyosis diagnosis via transvaginal ultrasound (TVUS). \
You use the MUSA (Morphological Uterus Sonographic Assessment) criteria as \
your diagnostic framework.

When analysing a uterine ultrasound image, you produce a structured clinical \
caption covering ALL of the following sections. If a feature is not clearly \
visible, state "not clearly visualised" — never omit a section.

Required sections:
1. IMAGE TYPE — modality and view plane
2. UTERINE MORPHOLOGY — shape (globular/normal/irregular), size, orientation
3. MYOMETRIAL ASSESSMENT
   - Echogenicity: homogeneous / heterogeneous (grade if possible)
   - Echogenic islands: present/absent — location if present
   - Myometrial cysts: present/absent — location if present
   - Asymmetric thickening: which wall, or symmetric
   - Subendometrial echogenic lines/nodules: present/absent
   - For fibroid cases: mass border (well-defined/ill-defined), \
capsule presence, vascularity pattern (peripheral-circular vs diffuse-translesional)
4. JUNCTIONAL ZONE — well-defined / irregular / poorly visualised / disrupted
5. ENDOMETRIUM — appearance and regularity
6. VISIBLE ANNOTATIONS — describe any measurement markers, arrows, or \
coloured lines and their clinical relevance
7. IMPRESSION — findings consistent with / suggestive of / not indicative of \
adenomyosis or fibroid. Specify focal or diffuse if determinable. Use \
appropriate uncertainty language.

Use precise sonographic terminology throughout."""

CAPTION_USER_PROMPT = (
    "Please generate a complete structured clinical caption for this uterine "
    "ultrasound image following the 7-section format specified."
)


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CaptionResult:
    caption: str
    model_id: str
    inference_time_s: float
    approx_tokens: int
    quantisation: str
    visual_token_budget: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        return (
            f"[{self.model_id}] "
            f"time={self.inference_time_s:.1f}s | "
            f"~tokens={self.approx_tokens} | "
            f"quant={self.quantisation}\n\n"
            f"{self.caption}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  GemmaVision
# ─────────────────────────────────────────────────────────────────────────────
class GemmaVision:
    """
    Wrapper around Gemma 4 for adenomyosis ultrasound captioning.

    Parameters
    ----------
    model_id : str, optional
        HuggingFace model repo. Defaults to settings.vision_model_id.
    quantisation : str, optional
        "none" | "4bit" | "8bit". Defaults to settings.quantisation.
    visual_token_budget : int, optional
        One of {70, 140, 280, 560, 1120}. Defaults to settings.visual_token_budget.
        Higher = more image detail captured, more VRAM used.
        Recommended: 560 for ultrasound (dense texture).
    device : str, optional
        "cuda" | "cpu". Auto-detected if not specified.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        quantisation: Optional[str] = None,
        visual_token_budget: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or settings.vision_model_id
        self.quantisation = quantisation or settings.quantisation
        self.visual_token_budget = visual_token_budget or settings.visual_token_budget
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None
        self._loaded = False

        logger.info(f"GemmaVision initialised — model={self.model_id} | "
                    f"quant={self.quantisation} | device={self.device} | "
                    f"token_budget={self.visual_token_budget}")

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def load(self) -> "GemmaVision":
        """Load model and processor. Called automatically on first generate call."""
        if self._loaded:
            return self

        logger.info(f"Loading {self.model_id} ...")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Quantisation config
        quant_cfg = None
        if self.quantisation == "4bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("  → 4-bit NF4 quantisation enabled")
        elif self.quantisation == "8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("  → 8-bit quantisation enabled")

        # Processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Model
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quant_cfg,
            trust_remote_code=True,
            attn_implementation="eager",  # flash_attention_2 if supported
        )
        self._model.eval()

        n_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        logger.success(f"Model loaded — {n_params:.2f}B parameters")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  VRAM: {vram_used:.2f} / {vram_total:.1f} GB")

        self._loaded = True
        return self

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded from memory")

    # ── Core generation ───────────────────────────────────────────────────────
    def generate_caption(
        self,
        image: Image.Image,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> CaptionResult:
        """
        Generate a structured adenomyosis caption for a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            The ultrasound image to analyse.
        system_prompt : str, optional
            Override the default adenomyosis system prompt.
        user_prompt : str, optional
            Override the default user instruction.
        max_new_tokens : int, optional
            Override settings.max_new_tokens.

        Returns
        -------
        CaptionResult
            Dataclass with caption text, timing, and metadata.
        """
        if not self._loaded:
            self.load()

        _sys  = system_prompt or SYSTEM_PROMPT
        _usr  = user_prompt   or CAPTION_USER_PROMPT
        _max  = max_new_tokens or settings.max_new_tokens

        t0 = time.time()

        try:
            caption = self._run_inference(image, _sys, _usr, _max)
            elapsed = time.time() - t0
            return CaptionResult(
                caption=caption.strip(),
                model_id=self.model_id,
                inference_time_s=round(elapsed, 2),
                approx_tokens=len(caption.split()),
                quantisation=self.quantisation,
                visual_token_budget=self.visual_token_budget,
            )

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"Caption generation failed: {exc}")
            return CaptionResult(
                caption="",
                model_id=self.model_id,
                inference_time_s=round(elapsed, 2),
                approx_tokens=0,
                quantisation=self.quantisation,
                visual_token_budget=self.visual_token_budget,
                error=str(exc),
            )

    def _run_inference(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Internal: build Gemma 4 chat format and run generation."""

        # Gemma 4 chat template:
        # - System role is supported natively
        # - Image MUST come before text in the user turn
        # - visual_token_budget controls how many image tokens are used
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # visual_token_budget is passed via processor kwargs below
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Apply chat template — produces a formatted prompt string
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenise.
        # visual_token_budget controls the approximate number of image tokens
        # used by the vision encoder (valid values: 70, 140, 280, 560, 1120).
        # The Gemma 4 processor derives token count from the input image
        # resolution — resize the image before this call if you need a tighter
        # budget than the processor default (typically 896×896 → ~560 tokens).
        inputs = self._processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens (strip the prompt)
        input_length = inputs["input_ids"].shape[-1]
        new_tokens   = output_ids[0][input_length:]
        caption      = self._processor.decode(new_tokens, skip_special_tokens=True)

        return caption

    # ── Context manager support ───────────────────────────────────────────────
    def __enter__(self) -> "GemmaVision":
        return self.load()

    def __exit__(self, *_) -> None:
        self.unload()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"GemmaVision(model={self.model_id}, "
            f"quant={self.quantisation}, "
            f"token_budget={self.visual_token_budget}, "
            f"status={status})"
        )
