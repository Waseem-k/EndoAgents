"""
EndoAgents — Project-wide settings loaded from .env
All modules import from here: `from config.settings import settings`
"""

import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings  # pip install pydantic-settings


class Settings(BaseSettings):
    # ── Model ────────────────────────────────────────────────────────────────
    vision_model_id: str = "google/gemma-4-E4B-it"
    quantisation: str = "4bit"          # none | 4bit | 8bit
    visual_token_budget: int = 560
    max_new_tokens: int = 600

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_docs_dir: str = "rag/documents"
    chroma_db_dir: str = "rag/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    annotation_sheet_path: str = "data/EndoAgents_Annotation_Sheet_v1.1.xlsx"
    lora_rank: int = 16
    lora_alpha: int = 32
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_output_dir: str = "evaluation/results"
    judge_threshold: float = 0.7
    judge_max_loops: int = 2

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir: str = "logs"

    @field_validator("quantisation")
    @classmethod
    def validate_quantisation(cls, v: str) -> str:
        allowed = {"none", "4bit", "8bit"}
        if v not in allowed:
            raise ValueError(f"quantisation must be one of {allowed}")
        return v

    @field_validator("visual_token_budget")
    @classmethod
    def validate_token_budget(cls, v: int) -> int:
        allowed = {70, 140, 280, 560, 1120}
        if v not in allowed:
            raise ValueError(f"visual_token_budget must be one of {allowed}")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()

# Ensure key directories exist at import time
for _dir in [
    settings.rag_docs_dir,
    settings.chroma_db_dir,
    settings.eval_output_dir,
    settings.log_dir,
]:
    Path(_dir).mkdir(parents=True, exist_ok=True)
