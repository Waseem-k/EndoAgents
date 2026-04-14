"""
data/loader.py
─────────────────────────────────────────────────────────────────────────────
Reads the completed EndoAgents annotation Excel sheet and produces
train / val / test splits ready for LoRA fine-tuning.

Usage (once dataset is received from Dr. Gennari):
    from data.loader import AnnotationDataset
    dataset = AnnotationDataset("data/EndoAgents_Annotation_Sheet_v1.1.xlsx")
    train, val, test = dataset.get_splits()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from config.settings import settings


REQUIRED_COLUMNS = [
    "Image ID",
    "Image Filename",
    "Pathology Class",
    "Diagnosis Impression",
    "Annotator Notes",   # contains the full narrative caption
]

PATHOLOGY_CLASSES = {"Adenomyosis", "Fibroid", "Normal"}


@dataclass
class AnnotationSample:
    image_id: str
    filename: str
    pathology_class: str
    impression: str
    caption: str                         # full narrative caption from annotator
    # Optional structured fields
    uterine_shape: str = ""
    echogenicity: str = ""
    echogenic_islands: str = ""
    junctional_zone: str = ""
    confidence: str = ""
    image_path: Optional[str] = None    # resolved at load time if images exist


@dataclass
class DatasetSplit:
    samples: List[AnnotationSample]
    split_name: str

    def __len__(self) -> int:
        return len(self.samples)

    def class_distribution(self) -> dict:
        dist = {}
        for s in self.samples:
            dist[s.pathology_class] = dist.get(s.pathology_class, 0) + 1
        return dist

    def __repr__(self) -> str:
        return (
            f"DatasetSplit(split={self.split_name}, "
            f"n={len(self)}, "
            f"classes={self.class_distribution()})"
        )


class AnnotationDataset:
    """
    Loads and splits the EndoAgents annotation sheet.

    Parameters
    ----------
    sheet_path : str
        Path to the Excel annotation sheet.
    images_dir : str, optional
        Directory containing the ultrasound images.
        If provided, image_path is resolved for each sample.
    train_ratio : float
        Proportion for training set. Default from settings.
    val_ratio : float
        Proportion for validation set.
    test_ratio : float
        Proportion for test set.
    random_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        sheet_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        random_seed: int = 42,
    ) -> None:
        self.sheet_path  = sheet_path or settings.annotation_sheet_path
        self.images_dir  = images_dir
        self.train_ratio = train_ratio or settings.split_train
        self.val_ratio   = val_ratio   or settings.split_val
        self.test_ratio  = test_ratio  or settings.split_test
        self.seed        = random_seed

        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "train + val + test ratios must sum to 1.0"

        self._samples: List[AnnotationSample] = []
        self._loaded = False

    def load(self) -> "AnnotationDataset":
        """Read the Excel sheet and parse all samples."""
        path = Path(self.sheet_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Annotation sheet not found: {path}\n"
                "Waiting for dataset from Dr. Gennari — expected in ~2 weeks."
            )

        logger.info(f"Loading annotation sheet: {path}")
        df = pd.read_excel(path, sheet_name="Annotations", header=3)
        # Row 4 in Excel = index 0 after header=3 (0-indexed after 4 header rows)
        df.columns = df.columns.str.strip()

        # Drop empty rows
        df = df.dropna(subset=["Image ID", "Pathology Class"])

        # Validate required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")

        # Validate pathology classes
        invalid = set(df["Pathology Class"].unique()) - PATHOLOGY_CLASSES - {None, float("nan")}
        if invalid:
            logger.warning(f"Unexpected Pathology Class values: {invalid}")

        samples = []
        for _, row in df.iterrows():
            pc = str(row.get("Pathology Class", "")).strip()
            if pc not in PATHOLOGY_CLASSES:
                continue

            filename = str(row.get("Image Filename", "")).strip()
            image_path = None
            if self.images_dir and filename:
                candidate = Path(self.images_dir) / filename
                if candidate.exists():
                    image_path = str(candidate)

            sample = AnnotationSample(
                image_id       = str(row.get("Image ID", "")).strip(),
                filename       = filename,
                pathology_class= pc,
                impression     = str(row.get("Diagnosis Impression", "")).strip(),
                caption        = str(row.get("Annotator Notes", "")).strip(),
                uterine_shape  = str(row.get("Uterine Shape", "")).strip(),
                echogenicity   = str(row.get("Myometrial Echogenicity", "")).strip(),
                echogenic_islands = str(row.get("Echogenic Islands", "")).strip(),
                junctional_zone= str(row.get("Junctional Zone", "")).strip(),
                confidence     = str(row.get("Confidence Level", "")).strip(),
                image_path     = image_path,
            )
            samples.append(sample)

        self._samples = samples
        self._loaded  = True

        logger.success(
            f"Loaded {len(samples)} samples — "
            f"{self._class_counts()}"
        )
        return self

    def _class_counts(self) -> str:
        counts = {}
        for s in self._samples:
            counts[s.pathology_class] = counts.get(s.pathology_class, 0) + 1
        return " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    def get_splits(
        self,
    ) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Stratified train / val / test split by Pathology Class.

        Returns
        -------
        (train, val, test) DatasetSplit objects
        """
        if not self._loaded:
            self.load()

        if len(self._samples) == 0:
            raise ValueError("No valid samples found in the annotation sheet.")

        labels = [s.pathology_class for s in self._samples]

        # First split: train vs (val + test)
        train_s, temp_s, train_l, temp_l = train_test_split(
            self._samples, labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=labels,
            random_state=self.seed,
        )

        # Second split: val vs test (from the temp set)
        val_ratio_adj = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_s, test_s = train_test_split(
            temp_s,
            test_size=(1 - val_ratio_adj),
            stratify=temp_l,
            random_state=self.seed,
        )

        train = DatasetSplit(train_s, "train")
        val   = DatasetSplit(val_s,   "val")
        test  = DatasetSplit(test_s,  "test")

        logger.info(
            f"Splits — train: {len(train)} | val: {len(val)} | test: {len(test)}"
        )
        return train, val, test

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        status = f"{len(self._samples)} samples" if self._loaded else "not loaded"
        return f"AnnotationDataset(path={self.sheet_path}, {status})"
