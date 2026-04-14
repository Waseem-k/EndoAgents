"""
evaluation/feature_audit.py
─────────────────────────────────────────────────────────────────────────────
Checks which of the 7 required adenomyosis caption sections are addressed
in a generated caption. Used by the Judge Agent and evaluation harness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


# ── Feature definitions ───────────────────────────────────────────────────────
# Each entry: (display_name, [keywords to detect], required_for_classes)
# required_for_classes: set of pathology classes this feature must appear in.
# "ALL" means it is required regardless of class.

FEATURE_DEFINITIONS: List[Dict] = [
    {
        "name": "Image Type / Modality",
        "keywords": ["transvaginal", "transabdominal", "tvus", "sagittal",
                     "coronal", "ultrasound", "view"],
        "required_for": "ALL",
    },
    {
        "name": "Uterine Morphology",
        "keywords": ["globular", "morpholog", "shape", "enlarg", "size",
                     "pear", "uterus", "configuration", "orientation"],
        "required_for": "ALL",
    },
    {
        "name": "Myometrial Echogenicity",
        "keywords": ["heterogeneous", "homogeneous", "echogenicity",
                     "texture", "myometri"],
        "required_for": "ALL",
    },
    {
        "name": "Echogenic Islands",
        "keywords": ["echogenic island", "hyperechoic", "bright focus",
                     "echogenic foc", "hyperechogenic"],
        "required_for": {"Adenomyosis"},
    },
    {
        "name": "Myometrial Cysts",
        "keywords": ["cyst", "anechoic", "fluid-filled", "myometrial cyst"],
        "required_for": {"Adenomyosis"},
    },
    {
        "name": "Asymmetric Thickening",
        "keywords": ["asymmetr", "thicken", "posterior wall", "anterior wall",
                     "thicker", "wall thickness"],
        "required_for": {"Adenomyosis"},
    },
    {
        "name": "Fibroid Mass Features",
        "keywords": ["well-defined", "ill-defined", "pseudocapsule", "capsule",
                     "peripheral", "circular vascular", "translesional",
                     "fibroid", "leiomyoma", "intramural", "subserosal",
                     "submucosal"],
        "required_for": {"Fibroid"},
    },
    {
        "name": "Junctional Zone",
        "keywords": ["junctional", "jz ", "j.z.", "endometrial-myometrial",
                     "junctional zone"],
        "required_for": "ALL",
    },
    {
        "name": "Endometrium",
        "keywords": ["endometri", "endometrium"],
        "required_for": "ALL",
    },
    {
        "name": "Annotations / Markers",
        "keywords": ["arrow", "caliper", "marker", "annotati", "measur",
                     "red", "green", "line"],
        "required_for": "ALL",
    },
    {
        "name": "Overall Impression",
        "keywords": ["consistent with", "suggestive of", "impression",
                     "adenomyosis", "fibroid", "normal uterus", "findings"],
        "required_for": "ALL",
    },
]


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class AuditResult:
    coverage: Dict[str, bool]            # feature_name → detected
    score: int                           # number of features detected
    total: int                           # total features checked
    missing: List[str]                   # feature names not detected
    coverage_pct: float                  # score / total * 100
    pathology_class: str = "Adenomyosis"

    @property
    def passed(self) -> bool:
        """True if all REQUIRED features for this class are covered."""
        return len(self.missing) == 0

    def summary(self) -> str:
        lines = [
            f"Feature Coverage: {self.score}/{self.total} "
            f"({self.coverage_pct:.0f}%) — class={self.pathology_class}",
        ]
        for name, detected in self.coverage.items():
            icon = "✅" if detected else "❌"
            lines.append(f"  {icon}  {name}")
        if self.missing:
            lines.append(f"\nMissing: {', '.join(self.missing)}")
        return "\n".join(lines)


# ── Main audit function ───────────────────────────────────────────────────────
def audit_caption(
    caption: str,
    pathology_class: str = "Adenomyosis",
) -> AuditResult:
    """
    Check which required features are present in the caption.

    Parameters
    ----------
    caption : str
        The generated caption text to audit.
    pathology_class : str
        One of "Adenomyosis" | "Fibroid" | "Normal".
        Determines which features are required vs. optional.

    Returns
    -------
    AuditResult
    """
    cap_lower = caption.lower()
    coverage  = {}
    missing   = []

    for feat in FEATURE_DEFINITIONS:
        name     = feat["name"]
        keywords = feat["keywords"]
        required = feat["required_for"]

        # Is this feature required for this pathology class?
        if required == "ALL":
            is_required = True
        elif isinstance(required, set):
            is_required = pathology_class in required
        else:
            is_required = False

        detected = any(kw in cap_lower for kw in keywords)
        coverage[name] = detected

        if is_required and not detected:
            missing.append(name)

    # Only count features relevant to this class
    relevant = [
        f for f in FEATURE_DEFINITIONS
        if f["required_for"] == "ALL"
        or (isinstance(f["required_for"], set) and pathology_class in f["required_for"])
    ]
    total = len(relevant)
    score = sum(coverage[f["name"]] for f in relevant)

    return AuditResult(
        coverage=coverage,
        score=score,
        total=total,
        missing=missing,
        coverage_pct=round(score / total * 100, 1) if total > 0 else 0.0,
        pathology_class=pathology_class,
    )
