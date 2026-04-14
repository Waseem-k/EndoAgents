# EndoAgents 🔬

**Multi-agent NLP pipeline for adenomyosis ultrasound captioning**  
OVGU × CS Collaboration — Dr. med. Paolo Gennari × Student Team

---

## Research Question

> Does multi-agent NLP deliberation produce clinically more complete and accurate captions than a single VLM given the same ultrasound image?

---

## Architecture

```
Anonymised Ultrasound Image (Sagittal TVUS)
            │
            ▼
    [GemmaVision]              ← Gemma 4-E4B, fine-tuned on OVGU dataset
            │  raw caption
            ▼
    [NarratorAgent]            ← Enforces 7-section MUSA format + self-reflects
            │  structured sections + per-section confidence
            ├─────────────────────────────────────────┐
            ▼                                         ▼
    [RAGAgent]                               (parallel dispatch)
    Retrieves ISUOG/MUSA guideline
    passages from ChromaDB
            │  clinical context
            └─────────────────────────────────────────┘
                              │
                              ▼
                  [CaptionSynthesiser]
                  Confidence-weighted merge
                              │ draft caption
                              ▼
                        [JudgeAgent]
                Groundedness / Completeness /
                Clinical Consistency scoring
                              │
               ┌──────────────┴───────────────┐
               │ pass (≥0.7 on all dims)       │ fail
               ▼                               ▼
         Final Caption                 Structured feedback
                                       → Synthesiser (max 2×)
```

**Three-class output:** Adenomyosis | Fibroid | Normal

---

## Project Structure

```
EndoAgents/
├── agents/
│   ├── narrator.py          # Radiological Narrator Agent (Day 3-4)
│   ├── rag_agent.py         # Medical Literature RAG Agent (Day 5-7)
│   ├── synthesiser.py       # Caption Synthesiser (Day 8-9)
│   └── judge.py             # Judge Agent (Day 10-11)
├── models/
│   └── vision.py            # Gemma 4 inference wrapper ✅
├── rag/
│   ├── documents/           # Place ISUOG / MUSA PDFs here
│   ├── ingest.py            # Ingest PDFs → ChromaDB (Day 5)
│   └── retriever.py         # Query + summarise (Day 6-7)
├── evaluation/
│   ├── feature_audit.py     # 11-feature coverage checker ✅
│   ├── metrics.py           # ROUGE-L, BERTScore, CIDEr (Day 12-13)
│   └── trulens_eval.py      # TruLens triad (Day 12-13)
├── data/
│   ├── loader.py            # Excel annotation sheet → splits ✅
│   └── finetune_config.py   # LoRA config for Gemma 4 (Day 14)
├── notebooks/
│   ├── 01_baseline_gemma4.ipynb   # Zero-shot baseline ✅
│   └── 02_finetune.ipynb          # LoRA fine-tuning (Day 14)
├── config/
│   └── settings.py          # Pydantic settings from .env ✅
├── orchestrator.py           # Full pipeline (Day 8-9)
├── requirements.txt          # ✅
├── .env.example              # ✅  → copy to .env and fill in
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/EndoAgents.git
cd EndoAgents
pip install -r requirements.txt
cp .env.example .env          # edit .env with your settings
```

**Gemma 4 requires a HuggingFace account and model licence acceptance:**  
→ https://huggingface.co/google/gemma-4-E4B-it

---

## Development Schedule (2-week sprint before dataset arrives)

| Days | Task | Status |
|---|---|---|
| 1–2 | Gemma 4 inference wrapper + zero-shot baseline | ✅ Done |
| 3–4 | Narrator Agent — 7-section parser + self-reflection | 🔲 |
| 5–7 | RAG Agent — ChromaDB ingest + retrieval | 🔲 |
| 8–9 | Orchestrator + Caption Synthesiser | 🔲 |
| 10–11 | Judge Agent — 3-dimension scoring + feedback loop | 🔲 |
| 12–13 | Evaluation harness — ROUGE-L, BERTScore, CIDEr, TruLens | 🔲 |
| 14 | Data loader + LoRA fine-tuning config (ready to run) | 🔲 |

---

## Dataset

Provided by Dr. med. Paolo Gennari, OVGU Medical Campus.  
Expected: ~2 weeks (pending ethics committee approval).

- **Target:** 150–300 annotated sagittal TVUS image-caption pairs
- **Classes:** Adenomyosis (50%) / Fibroid (25%) / Normal (25%)
- **Format:** Structured 7-section captions per MUSA criteria
- **Fine-tuning:** LoRA (rank=16) on `google/gemma-4-E4B-it`

---

## Evaluation

| Metric | Tool | Target (post fine-tune) |
|---|---|---|
| ROUGE-L | rouge-score | > 0.45 |
| BERTScore | bert-score | > 0.85 |
| CIDEr | pycocoevalcap | > 1.0 |
| Groundedness | TruLens | > 0.7 |
| Feature Coverage | feature_audit.py | 100% |
| Human (Likert) | Dr. Gennari's team | ≥ 4/5 |

---

## Citation

If you use this work, please cite:  
> EndoAgents: Multi-Agent NLP Pipeline for Adenomyosis Ultrasound Captioning.  
> Khan W., Shaikh M.A., Vhankhade S.D. — OVGU, 2025.
