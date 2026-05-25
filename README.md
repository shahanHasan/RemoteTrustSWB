# EMSCAD Traditional ML Fraud Detector (CPU-Friendly)

Traditional ML benchmark + soft-voting ensemble for fake job detection on EMSCAD, with recall-first thresholding and optional LIME local explanations.
The live scoring flow now also includes a rules-based trust and quality layer.

## What is implemented

- Feature contract: `combined_text` (joined text fields) + numeric flags (`telecommuting`, `has_company_logo`, `has_questions`)
- Candidate models:
  - `logreg_tfidf`
  - `sgd_log_tfidf`
  - `linsvc_cal_tfidf`
  - `cnb_bow`
  - `voting_soft`
- Validation-time threshold selection for recall-first operation (`target_recall=0.90` default)
- Champion selection across candidates: recall constraint met, then precision, then PR-AUC
- Artifacts:
  - `model.joblib`
  - `config.json`
  - `metrics.json` (includes per-model validation/test sections)
  - `model_comparison.csv`
  - `confusion_matrix_test.png`
  - `example_explanations.json`

## Project layout

- `src/job_fraud_detector/train.py`: training orchestration and artifact generation
- `src/job_fraud_detector/modeling.py`: model registry, pipelines, grids, thresholding, metrics
- `src/job_fraud_detector/inference.py`: `FraudDetector` inference + optional LIME explanations
- `src/job_fraud_detector/live_sources.py`: fetch/normalize live jobs from public sources
- `src/job_fraud_detector/rules.py`: trust + quality heuristic scoring engine
- `scripts/train_model.py`: training wrapper
- `scripts/score_live_sources.py`: live-source scoring CLI
- `notebooks/live_demo.ipynb`: notebook demo for local/live scoring and explanations
  - Includes trust/quality/final score formula breakdown tables and hard-cap trigger inspection
- `docs/scam_job_data_sources.md`: reference list of external APIs/datasets (not used in training)
- `docs/modeling_references.md`: literature and official-doc links used for model selection rationale
- `docs/heuristics_rulebook.md`: full rule catalog and score semantics

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
PYTHONPATH=src python -m job_fraud_detector.train \
  --data-path /Users/shahan/Downloads/fake_job_postings.csv \
  --artifacts-dir artifacts/emscad_light_model \
  --seed 42 \
  --target-recall 0.90 \
  --test-size 0.15 \
  --val-size 0.15
```

### Training flags

Core:
- `--data-path`
- `--artifacts-dir`
- `--seed`
- `--target-recall`
- `--test-size`
- `--val-size`

Model selection:
- `--candidate-models` (comma-separated IDs)
- `--disable-voting`
- `--c-grid` (optional LogisticRegression C override)

Reporting/runtime:
- `--plot-confusion-matrix` / `--no-plot-confusion-matrix`
- `--max-rows` (smoke runs)
- `--lime-num-samples`

## Inference API

```python
from job_fraud_detector.inference import FraudDetector

detector = FraudDetector.from_artifacts("artifacts/emscad_light_model")
result = detector.predict_posting(posting_dict, with_explanation=True)
```

Returns:
- `fraud_probability`
- `prediction`
- `threshold`
- optional `lime_explanation`

You can force a specific trained candidate at inference time:

```python
voting_detector = FraudDetector.from_artifacts(
    "artifacts/emscad_light_model",
    model_name="voting_soft",
)
```

## Score live public sources

```bash
export USAJOBS_API_KEY="your-usajobs-key"
export USAJOBS_USER_AGENT="you@example.com"
PYTHONPATH=src python scripts/score_live_sources.py \
  --artifacts-dir artifacts/emscad_light_model \
  --per-source 5 \
  --model-name voting_soft
```

`USAJOBS_USER_AGENT` should be the same email registered for your USAJOBS API key.

Useful options:
- `--prefer-voting`
- `--with-explanations`
- `--no-heuristics` (ML-only)
- `--disable-i18n` (skip language detection + translation normalization)
- `--fail-fast-fetch` (surface provider/API errors)

### Internationalization in live scoring

Live scoring now performs language normalization before inference:
- detect posting language (`langdetect` + script heuristic fallback)
- if confidently non-English, attempt translation to English (`deep-translator` Google backend)
- if translation fails, apply fairness softening so non-English posts are not over-penalized by English-only signals

This metadata is included in outputs (`detected_language`, `translation_applied`, `translation_error`).

## Gradio job board dashboard

Run the interactive hackathon dashboard locally:

```bash
export USAJOBS_API_KEY="your-usajobs-key"
export USAJOBS_USER_AGENT="you@example.com"
export ARTIFACTS_DIR="artifacts/emscad_light_model"
export MODEL_NAME="voting_soft"
python app.py
```

What it includes:
- live jobs from 4 sources (We Work Remotely, Jobicy, Remotive, USAJOBS)
- trust, quality, and final opportunity scores (color coded)
- score explanations with hard-cap reasons and positive/risk evidence
- pagination and filters for source, badge, risk level, search, and sorting

### Deploy to Hugging Face Spaces (Gradio)

1. Create a new Space with **SDK: Gradio**.
2. Push this repo to the Space (root contains `app.py` entrypoint).
3. In Space settings, add secrets:
   - `USAJOBS_API_KEY`
   - `USAJOBS_USER_AGENT`
   - optional: `ARTIFACTS_DIR` (default is `artifacts/emscad_light_model`)
   - optional: `MODEL_NAME` (default is `voting_soft`)
4. Ensure model artifacts are present in the repo at `artifacts/emscad_light_model/`.
5. Restart the Space.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

Set custom dataset path for integration tests if needed:

```bash
export EMSCAD_DATA_PATH=/path/to/fake_job_postings.csv
```
