#!/usr/bin/env python3
"""Fetch 5 jobs per live source and score them with the trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from job_fraud_detector.inference import FraudDetector
from job_fraud_detector.live_sources import SOURCES, fetch_jobs_from_sources, score_live_jobs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score live jobs from public remote-job feeds.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/emscad_light_model",
        help="Directory containing model.joblib and config.json",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=5,
        help="How many jobs to pull from each source.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Optional model id from artifacts (e.g. voting_soft, logreg_tfidf).",
    )
    parser.add_argument(
        "--prefer-voting",
        action="store_true",
        help="If set, load voting_soft when available in artifacts.",
    )
    parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="Include LIME explanations for each scored job.",
    )
    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Disable trust/quality heuristic layer and return ML-only scoring.",
    )
    parser.add_argument(
        "--disable-i18n",
        action="store_true",
        help="Disable language detection/translation normalization before scoring.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full scored results as JSON.",
    )
    parser.add_argument(
        "--fail-fast-fetch",
        action="store_true",
        help="Raise source fetch errors instead of skipping failed sources.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    detector = FraudDetector.from_artifacts(
        Path(args.artifacts_dir),
        model_name=(args.model_name or None),
        prefer_voting=args.prefer_voting,
    )
    jobs = fetch_jobs_from_sources(
        SOURCES,
        per_source=args.per_source,
        fail_fast=args.fail_fast_fetch,
    )
    scored = score_live_jobs(
        detector,
        jobs,
        with_explanations=args.with_explanations,
        with_heuristics=not args.no_heuristics,
        num_features=10,
        num_samples=1500,
        enable_i18n=not args.disable_i18n,
    )

    df = pd.DataFrame(scored)
    if df.empty:
        print("No jobs were fetched from the selected sources.")
        print(
            "Tip: run with --fail-fast-fetch to see provider errors, and confirm "
            "USAJOBS_API_KEY + USAJOBS_USER_AGENT are set."
        )
        return

    sort_col = "final_opportunity_score" if "final_opportunity_score" in df.columns else "fraud_probability"
    df = df.sort_values(sort_col, ascending=False)
    display_cols = [
        "source",
        "title",
        "company_profile",
        "location",
        "posted_date",
        "trust_score",
        "quality_score",
        "final_opportunity_score",
        "badge",
        "fraud_probability",
        "detected_language",
        "translation_applied",
        "prediction",
        "threshold",
        "job_url",
    ]
    if detector.model_name:
        print(f"Using model: {detector.model_name} (threshold={detector.threshold:.4f})\n")
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].to_string(index=False, max_colwidth=80))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
        print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
