"""Training entrypoint for EMSCAD traditional-ML benchmark and ensemble detector."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

# Ensure plotting works in restricted/non-GUI environments.
_mpl_cache_dir = Path(tempfile.gettempdir()) / "job_fraud_detector_mpl_cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from .constants import ID_COLUMN, TARGET_COLUMN
from .features import prepare_features, validate_required_columns
from .inference import FraudDetector
from .modeling import (
    BASE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    build_candidate_pipeline,
    candidate_parameter_grid,
    evaluate_binary_classification,
    fit_and_predict_proba,
    select_threshold_for_recall,
)


def _parse_c_grid(raw: str | None) -> list[float]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    return [float(item) for item in values]


def _parse_candidate_models(raw: str | None, disable_voting: bool) -> list[str]:
    if raw:
        selected = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        selected = list(CANDIDATE_MODEL_IDS)

    invalid = [model_id for model_id in selected if model_id not in CANDIDATE_MODEL_IDS]
    if invalid:
        raise ValueError(
            f"Unknown model ids in --candidate-models: {invalid}. "
            f"Valid ids: {CANDIDATE_MODEL_IDS}"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in selected:
        if model_id not in seen:
            deduped.append(model_id)
            seen.add(model_id)

    if disable_voting:
        deduped = [model_id for model_id in deduped if model_id != "voting_soft"]

    if not deduped:
        raise ValueError(
            "No candidate models remain after parsing flags. "
            "Provide --candidate-models or remove --disable-voting."
        )

    return deduped


def _ensure_split_sizes(val_size: float, test_size: float) -> None:
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be in (0, 1).")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("--val-size + --test-size must be < 1.")


def _constraint_met(threshold_info: dict[str, Any]) -> bool:
    return threshold_info.get("mode") == "target_recall_met"


def _selection_key(
    threshold_info: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    return (
        1.0 if _constraint_met(threshold_info) else 0.0,
        float(threshold_info["precision"]),
        float(metrics["pr_auc"]),
        float(threshold_info["recall"]),
        float(threshold_info["f1"]),
    )


def _family_grid(model_id: str, logistic_c_grid_override: list[float]) -> list[dict[str, Any]]:
    grid = candidate_parameter_grid(model_id)
    if model_id == "logreg_tfidf" and logistic_c_grid_override:
        # Preserve non-C combinations from the default grid while replacing C values.
        expanded: list[dict[str, Any]] = []
        seen: set[tuple[int, int, float]] = set()
        for item in grid:
            for c_value in logistic_c_grid_override:
                candidate = dict(item)
                candidate["C"] = float(c_value)
                key = (
                    int(candidate.get("vectorizer_min_df", 3)),
                    int(candidate.get("vectorizer_max_features", 60000)),
                    float(candidate["C"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(candidate)
        return expanded
    return grid


def _tune_one_model_family(
    model_id: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    target_recall: float,
    logistic_c_grid_override: list[float],
) -> dict[str, Any]:
    grid = _family_grid(model_id=model_id, logistic_c_grid_override=logistic_c_grid_override)

    best: dict[str, Any] | None = None
    for params in grid:
        pipeline = build_candidate_pipeline(
            model_id=model_id,
            params=params,
            random_state=seed,
        )
        val_probs = fit_and_predict_proba(
            pipeline=pipeline,
            train_x=x_train,
            train_y=y_train,
            eval_x=x_val,
        )
        threshold_info = select_threshold_for_recall(
            y_true=y_val.to_numpy(),
            y_prob=val_probs,
            target_recall=target_recall,
        ).to_dict()
        val_metrics = evaluate_binary_classification(
            y_true=y_val.to_numpy(),
            y_prob=val_probs,
            threshold=float(threshold_info["threshold"]),
        )

        candidate = {
            "model_id": model_id,
            "params": params,
            "pipeline": pipeline,
            "threshold_info": threshold_info,
            "validation_metrics": val_metrics,
        }

        if best is None or _selection_key(threshold_info, val_metrics) > _selection_key(
            best["threshold_info"],
            best["validation_metrics"],
        ):
            best = candidate

    if best is None:
        raise ValueError(f"No candidate trained for model family '{model_id}'.")

    return best


def _build_voting_candidate(
    tuned_base_models: dict[str, dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    target_recall: float,
) -> dict[str, Any] | None:
    if len(tuned_base_models) < 2:
        return None

    estimators: list[tuple[str, Any]] = []
    for model_id in BASE_MODEL_IDS:
        payload = tuned_base_models.get(model_id)
        if payload is None:
            continue
        base_pipeline = build_candidate_pipeline(
            model_id=model_id,
            params=payload["params"],
            random_state=seed,
        )
        estimators.append((model_id, base_pipeline))

    if len(estimators) < 2:
        return None

    voting_pipeline = build_candidate_pipeline(
        model_id="voting_soft",
        params={
            "estimators": estimators,
            "n_jobs": None,
        },
        random_state=seed,
    )

    val_probs = fit_and_predict_proba(
        pipeline=voting_pipeline,
        train_x=x_train,
        train_y=y_train,
        eval_x=x_val,
    )
    threshold_info = select_threshold_for_recall(
        y_true=y_val.to_numpy(),
        y_prob=val_probs,
        target_recall=target_recall,
    ).to_dict()
    val_metrics = evaluate_binary_classification(
        y_true=y_val.to_numpy(),
        y_prob=val_probs,
        threshold=float(threshold_info["threshold"]),
    )

    return {
        "model_id": "voting_soft",
        "params": {
            "estimators": [name for name, _ in estimators],
            "weights": None,
        },
        "pipeline": voting_pipeline,
        "threshold_info": threshold_info,
        "validation_metrics": val_metrics,
    }


def _save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["legitimate", "fraudulent"],
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Selected Model Confusion Matrix (Test)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _serialize_model_report(
    candidate: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    threshold_info = candidate["threshold_info"]
    val_metrics = candidate["validation_metrics"]
    return {
        "model_id": candidate["model_id"],
        "threshold": float(threshold_info["threshold"]),
        "threshold_mode": threshold_info["mode"],
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_f1": float(val_metrics["f1"]),
        "val_pr_auc": float(val_metrics["pr_auc"]),
        "val_roc_auc": float(val_metrics["roc_auc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
    }


def _build_terminal_table(summary_rows: list[dict[str, Any]], selected_model_name: str) -> str:
    header = (
        "model".ljust(18)
        + " val_recall ".rjust(12)
        + " val_prec ".rjust(11)
        + " val_pr_auc ".rjust(12)
        + " test_recall ".rjust(13)
        + " test_prec ".rjust(11)
        + " test_pr_auc ".rjust(12)
        + " threshold ".rjust(11)
        + " mode"
    )
    lines = [header, "-" * len(header)]
    for row in summary_rows:
        model = row["model_id"]
        marker = "*" if model == selected_model_name else " "
        line = (
            f"{marker}{model[:17]}".ljust(18)
            + f"{row['val_recall']:.4f}".rjust(12)
            + f"{row['val_precision']:.4f}".rjust(11)
            + f"{row['val_pr_auc']:.4f}".rjust(12)
            + f"{row['test_recall']:.4f}".rjust(13)
            + f"{row['test_precision']:.4f}".rjust(11)
            + f"{row['test_pr_auc']:.4f}".rjust(12)
            + f"{row['threshold']:.4f}".rjust(11)
            + f" {row['threshold_mode']}"
        )
        lines.append(line)
    return "\n".join(lines)


def train(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_split_sizes(args.val_size, args.test_size)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    read_csv_kwargs: dict[str, Any] = {}
    if args.max_rows is not None:
        read_csv_kwargs["nrows"] = args.max_rows

    raw_df = pd.read_csv(args.data_path, **read_csv_kwargs)
    if ID_COLUMN in raw_df.columns:
        raw_df = raw_df.drop(columns=[ID_COLUMN])

    validate_required_columns(raw_df, include_target=True)
    prepared = prepare_features(raw_df, include_target=True)

    x_all = prepared.drop(columns=[TARGET_COLUMN])
    y_all = prepared[TARGET_COLUMN].astype(int)

    all_indices = prepared.index.to_numpy()
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(args.val_size + args.test_size),
        random_state=args.seed,
        stratify=y_all,
    )

    temp_labels = y_all.loc[temp_idx]
    val_ratio_in_temp = args.val_size / (args.val_size + args.test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio_in_temp),
        random_state=args.seed,
        stratify=temp_labels,
    )

    x_train = x_all.loc[train_idx]
    y_train = y_all.loc[train_idx]
    x_val = x_all.loc[val_idx]
    y_val = y_all.loc[val_idx]
    x_test = x_all.loc[test_idx]
    y_test = y_all.loc[test_idx]

    requested_models = _parse_candidate_models(
        raw=args.candidate_models,
        disable_voting=args.disable_voting,
    )

    logistic_c_grid_override = _parse_c_grid(args.c_grid)

    tuned_base_models: dict[str, dict[str, Any]] = {}
    model_candidates: dict[str, dict[str, Any]] = {}

    base_to_train = [model_id for model_id in requested_models if model_id in BASE_MODEL_IDS]
    if "voting_soft" in requested_models and not base_to_train:
        # If user asks for voting only, build it from all known base families.
        base_to_train = list(BASE_MODEL_IDS)

    for model_id in base_to_train:
        tuned = _tune_one_model_family(
            model_id=model_id,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=args.seed,
            target_recall=args.target_recall,
            logistic_c_grid_override=logistic_c_grid_override,
        )
        tuned_base_models[model_id] = tuned

    for model_id in requested_models:
        if model_id in tuned_base_models:
            model_candidates[model_id] = tuned_base_models[model_id]

    if "voting_soft" in requested_models:
        voting_candidate = _build_voting_candidate(
            tuned_base_models=tuned_base_models,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=args.seed,
            target_recall=args.target_recall,
        )
        if voting_candidate is not None:
            model_candidates["voting_soft"] = voting_candidate

    if not model_candidates:
        raise ValueError("No model candidates were produced after tuning.")

    per_model_validation: dict[str, Any] = {}
    per_model_test: dict[str, Any] = {}
    model_rows: list[dict[str, Any]] = []

    champion: dict[str, Any] | None = None
    champion_test_metrics: dict[str, Any] | None = None
    champion_test_probs: np.ndarray | None = None

    for model_id, candidate in model_candidates.items():
        threshold = float(candidate["threshold_info"]["threshold"])
        test_probs = candidate["pipeline"].predict_proba(x_test)[:, 1]
        test_metrics = evaluate_binary_classification(
            y_true=y_test.to_numpy(),
            y_prob=test_probs,
            threshold=threshold,
        )

        per_model_validation[model_id] = {
            "params": candidate["params"],
            "threshold_selection": candidate["threshold_info"],
            "metrics": candidate["validation_metrics"],
        }
        per_model_test[model_id] = {
            "threshold": threshold,
            "metrics": test_metrics,
        }

        row = _serialize_model_report(candidate, test_metrics)
        model_rows.append(row)

        if champion is None:
            champion = candidate
            champion_test_metrics = test_metrics
            champion_test_probs = test_probs
        else:
            curr_key = _selection_key(
                candidate["threshold_info"], candidate["validation_metrics"]
            )
            best_key = _selection_key(
                champion["threshold_info"], champion["validation_metrics"]
            )
            if curr_key > best_key:
                champion = candidate
                champion_test_metrics = test_metrics
                champion_test_probs = test_probs

    if champion is None or champion_test_metrics is None or champion_test_probs is None:
        raise RuntimeError("Champion model selection failed unexpectedly.")

    champion_model_name = str(champion["model_id"])
    champion_threshold = float(champion["threshold_info"]["threshold"])
    available_models = sorted(model_candidates.keys())
    thresholds_by_model = {
        model_id: float(candidate["threshold_info"]["threshold"])
        for model_id, candidate in model_candidates.items()
    }

    candidate_models_dir = artifacts_dir / "candidate_models"
    candidate_models_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_paths: dict[str, str] = {}
    for model_id, candidate in model_candidates.items():
        rel_path = f"candidate_models/{model_id}.joblib"
        joblib.dump(candidate["pipeline"], artifacts_dir / rel_path)
        candidate_model_paths[model_id] = rel_path

    config = {
        "threshold": champion_threshold,
        "selected_model_name": champion_model_name,
        "available_models": available_models,
        "thresholds_by_model": thresholds_by_model,
        "candidate_model_paths": candidate_model_paths,
        "target_recall": float(args.target_recall),
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "candidate_models": requested_models,
        "disable_voting": bool(args.disable_voting),
        "c_grid": logistic_c_grid_override,
        "plot_confusion_matrix": bool(args.plot_confusion_matrix),
    }

    report = {
        "config": config,
        "dataset": {
            "rows_total": int(len(prepared)),
            "fraud_rate": float(y_all.mean()),
            "split_sizes": {
                "train": int(len(train_idx)),
                "validation": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        },
        "selected_model_name": champion_model_name,
        "threshold_selection": champion["threshold_info"],
        "validation_metrics": champion["validation_metrics"],
        "test_metrics": champion_test_metrics,
        "available_models": available_models,
        "thresholds_by_model": thresholds_by_model,
        "per_model_validation": per_model_validation,
        "per_model_test": per_model_test,
    }

    joblib.dump(champion["pipeline"], artifacts_dir / "model.joblib")
    with (artifacts_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    comparison_df = pd.DataFrame(model_rows)
    comparison_csv_path = artifacts_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    cm_path = artifacts_dir / "confusion_matrix_test.png"
    if args.plot_confusion_matrix:
        test_preds = (champion_test_probs >= champion_threshold).astype(int)
        _save_confusion_matrix_png(
            y_true=y_test.to_numpy(),
            y_pred=test_preds,
            output_path=cm_path,
        )

    detector = FraudDetector(pipeline=champion["pipeline"], threshold=champion_threshold)
    raw_eval_frame = raw_df.loc[test_idx].copy()
    eval_with_scores = raw_eval_frame.copy()
    eval_with_scores["actual_label"] = y_test.to_numpy()
    eval_with_scores["fraud_probability"] = champion_test_probs
    eval_with_scores["prediction"] = (champion_test_probs >= champion_threshold).astype(int)

    examples: list[dict[str, Any]] = []

    fraud_samples = eval_with_scores[
        (eval_with_scores["actual_label"] == 1)
        & (eval_with_scores["fraud_probability"] >= champion_threshold)
    ]
    if fraud_samples.empty:
        fraud_samples = eval_with_scores[eval_with_scores["actual_label"] == 1]

    legit_samples = eval_with_scores[
        (eval_with_scores["actual_label"] == 0)
        & (eval_with_scores["fraud_probability"] < champion_threshold)
    ]
    if legit_samples.empty:
        legit_samples = eval_with_scores[eval_with_scores["actual_label"] == 0]

    for label, sample_df in (("fraud", fraud_samples.head(1)), ("legitimate", legit_samples.head(1))):
        if sample_df.empty:
            continue
        row = sample_df.iloc[0]
        explanation = detector.predict_posting(
            posting=row.to_dict(),
            with_explanation=True,
            num_features=10,
            num_samples=args.lime_num_samples,
        )
        examples.append(
            {
                "kind": label,
                "title": str(row.get("title", "")),
                "actual_label": int(row["actual_label"]),
                **explanation,
            }
        )

    with (artifacts_dir / "example_explanations.json").open("w", encoding="utf-8") as fp:
        json.dump(examples, fp, indent=2)

    terminal_rows_sorted = sorted(
        model_rows,
        key=lambda row: (
            1.0 if row["threshold_mode"] == "target_recall_met" else 0.0,
            row["val_precision"],
            row["val_pr_auc"],
            row["val_recall"],
        ),
        reverse=True,
    )
    table = _build_terminal_table(terminal_rows_sorted, selected_model_name=champion_model_name)

    report["terminal_summary_table"] = table
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EMSCAD traditional-ML benchmark and soft-voting fraud detector."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to fake_job_postings.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Output directory for model and reports.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--c-grid",
        type=str,
        default=None,
        help=(
            "Optional comma-separated LogisticRegression C values override, "
            "e.g. '0.5,1.0,2.0'."
        ),
    )
    parser.add_argument(
        "--candidate-models",
        type=str,
        default=None,
        help=(
            "Optional comma-separated model ids. "
            f"Default: {','.join(CANDIDATE_MODEL_IDS)}"
        ),
    )
    parser.add_argument(
        "--disable-voting",
        action="store_true",
        help="Skip the soft-voting ensemble candidate.",
    )
    parser.add_argument(
        "--plot-confusion-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion_matrix_test.png for the selected model (default: true).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick smoke runs.",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=1500,
        help="Neighborhood sample count for artifact explanation generation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = train(args)

    print("Training complete.")
    print(f"Selected model: {report['selected_model_name']}")
    threshold_meta = report["threshold_selection"]
    print(f"Threshold: {threshold_meta['threshold']:.4f} ({threshold_meta['mode']})")
    print(report["terminal_summary_table"])


if __name__ == "__main__":
    main()
