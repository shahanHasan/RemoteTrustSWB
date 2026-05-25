"""Model construction, candidate registry, threshold selection, and metric helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .constants import NUMERIC_COLUMNS

BASE_MODEL_IDS = [
    "logreg_tfidf",
    "sgd_log_tfidf",
    "linsvc_cal_tfidf",
    "cnb_bow",
]

CANDIDATE_MODEL_IDS = [*BASE_MODEL_IDS, "voting_soft"]


def _make_tfidf_vectorizer(min_df: int = 3, max_features: int = 60000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def _make_count_vectorizer(min_df: int = 3, max_features: int = 60000) -> CountVectorizer:
    return CountVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
    )


def _build_preprocessor(text_vectorizer: Any) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "combined_text"),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ]
    )


def build_candidate_pipeline(
    model_id: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build one candidate model pipeline from id and optional hyperparameters."""
    cfg = params or {}

    vectorizer_min_df = int(cfg.get("vectorizer_min_df", 3))
    vectorizer_max_features = int(cfg.get("vectorizer_max_features", 60000))

    if model_id == "logreg_tfidf":
        preprocessor = _build_preprocessor(
            _make_tfidf_vectorizer(
                min_df=vectorizer_min_df,
                max_features=vectorizer_max_features,
            )
        )
        classifier = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            C=float(cfg.get("C", 1.0)),
            max_iter=1000,
            random_state=random_state,
        )

    elif model_id == "sgd_log_tfidf":
        preprocessor = _build_preprocessor(
            _make_tfidf_vectorizer(
                min_df=vectorizer_min_df,
                max_features=vectorizer_max_features,
            )
        )
        classifier = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=float(cfg.get("alpha", 1e-4)),
            penalty=str(cfg.get("penalty", "l2")),
            max_iter=int(cfg.get("max_iter", 3000)),
            tol=float(cfg.get("tol", 1e-3)),
            random_state=random_state,
        )

    elif model_id == "linsvc_cal_tfidf":
        preprocessor = _build_preprocessor(
            _make_tfidf_vectorizer(
                min_df=vectorizer_min_df,
                max_features=vectorizer_max_features,
            )
        )
        base_estimator = LinearSVC(
            C=float(cfg.get("C", 1.0)),
            class_weight="balanced",
            max_iter=int(cfg.get("max_iter", 5000)),
            random_state=random_state,
        )
        classifier = CalibratedClassifierCV(
            estimator=base_estimator,
            method=str(cfg.get("calibration_method", "sigmoid")),
            cv=int(cfg.get("calibration_cv", 3)),
        )

    elif model_id == "cnb_bow":
        preprocessor = _build_preprocessor(
            _make_count_vectorizer(
                min_df=vectorizer_min_df,
                max_features=vectorizer_max_features,
            )
        )
        classifier = ComplementNB(
            alpha=float(cfg.get("alpha", 1.0)),
            norm=bool(cfg.get("norm", False)),
        )

    elif model_id == "voting_soft":
        estimators = cfg.get("estimators")
        if not estimators:
            raise ValueError("voting_soft requires 'estimators' in params.")
        classifier = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=cfg.get("weights"),
            n_jobs=cfg.get("n_jobs", None),
        )
        return Pipeline(steps=[("classifier", classifier)])

    else:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. Valid ids: {CANDIDATE_MODEL_IDS}."
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def candidate_parameter_grid(model_id: str) -> list[dict[str, Any]]:
    """Return compact parameter combinations for each model family."""
    if model_id == "logreg_tfidf":
        grid = {
            "vectorizer_min_df": [2, 3],
            "vectorizer_max_features": [50000, 60000],
            "C": [0.5, 1.0, 2.0],
        }
        return list(ParameterGrid(grid))

    if model_id == "sgd_log_tfidf":
        grid = {
            "vectorizer_min_df": [2, 3],
            "vectorizer_max_features": [50000],
            "alpha": [1e-5, 1e-4],
            "penalty": ["l2", "elasticnet"],
            "max_iter": [3000],
            "tol": [1e-3],
        }
        return list(ParameterGrid(grid))

    if model_id == "linsvc_cal_tfidf":
        grid = {
            "vectorizer_min_df": [2, 3],
            "vectorizer_max_features": [50000],
            "C": [0.5, 1.0, 2.0],
            "max_iter": [5000],
            "calibration_method": ["sigmoid"],
            "calibration_cv": [3],
        }
        return list(ParameterGrid(grid))

    if model_id == "cnb_bow":
        grid = {
            "vectorizer_min_df": [2, 3],
            "vectorizer_max_features": [50000, 60000],
            "alpha": [0.1, 0.5, 1.0],
            "norm": [False, True],
        }
        return list(ParameterGrid(grid))

    if model_id == "voting_soft":
        return [{}]

    raise ValueError(f"No parameter grid registered for model_id '{model_id}'.")


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    preds = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return precision, recall, f1


def select_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> ThresholdResult:
    """Pick threshold with max precision under recall constraint; fallback to max recall."""
    unique_thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.concatenate(([0.0], unique_thresholds, [1.0]))

    best_constrained: ThresholdResult | None = None
    best_fallback: ThresholdResult | None = None

    for threshold in thresholds:
        precision, recall, f1 = _score_at_threshold(y_true, y_prob, float(threshold))

        fallback_candidate = ThresholdResult(
            threshold=float(threshold),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            mode="max_recall_fallback",
        )
        if best_fallback is None:
            best_fallback = fallback_candidate
        elif fallback_candidate.recall > best_fallback.recall:
            best_fallback = fallback_candidate
        elif np.isclose(fallback_candidate.recall, best_fallback.recall) and fallback_candidate.precision > best_fallback.precision:
            best_fallback = fallback_candidate

        if recall >= target_recall:
            constrained_candidate = ThresholdResult(
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                mode="target_recall_met",
            )
            if best_constrained is None:
                best_constrained = constrained_candidate
            elif constrained_candidate.precision > best_constrained.precision:
                best_constrained = constrained_candidate
            elif np.isclose(constrained_candidate.precision, best_constrained.precision) and constrained_candidate.recall > best_constrained.recall:
                best_constrained = constrained_candidate

    if best_constrained is not None:
        return best_constrained
    if best_fallback is None:
        raise ValueError("Unable to select a threshold from empty probabilities.")
    return best_fallback


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Build complete metric report for a thresholded binary classifier."""
    y_pred = (y_prob >= threshold).astype(int)

    report = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }
    return report


def fit_and_predict_proba(
    pipeline: Pipeline,
    train_x: pd.DataFrame,
    train_y: Iterable[int],
    eval_x: pd.DataFrame,
) -> np.ndarray:
    """Train a pipeline and return positive-class probabilities on eval set."""
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(eval_x)
    return probabilities[:, 1]
