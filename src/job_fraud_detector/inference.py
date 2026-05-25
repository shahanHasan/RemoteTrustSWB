"""Inference and LIME explainability utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .constants import CLASS_NAMES, NUMERIC_COLUMNS
from .features import prepare_single_posting


class FraudDetector:
    """Wrapper for fraud scoring and local explanations."""

    def __init__(self, pipeline: Any, threshold: float, model_name: str | None = None):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.model_name = model_name
        self.explainer = None

    def _get_explainer(self):
        if self.explainer is not None:
            return self.explainer
        try:
            from lime.lime_text import LimeTextExplainer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LIME is not installed in this Python environment. "
                "Install it with `pip install lime` or use the project `.venv` kernel."
            ) from exc
        self.explainer = LimeTextExplainer(class_names=CLASS_NAMES)
        return self.explainer

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        model_name: str | None = None,
        prefer_voting: bool = False,
    ) -> "FraudDetector":
        artifacts_path = Path(artifacts_dir)
        with (artifacts_path / "config.json").open("r", encoding="utf-8") as fp:
            config = json.load(fp)

        selected_model_name = config.get("selected_model_name")
        available_models = set(config.get("available_models", []))
        candidate_model_paths: dict[str, str] = config.get("candidate_model_paths", {})
        thresholds_by_model: dict[str, float] = config.get("thresholds_by_model", {})

        resolved_model_name = model_name
        if resolved_model_name is None and prefer_voting and "voting_soft" in available_models:
            resolved_model_name = "voting_soft"
        if resolved_model_name is None:
            resolved_model_name = selected_model_name

        if resolved_model_name is not None and available_models and resolved_model_name not in available_models:
            raise ValueError(
                f"Model '{resolved_model_name}' is not available in artifacts. "
                f"Available models: {sorted(available_models)}"
            )

        if resolved_model_name and resolved_model_name in candidate_model_paths:
            model_path = artifacts_path / candidate_model_paths[resolved_model_name]
        elif resolved_model_name and (artifacts_path / "candidate_models" / f"{resolved_model_name}.joblib").exists():
            model_path = artifacts_path / "candidate_models" / f"{resolved_model_name}.joblib"
        else:
            model_path = artifacts_path / "model.joblib"

        model = joblib.load(model_path)

        if resolved_model_name and resolved_model_name in thresholds_by_model:
            threshold = float(thresholds_by_model[resolved_model_name])
        elif resolved_model_name and resolved_model_name == selected_model_name:
            threshold = float(config["threshold"])
        else:
            threshold = float(config["threshold"])

        return cls(
            pipeline=model,
            threshold=threshold,
            model_name=resolved_model_name,
        )

    def predict_dataframe(self, prepared_features: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud probability and label for already-prepared feature rows."""
        probs = self.pipeline.predict_proba(prepared_features)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        out = prepared_features.copy()
        out["fraud_probability"] = probs
        out["prediction"] = preds
        return out

    def _lime_predict_fn(self, base_row: pd.DataFrame):
        numeric_values = {col: int(base_row.iloc[0][col]) for col in NUMERIC_COLUMNS}

        def predict_fn(text_samples: list[str]):
            frame = pd.DataFrame({"combined_text": text_samples})
            for col, val in numeric_values.items():
                frame[col] = val
            return self.pipeline.predict_proba(frame)

        return predict_fn

    def explain_posting(
        self,
        posting: Mapping[str, Any],
        num_features: int = 10,
        num_samples: int = 3000,
    ) -> dict[str, Any]:
        """Score one posting and return probability, class, and LIME contribution list."""
        prepared = prepare_single_posting(posting)
        probability = float(self.pipeline.predict_proba(prepared)[0, 1])
        prediction = int(probability >= self.threshold)

        raw_text = prepared.iloc[0]["combined_text"]
        explanation = self._get_explainer().explain_instance(
            raw_text,
            classifier_fn=self._lime_predict_fn(prepared),
            labels=[1],
            num_features=num_features,
            num_samples=num_samples,
        )
        top_features = explanation.as_list(label=1)

        return {
            "fraud_probability": probability,
            "prediction": prediction,
            "threshold": self.threshold,
            "lime_explanation": [
                {"token": token, "weight": float(weight)}
                for token, weight in top_features
            ],
        }

    def predict_posting(
        self,
        posting: Mapping[str, Any],
        with_explanation: bool = False,
        num_features: int = 10,
        num_samples: int = 3000,
    ) -> dict[str, Any]:
        """Public inference function for one posting dict."""
        prepared = prepare_single_posting(posting)
        probability = float(self.pipeline.predict_proba(prepared)[0, 1])
        prediction = int(probability >= self.threshold)
        result: dict[str, Any] = {
            "fraud_probability": probability,
            "prediction": prediction,
            "threshold": self.threshold,
        }

        if with_explanation:
            explanation = self._get_explainer().explain_instance(
                prepared.iloc[0]["combined_text"],
                classifier_fn=self._lime_predict_fn(prepared),
                labels=[1],
                num_features=num_features,
                num_samples=num_samples,
            )
            result["lime_explanation"] = [
                {"token": token, "weight": float(weight)}
                for token, weight in explanation.as_list(label=1)
            ]

        return result
