import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.modeling import (
    BASE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    build_candidate_pipeline,
)


class ModelingRegistryTests(unittest.TestCase):
    def _sample_frame(self) -> tuple[pd.DataFrame, pd.Series]:
        rows = []
        labels = []
        for i in range(16):
            is_fraud = i % 2
            text = (
                "urgent transfer fee required remote job"
                if is_fraud
                else "software engineer remote team benefits"
            )
            rows.append(
                {
                    "combined_text": text,
                    "telecommuting": 1,
                    "has_company_logo": 0 if is_fraud else 1,
                    "has_questions": 0 if is_fraud else 1,
                }
            )
            labels.append(is_fraud)
        return pd.DataFrame(rows), pd.Series(labels)

    def test_candidate_registry_contains_expected_ids(self):
        self.assertIn("voting_soft", CANDIDATE_MODEL_IDS)
        for model_id in BASE_MODEL_IDS:
            self.assertIn(model_id, CANDIDATE_MODEL_IDS)

    def test_base_candidates_support_probability_output(self):
        x, y = self._sample_frame()
        for model_id in BASE_MODEL_IDS:
            with self.subTest(model_id=model_id):
                pipeline = build_candidate_pipeline(model_id=model_id, random_state=42)
                pipeline.fit(x, y)
                probs = pipeline.predict_proba(x)
                self.assertEqual(probs.shape, (len(x), 2))

    def test_voting_pipeline_builds_and_predicts(self):
        x, y = self._sample_frame()
        logreg = build_candidate_pipeline("logreg_tfidf", random_state=42)
        cnb = build_candidate_pipeline("cnb_bow", random_state=42)

        voting_pipeline = build_candidate_pipeline(
            model_id="voting_soft",
            params={"estimators": [("logreg_tfidf", logreg), ("cnb_bow", cnb)]},
            random_state=42,
        )
        voting_pipeline.fit(x, y)
        probs = voting_pipeline.predict_proba(x)
        self.assertEqual(probs.shape, (len(x), 2))


if __name__ == "__main__":
    unittest.main()
