import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.constants import NUMERIC_COLUMNS, TARGET_COLUMN, TEXT_COLUMNS
from job_fraud_detector.features import prepare_features, validate_required_columns


class FeaturePreparationTests(unittest.TestCase):
    def test_validate_required_columns_raises_for_missing_columns(self):
        df = pd.DataFrame({"title": ["A"]})
        with self.assertRaises(ValueError):
            validate_required_columns(df, include_target=True)

    def test_prepare_features_handles_nulls_and_numeric_coercion(self):
        row = {col: None for col in TEXT_COLUMNS}
        row.update({
            "telecommuting": "1",
            "has_company_logo": "",
            "has_questions": None,
            TARGET_COLUMN: "1",
        })
        df = pd.DataFrame([row])

        result = prepare_features(df, include_target=True)

        self.assertIn("combined_text", result.columns)
        self.assertEqual(result.iloc[0]["combined_text"], "")
        for col in NUMERIC_COLUMNS:
            self.assertIsInstance(int(result.iloc[0][col]), int)
        self.assertEqual(int(result.iloc[0]["telecommuting"]), 1)
        self.assertEqual(int(result.iloc[0]["has_company_logo"]), 0)
        self.assertEqual(int(result.iloc[0]["has_questions"]), 0)
        self.assertEqual(int(result.iloc[0][TARGET_COLUMN]), 1)


if __name__ == "__main__":
    unittest.main()
