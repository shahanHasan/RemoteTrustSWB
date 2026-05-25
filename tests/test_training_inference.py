import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.inference import FraudDetector

DATA_PATH = Path(os.environ.get("EMSCAD_DATA_PATH", "/Users/shahan/Downloads/fake_job_postings.csv"))


class TrainingAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_PATH.exists():
            raise unittest.SkipTest(f"Dataset not found at {DATA_PATH}")

        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.artifacts_dir = Path(cls.tmpdir.name)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(SRC)

        cmd = [
            sys.executable,
            "-m",
            "job_fraud_detector.train",
            "--data-path",
            str(DATA_PATH),
            "--artifacts-dir",
            str(cls.artifacts_dir),
            "--seed",
            "42",
            "--target-recall",
            "0.90",
            "--test-size",
            "0.15",
            "--val-size",
            "0.15",
            "--max-rows",
            "1200",
            "--candidate-models",
            "logreg_tfidf,cnb_bow,voting_soft",
            "--c-grid",
            "1.0",
            "--lime-num-samples",
            "300",
        ]
        run = subprocess.run(cmd, check=True, cwd=str(ROOT), env=env, capture_output=True, text=True)
        cls.train_stdout = run.stdout

        cls.detector = FraudDetector.from_artifacts(cls.artifacts_dir)

        with (cls.artifacts_dir / "metrics.json").open("r", encoding="utf-8") as fp:
            cls.metrics = json.load(fp)

        with (cls.artifacts_dir / "example_explanations.json").open("r", encoding="utf-8") as fp:
            cls.examples = json.load(fp)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_training_smoke_outputs_artifacts(self):
        expected = [
            "model.joblib",
            "config.json",
            "metrics.json",
            "example_explanations.json",
            "model_comparison.csv",
            "confusion_matrix_test.png",
            "candidate_models/logreg_tfidf.joblib",
            "candidate_models/cnb_bow.joblib",
            "candidate_models/voting_soft.joblib",
        ]
        for filename in expected:
            self.assertTrue((self.artifacts_dir / filename).exists(), f"Missing artifact: {filename}")

    def test_metrics_report_contains_required_fields(self):
        test_metrics = self.metrics["test_metrics"]
        for key in ["precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"]:
            self.assertIn(key, test_metrics)

        self.assertIn("threshold", self.metrics["threshold_selection"])
        self.assertIn("selected_model_name", self.metrics)
        self.assertIn("per_model_validation", self.metrics)
        self.assertIn("per_model_test", self.metrics)
        self.assertGreaterEqual(len(self.metrics["per_model_validation"]), 2)

    def test_cli_output_includes_selected_model_and_threshold(self):
        self.assertIn("Selected model:", self.train_stdout)
        self.assertIn("Threshold:", self.train_stdout)

    def test_inference_contract_unchanged(self):
        posting = {
            "title": "Remote Data Entry Specialist",
            "location": "US, CA, San Francisco",
            "department": "Operations",
            "company_profile": "Growing startup with distributed team.",
            "description": "We are hiring a remote specialist to process digital forms and validate records.",
            "requirements": "Strong communication skills and attention to detail.",
            "benefits": "Health insurance and PTO.",
            "employment_type": "Full-time",
            "required_experience": "Entry level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Administrative",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        }
        result = self.detector.predict_posting(posting, with_explanation=False)
        self.assertIn("fraud_probability", result)
        self.assertIn("prediction", result)
        self.assertIn("threshold", result)

    def test_inference_with_lime_returns_non_empty_explanation(self):
        posting = {
            "title": "Remote Data Entry Specialist",
            "location": "US, CA, San Francisco",
            "department": "Operations",
            "company_profile": "Growing startup with distributed team.",
            "description": "We are hiring a remote specialist to process digital forms and validate records.",
            "requirements": "Strong communication skills and attention to detail.",
            "benefits": "Health insurance and PTO.",
            "employment_type": "Full-time",
            "required_experience": "Entry level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Administrative",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        }
        result = self.detector.predict_posting(posting, with_explanation=True, num_features=10, num_samples=300)
        self.assertIn("lime_explanation", result)
        self.assertGreater(len(result["lime_explanation"]), 0)

    def test_inference_can_explicitly_load_voting_model(self):
        posting = {
            "title": "Remote Data Entry Specialist",
            "location": "US, CA, San Francisco",
            "department": "Operations",
            "company_profile": "Growing startup with distributed team.",
            "description": "We are hiring a remote specialist to process digital forms and validate records.",
            "requirements": "Strong communication skills and attention to detail.",
            "benefits": "Health insurance and PTO.",
            "employment_type": "Full-time",
            "required_experience": "Entry level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Administrative",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        }
        voting_detector = FraudDetector.from_artifacts(self.artifacts_dir, model_name="voting_soft")
        self.assertEqual(voting_detector.model_name, "voting_soft")
        result = voting_detector.predict_posting(posting, with_explanation=False)
        self.assertIn("fraud_probability", result)
        self.assertIn("prediction", result)
        self.assertIn("threshold", result)

    def test_behavioral_examples_meet_expected_direction(self):
        fraud = next((item for item in self.examples if item.get("kind") == "fraud"), None)
        legit = next((item for item in self.examples if item.get("kind") == "legitimate"), None)

        if fraud is None or legit is None:
            self.skipTest("Could not build both fraud and legitimate examples in artifact generation.")

        threshold = float(fraud["threshold"])
        self.assertGreaterEqual(float(fraud["fraud_probability"]), threshold)
        self.assertLess(float(legit["fraud_probability"]), threshold)


if __name__ == "__main__":
    unittest.main()
