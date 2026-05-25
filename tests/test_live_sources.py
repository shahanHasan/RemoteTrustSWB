import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.live_sources import (  # noqa: E402
    _parse_usajobs,
    model_payload_from_live_job,
    score_live_jobs,
)
from job_fraud_detector.i18n import TranslationResult  # noqa: E402


class _DummyDetector:
    def predict_posting(self, posting, with_explanation=False, num_features=10, num_samples=1500):
        _ = (posting, num_features, num_samples)
        result = {
            "fraud_probability": 0.2,
            "prediction": 0,
            "threshold": 0.5,
        }
        if with_explanation:
            result["lime_explanation"] = [("remote", -0.1), ("telegram", 0.2)]
        return result


class _BatchCapableDetector:
    threshold = 0.5

    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def predict_dataframe(self, prepared_features):
        self.batch_calls += 1
        out = prepared_features.copy()
        out["fraud_probability"] = 0.2
        out["prediction"] = 0
        return out

    def predict_posting(self, posting, with_explanation=False, num_features=10, num_samples=1500):
        _ = (posting, with_explanation, num_features, num_samples)
        self.single_calls += 1
        return {
            "fraud_probability": 0.2,
            "prediction": 0,
            "threshold": 0.5,
        }


class LiveSourcesTests(unittest.TestCase):
    def test_model_payload_maps_expected_contract(self):
        live_job = {
            "title": "Remote Engineer",
            "location": "Remote - Canada",
            "company_profile": "Acme",
            "description": "Build APIs",
            "requirements": "Python",
            "benefits": "Health",
            "telecommuting": "1",
            "has_company_logo": "",
            "has_questions": None,
        }

        payload = model_payload_from_live_job(live_job)
        self.assertEqual(payload["title"], "Remote Engineer")
        self.assertEqual(payload["telecommuting"], 1)
        self.assertEqual(payload["has_company_logo"], 0)
        self.assertEqual(payload["has_questions"], 0)

    def test_score_live_jobs_includes_heuristics(self):
        job = {
            "source": "jobicy_api",
            "job_url": "https://jobs.ashbyhq.com/acme/123",
            "apply_url": "https://jobs.ashbyhq.com/acme/123",
            "posted_date": "2026-05-20T00:00:00+00:00",
            "title": "Senior Data Engineer",
            "company_profile": "Acme Analytics",
            "location": "Remote - US",
            "description": "Remote role. Build systems in Python and SQL.",
            "requirements": "Python SQL AWS",
            "benefits": "Health, PTO",
            "employment_type": "Full-time",
            "industry": "Technology",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 0,
        }
        detector = _DummyDetector()
        rows = score_live_jobs(detector, [job], with_explanations=True, with_heuristics=True)
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertIn("fraud_probability", row)
        self.assertIn("trust_score", row)
        self.assertIn("quality_score", row)
        self.assertIn("final_opportunity_score", row)
        self.assertIn("lime_explanation", row)
        self.assertTrue(row["is_known_ats_domain"])
        self.assertIn("description", row)
        self.assertIn("requirements", row)

    def test_parse_usajobs_extracts_remote_posting_fields(self):
        sample = {
            "SearchResult": {
                "SearchResultItems": [
                    {
                        "MatchedObjectDescriptor": {
                            "PositionTitle": "Data Scientist",
                            "PositionURI": "https://www.usajobs.gov/job/123",
                            "ApplyURI": ["https://www.usajobs.gov/apply/123"],
                            "PositionLocationDisplay": "Anywhere in the U.S. (remote job)",
                            "DepartmentName": "Department of Example",
                            "OrganizationName": "Example Agency",
                            "PublicationStartDate": "2026-05-15T00:00:00+00:00",
                            "PositionSchedule": ["Full-time"],
                            "UserArea": {
                                "Details": {
                                    "JobSummary": "Analyze public data.",
                                    "MajorDuties": "Build dashboards.",
                                    "QualificationSummary": "Experience with analytics.",
                                    "Education": "Bachelor's degree.",
                                    "Benefits": "Federal benefits.",
                                    "PositionRemuneration": [
                                        {
                                            "MinimumRange": "90000",
                                            "MaximumRange": "120000",
                                            "RateIntervalCode": "PA",
                                        }
                                    ],
                                }
                            },
                        }
                    }
                ]
            }
        }

        rows = _parse_usajobs(json.dumps(sample).encode("utf-8"), limit=5)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "usajobs_api")
        self.assertEqual(row["telecommuting"], 1)
        self.assertEqual(row["title"], "Data Scientist")
        self.assertIn("90000", row["salary_range"])

    def test_score_live_jobs_uses_language_normalization_metadata(self):
        job = {
            "source": "jobicy_api",
            "job_url": "https://example.com/jobs/abc",
            "apply_url": "https://example.com/jobs/abc",
            "posted_date": "2026-05-20T00:00:00+00:00",
            "title": "Ingeniero de Datos",
            "company_profile": "Empresa Ejemplo",
            "location": "Remote - Spain",
            "description": "Rol remoto para análisis de datos.",
            "requirements": "SQL y Python",
            "benefits": "Seguro médico",
            "employment_type": "Tiempo completo",
            "industry": "Tecnología",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 0,
        }
        translated_posting = dict(job)
        translated_posting.update(
            {
                "title": "Data Engineer",
                "description": "Remote role for data analysis.",
                "requirements": "SQL and Python",
                "detected_language": "es",
                "language_confidence": 0.98,
                "language_detector": "langdetect",
                "translation_applied": True,
                "translation_provider": "test-provider",
                "translation_error": None,
            }
        )

        detector = _DummyDetector()
        with mock.patch(
            "job_fraud_detector.live_sources.normalize_posting_language",
            return_value=TranslationResult(
                posting=translated_posting,
                language="es",
                language_confidence=0.98,
                language_detector="langdetect",
                translation_applied=True,
                translation_provider="test-provider",
                translation_error=None,
            ),
        ):
            rows = score_live_jobs(detector, [job], with_explanations=False, with_heuristics=True)

        row = rows[0]
        self.assertEqual(row["title"], "Data Engineer")
        self.assertEqual(row["description"], "Remote role for data analysis.")
        self.assertEqual(row["detected_language"], "es")
        self.assertTrue(row["translation_applied"])
        self.assertEqual(row["translation_provider"], "test-provider")
        self.assertEqual(row["original_title"], "Ingeniero de Datos")

    def test_score_live_jobs_prefers_batch_inference_when_available(self):
        jobs = [
            {
                "source": "jobicy_api",
                "job_url": "https://example.com/jobs/1",
                "apply_url": "https://example.com/jobs/1",
                "posted_date": "2026-05-20T00:00:00+00:00",
                "title": "Role 1",
                "company_profile": "Co 1",
                "location": "Remote",
                "description": "Remote role with details.",
                "requirements": "Requirements",
                "benefits": "Benefits",
                "employment_type": "Full-time",
                "industry": "Tech",
                "telecommuting": 1,
                "has_company_logo": 1,
                "has_questions": 0,
            },
            {
                "source": "jobicy_api",
                "job_url": "https://example.com/jobs/2",
                "apply_url": "https://example.com/jobs/2",
                "posted_date": "2026-05-20T00:00:00+00:00",
                "title": "Role 2",
                "company_profile": "Co 2",
                "location": "Remote",
                "description": "Another remote role with details.",
                "requirements": "Requirements",
                "benefits": "Benefits",
                "employment_type": "Full-time",
                "industry": "Tech",
                "telecommuting": 1,
                "has_company_logo": 1,
                "has_questions": 0,
            },
        ]
        detector = _BatchCapableDetector()
        rows = score_live_jobs(detector, jobs, with_explanations=False, with_heuristics=True, batch_size=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(detector.batch_calls, 1)
        self.assertEqual(detector.single_calls, 0)
        self.assertTrue(all("fraud_probability" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
