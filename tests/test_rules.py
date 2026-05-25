import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.rules import evaluate_job_posting


class RuleEngineTests(unittest.TestCase):
    def test_hard_caps_apply_for_high_risk_patterns(self):
        posting = {
            "title": "Remote product booster",
            "company_profile": "",
            "description": (
                "Reply YES on Telegram to start. "
                "Complete online tasks and product boosting for commission. "
                "Deposit crypto to unlock higher level tasks. "
                "You must pay an onboarding fee."
            ),
            "requirements": "No experience needed.",
            "benefits": "Daily pay",
            "employment_type": "Contract",
            "location": "Remote",
            "job_url": "https://bit.ly/suspicious-job",
            "apply_url": "https://bit.ly/suspicious-job",
            "telecommuting": 1,
        }

        result = evaluate_job_posting(posting, ml_fraud_probability=0.05)

        self.assertLessEqual(result["trust_score"], 25.0)
        self.assertEqual(result["fraud_risk_level"], "high")
        self.assertEqual(result["badge"], "Avoid")
        self.assertGreaterEqual(len(result["hard_caps"]), 1)
        self.assertTrue(result["matched_signals"]["pay_to_get_paid"])
        self.assertTrue(result["matched_signals"]["task_scam"])
        self.assertTrue(result["trust_cap_applied"])
        self.assertGreater(result["trust_pre_cap_score"], result["trust_score"])
        self.assertIn("trust_blend", result["weights"])
        self.assertIn("final_blend", result["weights"])

    def test_known_ats_domain_boosts_trust(self):
        posting = {
            "title": "Senior Data Engineer",
            "company_profile": "Well-known software company with distributed team.",
            "description": (
                "Build pipelines in Python and SQL. "
                "Collaborate with product and analytics teams."
            ),
            "requirements": "5+ years Python, SQL, cloud data stack, testing, CI/CD",
            "benefits": "Medical, dental, RRSP match, learning budget",
            "salary_range": "$140,000 - $180,000",
            "employment_type": "Full-time",
            "location": "Remote - US",
            "job_url": "https://jobs.ashbyhq.com/example/abcd1234",
            "apply_url": "https://jobs.ashbyhq.com/example/abcd1234",
            "posted_date": "2026-05-20T00:00:00+00:00",
            "telecommuting": 1,
        }

        result = evaluate_job_posting(posting, ml_fraud_probability=0.20)

        self.assertTrue(result["is_known_ats_domain"])
        self.assertEqual(result["ats_provider"], "ashby")
        self.assertGreater(result["trust_score"], 60.0)
        self.assertGreater(result["quality_score"], 60.0)
        self.assertIn("Apply domain matches known ATS infrastructure", " ".join(result["positive_evidence"]))
        self.assertAlmostEqual(result["weights"]["trust_blend"]["rule_engine_score"], 0.45)
        self.assertAlmostEqual(result["weights"]["final_blend"]["trust_score"], 0.70)

    def test_workday_and_major_job_board_detection(self):
        workday_post = {
            "title": "Product Manager",
            "company_profile": "Established enterprise employer",
            "description": "Own roadmap and delivery for platform capabilities.",
            "requirements": "5+ years product management experience",
            "employment_type": "Full-time",
            "location": "Remote - US",
            "job_url": "https://myview.wd3.myworkdayjobs.com/en-US/careers/job/123",
            "apply_url": "https://myview.wd3.myworkdayjobs.com/en-US/careers/job/123",
            "telecommuting": 1,
        }
        workday_result = evaluate_job_posting(workday_post, ml_fraud_probability=0.25)
        self.assertTrue(workday_result["is_known_ats_domain"])
        self.assertEqual(workday_result["ats_provider"], "workday")

        board_post = {
            "title": "Software Engineer",
            "company_profile": "Known company",
            "description": "Build backend services.",
            "requirements": "Python, SQL",
            "employment_type": "Full-time",
            "location": "Remote",
            "job_url": "https://www.linkedin.com/jobs/view/123456",
            "apply_url": "https://www.linkedin.com/jobs/view/123456",
            "telecommuting": 1,
        }
        board_result = evaluate_job_posting(board_post, ml_fraud_probability=0.25)
        self.assertTrue(board_result["is_major_job_board_domain"])
        self.assertEqual(board_result["job_board_provider"], "linkedin")

    def test_non_tech_posting_can_get_specificity_signals(self):
        posting = {
            "title": "Remote Customer Support Specialist",
            "company_profile": "E-commerce company serving North America.",
            "description": (
                "What you'll do: resolve customer issues, document case history, "
                "and collaborate with operations. You will manage SLAs and escalation queues."
            ),
            "requirements": (
                "- 2+ years of customer support experience\n"
                "- Zendesk or Freshdesk experience\n"
                "- Weekend availability and overlap hours in EST"
            ),
            "benefits": "Health benefits and PTO",
            "employment_type": "Full-time",
            "location": "Remote - US",
            "job_url": "https://careers.example.com/jobs/123",
            "apply_url": "https://careers.example.com/jobs/123",
            "telecommuting": 1,
        }
        result = evaluate_job_posting(posting, ml_fraud_probability=0.30)

        self.assertGreaterEqual(result["components"]["quality"]["role_specificity"], 7.0)
        self.assertTrue(result["matched_signals"]["years_experience_terms"])
        self.assertTrue(result["matched_signals"]["responsibility_terms"])
        self.assertTrue(result["matched_signals"]["schedule_terms"])

    def test_messaging_interview_variant_triggers_cap_25(self):
        posting = {
            "title": "Remote Client Success Specialist",
            "company_profile": "Digital services firm",
            "description": (
                "Own client onboarding and support escalations. "
                "Interview process is conducted over Telegram text only."
            ),
            "requirements": "2+ years support experience.",
            "employment_type": "Full-time",
            "location": "Remote - North America",
            "job_url": "https://www.linkedin.com/jobs/view/123456789",
            "apply_url": "https://www.linkedin.com/jobs/view/123456789",
            "telecommuting": 1,
        }
        result = evaluate_job_posting(posting, ml_fraud_probability=0.02)

        self.assertLessEqual(result["trust_score"], 25.0)
        self.assertTrue(result["matched_signals"]["messaging_app"])
        self.assertTrue(
            result["matched_signals"]["text_only_interview"]
            or result["matched_signals"]["interview_over_messaging"]
        )
        self.assertTrue(any(cap["cap"] == 25 for cap in result["hard_caps"]))

    def test_equipment_check_variant_triggers_fake_check_path(self):
        posting = {
            "title": "Remote Finance Assistant",
            "company_profile": "Established consulting group",
            "description": (
                "We will mail a check for equipment; deposit it and send part to approved vendor."
            ),
            "requirements": "2+ years AP/AR support",
            "employment_type": "Full-time",
            "location": "Remote - US",
            "job_url": "https://myview.wd3.myworkdayjobs.com/en-US/careers/job/finance-assistant-remote",
            "apply_url": "https://myview.wd3.myworkdayjobs.com/en-US/careers/job/finance-assistant-remote",
            "telecommuting": 1,
        }
        result = evaluate_job_posting(posting, ml_fraud_probability=0.01)

        self.assertTrue(result["matched_signals"]["fake_check"])
        self.assertTrue(result["matched_signals"]["equipment_check_pattern"])
        self.assertTrue(any(cap["cap"] == 20 for cap in result["hard_caps"]))
        self.assertLessEqual(result["trust_score"], 20.0)

    def test_non_english_without_translation_softens_language_sensitive_penalties(self):
        posting = {
            "title": "Ingeniero de Datos Remoto",
            "company_profile": "Empresa tecnológica consolidada",
            "description": (
                "Este puesto remoto colabora con producto y operaciones para diseñar "
                "pipelines de datos, tableros y automatizaciones de reportes semanales."
            ),
            "requirements": (
                "Más de tres años de experiencia en análisis de datos, comunicación con "
                "equipos multifuncionales y documentación técnica de procesos."
            ),
            "employment_type": "Tiempo completo",
            "location": "Remoto - España",
            "job_url": "https://empleos.example.com/rol/123",
            "apply_url": "https://empleos.example.com/rol/123",
            "telecommuting": 1,
        }

        baseline = evaluate_job_posting(posting, ml_fraud_probability=0.80)

        posting_with_language_context = dict(posting)
        posting_with_language_context.update(
            {
                "detected_language": "es",
                "language_confidence": 0.99,
                "translation_applied": False,
            }
        )
        softened = evaluate_job_posting(posting_with_language_context, ml_fraud_probability=0.80)

        self.assertTrue(softened["language_normalization_softened"])
        self.assertGreater(softened["trust_score"], baseline["trust_score"])
        self.assertIn("translation unavailable", " ".join(softened["positive_evidence"]).lower())

    def test_quality_rewards_benefits_described_in_description_text(self):
        posting = {
            "title": "Senior Backend Engineer",
            "company_profile": "Remote-first company",
            "description": (
                "About us: We build workflow software for distributed teams. "
                "Compensation range is discussed during hiring. "
                "Benefits include medical, dental, vision, generous PTO, "
                "parental leave, a 401(k) match, and equity grants."
            ),
            "requirements": "5+ years backend engineering experience.",
            "benefits": "",
            "employment_type": "Full-time",
            "location": "Remote - US",
            "job_url": "https://jobs.lever.co/example/123",
            "apply_url": "https://jobs.lever.co/example/123",
            "telecommuting": 1,
        }

        result = evaluate_job_posting(posting, ml_fraud_probability=0.20)

        self.assertGreaterEqual(result["components"]["quality"]["transparency"], 20.0)
        self.assertGreaterEqual(result["benefit_category_count"], 3)
        self.assertTrue(result["matched_signals"]["benefit_categories"])
        self.assertIn("benefits/perks are described", " ".join(result["positive_evidence"]).lower())

    def test_sparse_requirements_can_still_score_when_description_has_sections(self):
        posting = {
            "title": "Product Manager",
            "company_profile": "Global SaaS company",
            "description": (
                "About us: We are a distributed team building B2B workflow tools. "
                "What you'll do: Partner with engineering and design to ship roadmap priorities. "
                "You will define KPIs, lead discovery, and manage execution risks across teams. "
                "Requirements: 5+ years in product management, strong stakeholder communication, "
                "experience with SaaS metrics, and prior work in remote-first product organizations. "
                "Nice to have: analytics experimentation and platform experience. "
                "" + " ".join(["additional context"] * 120)
            ),
            "requirements": "",
            "benefits": "",
            "employment_type": "Full-time",
            "location": "Remote",
            "job_url": "https://weworkremotely.com/remote-jobs/example-product-manager",
            "apply_url": "https://weworkremotely.com/remote-jobs/example-product-manager",
            "telecommuting": 1,
        }

        result = evaluate_job_posting(posting, ml_fraud_probability=0.25)

        self.assertGreaterEqual(result["components"]["quality"]["role_specificity"], 12.0)
        self.assertTrue(result["matched_signals"]["qualification_section_terms"])
        self.assertTrue(result["matched_signals"]["responsibility_terms"])
        self.assertNotIn("Requirements section is sparse", result["risk_flags"])


if __name__ == "__main__":
    unittest.main()
