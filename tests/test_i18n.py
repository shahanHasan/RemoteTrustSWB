import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.i18n import detect_language_signal, normalize_posting_language  # noqa: E402


class LanguageI18nTests(unittest.TestCase):
    def test_portuguese_latin_script_title_not_misclassified_as_english(self):
        text = "[Banco de Talentos] Pessoas com deficiência"
        signal = detect_language_signal(text)
        self.assertEqual(signal.language, "pt")
        self.assertGreaterEqual(signal.confidence, 0.70)

    def test_spanish_latin_script_text_not_misclassified_as_english(self):
        text = "Ingeniero de Datos Remoto con experiencia en analisis y equipo"
        signal = detect_language_signal(text)
        self.assertEqual(signal.language, "es")
        self.assertGreaterEqual(signal.confidence, 0.70)

    def test_normalize_posting_language_exposes_detection_metadata(self):
        posting = {
            "title": "[Banco de Talentos] Pessoas com deficiência",
            "description": "",
            "requirements": "",
            "benefits": "",
        }
        result = normalize_posting_language(posting, enable_translation=False)
        self.assertEqual(result.language, "pt")
        self.assertEqual(result.posting["detected_language"], "pt")
        self.assertIn("language_confidence", result.posting)
        self.assertFalse(result.translation_applied)

    def test_detection_uses_description_not_only_title(self):
        posting = {
            "title": "Open Role",
            "description": "Pessoas com deficiência são bem-vindas para vaga remota.",
            "requirements": "Experiência com atendimento e comunicação.",
            "benefits": "Benefícios competitivos",
        }
        result = normalize_posting_language(posting, enable_translation=False)
        self.assertEqual(result.language, "pt")
        self.assertGreaterEqual(result.language_confidence, 0.70)

    def test_translation_backend_failure_is_soft_and_explained(self):
        posting = {
            "title": "Pessoa Desenvolvedora Backend",
            "description": "Vaga remota para desenvolvimento de APIs.",
            "requirements": "Experiência com Python",
            "benefits": "Plano de saúde",
        }

        with mock.patch(
            "job_fraud_detector.i18n._translate_text_google",
            side_effect=RuntimeError("translation backend unavailable"),
        ):
            result = normalize_posting_language(posting, enable_translation=True)

        self.assertEqual(result.language, "pt")
        self.assertFalse(result.translation_applied)
        self.assertEqual(result.translation_provider, "none")
        self.assertIsNotNone(result.translation_error)
        self.assertIn("translation backend unavailable", result.translation_error)


if __name__ == "__main__":
    unittest.main()
