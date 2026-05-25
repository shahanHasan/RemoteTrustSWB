"""Language detection and optional translation helpers for live scoring."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Mapping


TRANSLATABLE_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

LANGUAGE_CONFIDENCE_THRESHOLD = 0.65
MAX_TRANSLATION_CHARS = 1800
MIN_LANGDETECT_CHARS = 8
TRANSLATION_RETRIES = 3
TRANSLATION_RETRY_SLEEP_SECONDS = 0.7
TRANSLATION_REQUEST_GAP_SECONDS = 0.12
TRANSLATION_CACHE_SIZE = 512

_TRANSLATION_CACHE: dict[tuple[str, str, str], str] = {}

SCRIPT_HINT_PATTERNS: list[tuple[str, str]] = [
    ("zh", r"[\u4e00-\u9fff]"),
    ("ja", r"[\u3040-\u30ff]"),
    ("ko", r"[\uac00-\ud7a3]"),
    ("ru", r"[\u0400-\u04ff]"),
    ("ar", r"[\u0600-\u06ff]"),
    ("he", r"[\u0590-\u05ff]"),
    ("hi", r"[\u0900-\u097f]"),
    ("th", r"[\u0e00-\u0e7f]"),
]

LATIN_KEYWORD_HINTS: dict[str, set[str]] = {
    "pt": {
        "pessoas",
        "talentos",
        "deficiencia",
        "deficiência",
        "vaga",
        "candidatura",
        "trabalho",
        "beneficios",
        "benefícios",
        "empresa",
        "experiencia",
        "experiência",
        "remoto",
        "engenheiro",
        "analista",
        "banco",
    },
    "es": {
        "personas",
        "talento",
        "discapacidad",
        "vacante",
        "trabajo",
        "beneficios",
        "empresa",
        "experiencia",
        "remoto",
        "ingeniero",
        "analista",
        "equipo",
        "descripcion",
        "descripción",
    },
    "fr": {
        "equipe",
        "équipe",
        "poste",
        "travail",
        "experience",
        "expérience",
        "ingenieur",
        "ingénieur",
        "distance",
        "benefices",
        "bénéfices",
    },
    "de": {
        "stelle",
        "arbeit",
        "erfahrung",
        "vorteile",
        "unternehmen",
        "entwicklung",
        "ingenieur",
    },
    "it": {
        "lavoro",
        "esperienza",
        "benefici",
        "azienda",
        "ingegnere",
        "remoto",
    },
}

LATIN_DIACRITIC_HINTS: list[tuple[str, str]] = [
    ("pt", r"[ãõ]|\b(?:não|ação|ações|deficiência|experiência)\b"),
    ("es", r"[ñ¡¿]|\b(?:descripción|experiencia|equipo)\b"),
    ("fr", r"\b(?:équipe|expérience|télétravail)\b"),
    ("de", r"[äöüß]"),
    ("it", r"\b(?:lavoro|esperienza|azienda)\b"),
]


@dataclass(frozen=True)
class LanguageSignal:
    """Detected language signal for a posting."""

    language: str
    confidence: float
    detector: str


@dataclass(frozen=True)
class TranslationResult:
    """Translation payload for scoring pipeline."""

    posting: dict[str, Any]
    language: str
    language_confidence: float
    language_detector: str
    translation_applied: bool
    translation_provider: str
    translation_error: str | None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _translation_chunks(text: str, max_chars: int = MAX_TRANSLATION_CHARS) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            space_idx = normalized.rfind(" ", start, end)
            if space_idx > start + 100:
                end = space_idx
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _normalize_for_detection(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"www\.\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _latin_word_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)]


def _latin_language_hint(text: str) -> LanguageSignal:
    normalized = _normalize_for_detection(text).lower()
    if not re.search(r"[a-z]", normalized):
        return LanguageSignal(language="unknown", confidence=0.0, detector="latin-keyword-heuristic")

    tokens = _latin_word_tokens(normalized)
    token_set = set(tokens)

    scores: dict[str, float] = {lang: 0.0 for lang in LATIN_KEYWORD_HINTS}

    for lang, keywords in LATIN_KEYWORD_HINTS.items():
        keyword_hits = len(token_set.intersection(keywords))
        scores[lang] += float(keyword_hits)

    for lang, pattern in LATIN_DIACRITIC_HINTS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            scores[lang] = scores.get(lang, 0.0) + 2.0

    best_lang = max(scores, key=scores.get)
    best_score = scores.get(best_lang, 0.0)
    if best_score < 2.0:
        return LanguageSignal(language="en", confidence=0.55, detector="latin-keyword-heuristic")

    confidence = min(0.95, 0.55 + 0.10 * best_score)
    return LanguageSignal(language=best_lang, confidence=confidence, detector="latin-keyword-heuristic")


def _heuristic_language(text: str) -> LanguageSignal:
    lowered = _normalize_for_detection(text).lower()
    for language, pattern in SCRIPT_HINT_PATTERNS:
        if re.search(pattern, lowered):
            return LanguageSignal(language=language, confidence=0.99, detector="script-heuristic")
    latin_guess = _latin_language_hint(lowered)
    if latin_guess.language != "unknown":
        return latin_guess
    return LanguageSignal(language="unknown", confidence=0.0, detector="heuristic")


def detect_language_signal(text: str) -> LanguageSignal:
    """Detect language with probability, falling back to script heuristics."""
    candidate = _normalize_for_detection(text)
    if not candidate:
        return LanguageSignal(language="unknown", confidence=0.0, detector="heuristic")

    script_guess = _heuristic_language(candidate)
    if script_guess.detector == "script-heuristic":
        return script_guess

    latin_hint = _latin_language_hint(candidate)
    model_guess: LanguageSignal | None = None
    try:
        if len(candidate) >= MIN_LANGDETECT_CHARS:
            from langdetect import DetectorFactory, detect_langs  # type: ignore

            DetectorFactory.seed = 0
            predictions = detect_langs(candidate[:6000])
            if predictions:
                top = predictions[0]
                model_guess = LanguageSignal(
                    language=str(top.lang).lower(),
                    confidence=float(top.prob),
                    detector="langdetect",
                )
    except Exception:
        model_guess = None

    if model_guess is None:
        return latin_hint

    # Fix common short-text failure mode: Latin non-English titles defaulting to English.
    if (
        model_guess.language == "en"
        and model_guess.confidence <= 0.80
        and latin_hint.language not in {"en", "unknown"}
        and latin_hint.confidence >= 0.70
    ):
        return LanguageSignal(
            language=latin_hint.language,
            confidence=max(latin_hint.confidence, model_guess.confidence),
            detector="langdetect+latin-override",
        )

    # If langdetect is weak/ambiguous, trust Latin lexical hint when it is stronger.
    if (
        latin_hint.language not in {"en", "unknown"}
        and latin_hint.confidence >= model_guess.confidence + 0.10
    ):
        return LanguageSignal(
            language=latin_hint.language,
            confidence=latin_hint.confidence,
            detector="latin-heuristic-priority",
        )

    return model_guess


def _normalize_source_lang(code: str) -> str:
    if not code:
        return "auto"
    code = code.lower()
    if code in {"zh-cn", "zh-tw"}:
        return "zh-CN" if code == "zh-cn" else "zh-TW"
    return code


def _translate_chunk_with_retry(
    chunk: str,
    source_lang: str,
    target_lang: str,
    retries: int = TRANSLATION_RETRIES,
    sleep_seconds: float = TRANSLATION_RETRY_SLEEP_SECONDS,
) -> str:
    try:
        from deep_translator import GoogleTranslator  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "translation backend unavailable: install deep-translator in the active environment"
        ) from exc

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(chunk)
            return str(translated).strip()
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"translation failed for source={source_lang}") from last_error


def _cache_get_translation(text: str, source_lang: str, target_lang: str) -> str | None:
    key = (source_lang, target_lang, text)
    return _TRANSLATION_CACHE.get(key)


def _cache_set_translation(text: str, source_lang: str, target_lang: str, translated: str) -> None:
    if len(_TRANSLATION_CACHE) >= TRANSLATION_CACHE_SIZE:
        oldest_key = next(iter(_TRANSLATION_CACHE))
        _TRANSLATION_CACHE.pop(oldest_key, None)
    key = (source_lang, target_lang, text)
    _TRANSLATION_CACHE[key] = translated


def _translate_text_google(text: str, source_lang: str, target_lang: str = "en") -> str:
    chunks = _translation_chunks(text)
    if not chunks:
        return ""
    normalized_source = _normalize_source_lang(source_lang)

    translated_chunks: list[str] = []
    for idx, chunk in enumerate(chunks):
        cached = _cache_get_translation(chunk, normalized_source, target_lang)
        if cached is not None:
            translated_chunks.append(cached)
            continue

        # First attempt: use detected source language.
        try:
            translated = _translate_chunk_with_retry(
                chunk=chunk,
                source_lang=normalized_source,
                target_lang=target_lang,
            )
        except Exception:
            # Fallback: auto-detect source language at translator level.
            translated = _translate_chunk_with_retry(
                chunk=chunk,
                source_lang="auto",
                target_lang=target_lang,
            )
        if translated:
            _cache_set_translation(chunk, normalized_source, target_lang, translated)
            translated_chunks.append(translated)
        if idx < len(chunks) - 1:
            time.sleep(TRANSLATION_REQUEST_GAP_SECONDS)

    return "\n\n".join(translated_chunks)


def _translation_detection_text(posting: Mapping[str, Any]) -> str:
    fields = ["title", "description", "requirements", "benefits"]
    joined = " ".join(_clean_text(posting.get(field, "")) for field in fields)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def normalize_posting_language(
    posting: Mapping[str, Any],
    enable_translation: bool = True,
    target_language: str = "en",
) -> TranslationResult:
    """Detect language and optionally translate key fields to English."""
    posting_copy = dict(posting)
    detection_text = _translation_detection_text(posting_copy)
    signal = detect_language_signal(detection_text)

    non_english = signal.language not in {"en", "unknown"}
    confident_non_english = non_english and signal.confidence >= LANGUAGE_CONFIDENCE_THRESHOLD

    translation_applied = False
    translation_error: str | None = None
    translation_provider = "none"

    if enable_translation and confident_non_english:
        field_errors: list[str] = []
        translated_any_field = False
        for field in TRANSLATABLE_FIELDS:
            raw = _clean_text(posting_copy.get(field, ""))
            if not raw:
                continue
            try:
                translated_value = _translate_text_google(
                    raw,
                    source_lang=signal.language,
                    target_lang=target_language,
                )
                if translated_value:
                    posting_copy[field] = translated_value
                    if translated_value.strip() != raw.strip():
                        translated_any_field = True
            except Exception as exc:
                error_text = str(exc)
                if "translation backend unavailable" in error_text.lower():
                    field_errors = [error_text]
                    break
                field_errors.append(f"{field}: {error_text}")

        translation_applied = translated_any_field
        if translated_any_field:
            translation_provider = "deep-translator-google"
        if field_errors:
            translation_error = " | ".join(field_errors[:3])

    posting_copy["detected_language"] = signal.language
    posting_copy["language_confidence"] = round(signal.confidence, 4)
    posting_copy["language_detector"] = signal.detector
    posting_copy["translation_applied"] = bool(translation_applied)
    posting_copy["translation_provider"] = translation_provider
    posting_copy["translation_error"] = translation_error

    return TranslationResult(
        posting=posting_copy,
        language=signal.language,
        language_confidence=signal.confidence,
        language_detector=signal.detector,
        translation_applied=translation_applied,
        translation_provider=translation_provider,
        translation_error=translation_error,
    )
