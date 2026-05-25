"""Heuristic trust/quality scoring for live job postings.

This module combines government-guidance-inspired fraud red flags with
job-quality indicators into a transparent scoring layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any, Mapping
from urllib.parse import urlparse

TRUST_BLEND_WEIGHTS = {
    "rule_engine_score": 0.45,
    "ml_legitimacy_score": 0.55,
}

FINAL_BLEND_WEIGHTS = {
    "trust_score": 0.70,
    "quality_score": 0.30,
}

TRUST_COMPONENT_MAXIMA = {
    "identity_apply_integrity": 30.0,
    "communication_safety": 25.0,
    "monetary_safety": 20.0,
    "company_evidence": 15.0,
}

QUALITY_COMPONENT_MAXIMA = {
    "transparency": 30.0,
    "role_specificity": 25.0,
    "remote_clarity": 20.0,
    "apply_experience": 15.0,
    "freshness": 10.0,
}

KNOWN_ATS_DOMAINS = {
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "greenhouse.io",
    "jobs.lever.co",
    "hire.lever.co",
    "jobs.ashbyhq.com",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "myworkdaysite.com",
    "taleo.net",
    "icims.com",
    "smartrecruiters.com",
    "jobvite.com",
    "successfactors.com",
    "oraclecloud.com",
    "workable.com",
    "bamboohr.com",
    "teamtailor.com",
    "recruitee.com",
}

KNOWN_TRUSTED_BOARD_DOMAINS = {
    "weworkremotely.com",
    "jobicy.com",
    "remotive.com",
    "usajobs.gov",
}

MAJOR_JOB_BOARD_DOMAINS = {
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "monster.com",
    "simplyhired.com",
}

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "aol.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "gmx.com",
    "mail.com",
}

URL_SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "shorturl.at",
    "rebrand.ly",
    "is.gd",
    "rb.gy",
}

MESSAGING_APP_TERMS = [
    r"\bwhatsapp\b",
    r"\btelegram\b",
    r"\bsignal\b",
    r"\bdiscord\b",
    r"\bmessenger\b",
    r"\breply\s+(yes|interested)\b",
]

TEXT_INTERVIEW_TERMS = [
    r"\btext\s*[- ]?only\s+interview\b",
    r"\btext\s*[- ]?based\s+interview\b",
    r"\bsms\s+interview\b",
    r"\btext\s+message\s+interview\b",
    r"\binterview\s+over\s+text\b",
    r"\binterview\s+via\s+chat\b",
    r"\bchat\s+interview\s+only\b",
    r"\b(?:text|chat)\s*[- ]?only\s+(?:interview|screening)\b",
]

INTERVIEW_OVER_MESSAGING_TERMS = [
    r"\binterview(?:\s+process)?\s+(?:is\s+)?(?:conducted|held|done|completed)?\s*(?:over|via|through|on|in)\s+(?:telegram|whatsapp|signal|discord|messenger)\b",
    r"\b(?:telegram|whatsapp|signal|discord|messenger)\s+(?:interview|chat\s+interview|text\s+interview)\b",
    r"\b(?:telegram|whatsapp|signal|discord|messenger)\s+(?:text|chat)\s*[- ]?only\b",
]

PAY_TO_GET_PAID_TERMS = [
    r"\bpay\s+to\s+get\s+paid\b",
    r"\bpay\s+to\s+start\b",
    r"\bapplication\s+fee\b",
    r"\bupfront\s+fee\b",
    r"\bregistration\s+fee\b",
    r"\bonboarding\s+fee\b",
    r"\btraining\s+fee\b",
    r"\bsecurity\s+deposit\b",
    r"\bactivation\s+fee\b",
    r"\bunlock\s+(payment|fee)\b",
    r"\bupgrade\s+fee\b",
]

FAKE_CHECK_TERMS = [
    r"\bfake\s+check\b",
    r"\bcounterfeit\s+cheque?\b",
    r"\bdeposit\s+(a\s+)?check\b",
    r"\bdeposit\s+it\b",
    r"\bmobile\s+deposit\b",
    r"\bcash\s+(a\s+)?check\b",
    r"\bsend\s+(some\s+of\s+the\s+money\s+)?back\b",
    r"\bsend\s+(part|portion|some)\s+(to|of)\s+(a\s+)?vendor\b",
    r"\bcheck\s+will\s+be\s+mailed\b",
    r"\bmailed?\s+(you\s+)?(a\s+)?check\b",
    r"\boverpayment\b",
    r"\bmystery\s+shopper\b",
    r"\bsecret\s+shopper\b",
]

TASK_SCAM_TERMS = [
    r"\btask\s+scam\b",
    r"\bproduct\s+boost(ing)?\b",
    r"\bapp\s+optimization\b",
    r"\bonline\s+tasks?\b",
    r"\bcommission\s+per\s+click\b",
    r"\bweb\s+surveys?\b",
    r"\brating\s+tasks?\b",
    r"\breview\s+tasks?\b",
]

BANK_INFO_TERMS = [
    r"\bbank\s+account\b",
    r"\brouting\s+number\b",
    r"\baccount\s+number\b",
    r"\bsocial\s+security\s+number\b",
    r"\bssn\b",
    r"\bsin\b",
    r"\bpassport\b",
    r"\bdriver'?s\s+license\b",
    r"\bcredit\s+card\b",
]

CRYPTO_OR_GIFTCARD_TERMS = [
    r"\bcrypto(currency)?\b",
    r"\bbitcoin\b",
    r"\bethereum\b",
    r"\busdt\b",
    r"\bgift\s+cards?\b",
]

MONEY_MULE_TERMS = [
    r"\breceive\s+payments?\b",
    r"\bforward\s+money\b",
    r"\bpayment\s+processor\b",
    r"\bfinancial\s+agent\b",
    r"\bwire\s+transfer\b",
]

RECRUIT_OTHERS_TERMS = [
    r"\brecruit\s+others\b",
    r"\bdownline\b",
    r"\bpyramid\s+selling\b",
    r"\bmlm\b",
    r"\bmulti\s*[- ]?level\s+marketing\b",
]

GENERIC_JOB_TERMS = [
    r"\bno\s+experience\s+needed\b",
    r"\bimmediate\s+start\b",
    r"\bwork\s+from\s+your\s+phone\b",
    r"\beasy\s+money\b",
    r"\bguaranteed\s+income\b",
    r"\bearn\s+\$?\d+\s*(per\s+day|daily|weekly)\b",
    r"\bdaily\s+pay\b",
]

URGENCY_PRESSURE_TERMS = [
    r"\bact\s+now\b",
    r"\bstart\s+today\b",
    r"\blimited\s+slots?\b",
    r"\bimmediate\s+hiring\b",
    r"\bonly\s+\d+\s+spots?\b",
    r"\burgent(ly)?\b",
]

NO_INTERVIEW_TERMS = [
    r"\bno\s+interview\b",
    r"\binstant\s+hire\b",
    r"\bauto[- ]?approved\b",
    r"\bguaranteed\s+hire\b",
]

EQUIPMENT_CHECK_TERMS = [
    r"\bequipment\s+reimbursement\b",
    r"\bbuy\s+(your|the)\s+equipment\b",
    r"\bhome\s+office\s+kit\b",
    r"\bcheck\s+for\s+equipment\b",
    r"\bequipment\s+check\b",
    r"\bcheck\s+for\s+home\s+office\b",
    r"\bbuy\s+from\s+(an?\s+)?approved\s+vendor\b",
    r"\bsend\s+money\s+to\s+(an?\s+)?approved\s+vendor\b",
    r"\bvendor\s+for\s+equipment\b",
]

TIMEZONE_OR_REGION_TERMS = [
    r"\btimezone\b",
    r"\btime\s+zone\b",
    r"\bnorth\s+america\b",
    r"\bwithin\s+(the\s+)?(us|u\.s\.|usa|canada|eu|europe)\b",
    r"\bmust\s+reside\s+in\b",
    r"\bus[- ]based\b",
    r"\bcanada[- ]based\b",
]

ROLE_SPECIFICITY_TERMS = [
    # Engineering / data / analytics
    r"\bpython\b",
    r"\bsql\b",
    r"\bexcel\b",
    r"\btableau\b",
    r"\bpower\s*bi\b",
    r"\baws\b",
    r"\bgcp\b",
    r"\bjavascript\b",
    r"\breact\b",
    r"\bjava\b",
    r"\bc\+\+\b",
    r"\bdbt\b",
    r"\bairflow\b",
    r"\bsnowflake\b",
    r"\blooker\b",
    r"\bjira\b",
    r"\bconfluence\b",
    # Design / content
    r"\bfigma\b",
    r"\badobe\b",
    r"\bphotoshop\b",
    r"\billustrator\b",
    r"\bpremiere\b",
    r"\bindesign\b",
    r"\bcanva\b",
    r"\bwordpress\b",
    r"\bseo\b",
    r"\bsem\b",
    r"\bgoogle\s+analytics\b",
    # Sales / support / CX / operations
    r"\bsalesforce\b",
    r"\bhubspot\b",
    r"\bzendesk\b",
    r"\bintercom\b",
    r"\bfreshdesk\b",
    r"\bcrm\b",
    r"\berp\b",
    r"\bquickbooks\b",
    r"\bnetsuite\b",
    r"\bsap\b",
    r"\bshopify\b",
    r"\bamazon\s+seller\s+central\b",
    # Healthcare / regulated roles
    r"\bemr\b",
    r"\behr\b",
    r"\bepic\b",
    r"\bhipaa\b",
    r"\brn\b",
    r"\blpn\b",
    r"\bcna\b",
    r"\bcdl\b",
    # HR / recruiting
    r"\bworkday\b",
    r"\bgreenhouse\b",
    r"\blever\b",
    r"\bats\b",
]

YEARS_EXPERIENCE_TERMS = [
    r"\b\d+\+?\s+years?\s+of\s+experience\b",
    r"\b\d+\+?\s+years?\s+experience\b",
    r"\b\d+\+?\s+years?\s+of\s+[a-zA-Z0-9,\-\/\s]{1,60}\s+experience\b",
    r"\bminimum\s+\d+\+?\s+years?\b",
    r"\b\d+\+?\s+yrs?\b",
]

CREDENTIAL_OR_REQUIREMENT_TERMS = [
    r"\b(certification|certified|license|licensed|credential)\b",
    r"\b(bachelor'?s|master'?s|ph\.?d|diploma|degree)\b",
    r"\bbackground\s+check\b",
    r"\bwork\s+authorization\b",
]

RESPONSIBILITY_STRUCTURE_TERMS = [
    r"\bresponsibilit(y|ies)\b",
    r"\bkey\s+duties\b",
    r"\bwhat\s+you(?:'ll|\s+will)\s+do\b",
    r"\bday[- ]to[- ]day\b",
    r"\byou\s+will\b",
]

SCHEDULE_OR_AVAILABILITY_TERMS = [
    r"\bshift\b",
    r"\bweekend(s)?\b",
    r"\bon[- ]call\b",
    r"\bbusiness\s+hours\b",
    r"\bavailability\b",
    r"\boverlap\s+hours\b",
]

BENEFIT_CATEGORY_PATTERNS: dict[str, str] = {
    "health": r"\b(?:health|medical|dental|vision|insurance)\b",
    "time_off": r"\b(?:pto|paid\s+time\s+off|vacation|holiday)s?\b",
    "equity": r"\b(?:equity|stock|rsu|options)\b",
    "retirement": r"\b(?:401\(k\)|pension|rrsp)\b",
    "family": r"\b(?:parental|maternity|paternity)\b",
    "learning": r"\b(?:learning\s+budget|tuition|professional\s+development|training)\b",
    "wellness": r"\b(?:wellness|mental\s+health|eap)\b",
}

QUALIFICATION_SECTION_TERMS = [
    r"\brequirements?\b",
    r"\bqualifications?\b",
    r"\bmust\s+have\b",
    r"\bnice\s+to\s+have\b",
    r"\bwho\s+you\s+are\b",
    r"\bwhat\s+you(?:'ll|\s+will)\s+bring\b",
]

HIRING_PROCESS_TERMS = [
    r"\bhiring\s+process\b",
    r"\binterview\s+process\b",
    r"\bapplication\s+process\b",
    r"\bnext\s+steps\b",
    r"\bhow\s+to\s+apply\b",
]

REMOTE_POLICY_CLARITY_TERMS = [
    r"\bworldwide\b",
    r"\bglobal\b",
    r"\banywhere\b",
    r"\bremote[- ]?first\b",
    r"\bdistributed\b",
    r"\bcore\s+hours\b",
    r"\boverlap\s+hours\b",
    r"\b(?:async|asynchronous)\b",
    r"\btime\s*zone\b",
]

COMPENSATION_CLARITY_TERMS = [
    r"\bsalary\s+range\b",
    r"\bpay\s+range\b",
    r"\bcompensation\s+range\b",
    r"\bper\s+(hour|year|annum)\b",
    r"\bhourly\b",
    r"\bannually\b",
    r"\btotal\s+compensation\b",
]

COMPANY_CONTEXT_TERMS = [
    r"\babout\s+us\b",
    r"\babout\s+the\s+company\b",
    r"\bour\s+mission\b",
    r"\bour\s+values\b",
    r"\bour\s+culture\b",
    r"\bwho\s+we\s+are\b",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


def _domain_of(url: str) -> str:
    if not url:
        return ""
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    return parsed.netloc.lower().split(":")[0].lstrip("www.")


def _extract_text(posting: Mapping[str, Any]) -> str:
    fields = [
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
    return "\n".join(_clean(posting.get(field, "")) for field in fields).lower()


def _language_context(posting: Mapping[str, Any]) -> tuple[str, float, bool]:
    language = _clean(posting.get("detected_language", "")).lower()
    confidence_raw = posting.get("language_confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    translation_applied = bool(posting.get("translation_applied", False))
    return language, confidence, translation_applied


def _find_matches(text: str, patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(pattern)
    return matched


def _extract_email_domains(text: str) -> list[str]:
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    normalized = [domain.lower() for domain in emails]
    return sorted(set(normalized))


def _infer_ats_provider(domain: str) -> str:
    if not domain:
        return ""
    if "greenhouse" in domain:
        return "greenhouse"
    if "lever.co" in domain or "lever" in domain:
        return "lever"
    if "ashby" in domain:
        return "ashby"
    if "myworkdayjobs.com" in domain or "myworkdaysite.com" in domain:
        return "workday"
    if "taleo.net" in domain:
        return "taleo"
    if "icims.com" in domain:
        return "icims"
    if "smartrecruiters.com" in domain:
        return "smartrecruiters"
    if "jobvite.com" in domain:
        return "jobvite"
    if "successfactors.com" in domain:
        return "successfactors"
    if "oraclecloud.com" in domain:
        return "oracle-hcm"
    if "workable.com" in domain:
        return "workable"
    if "bamboohr.com" in domain:
        return "bamboohr"
    if "teamtailor.com" in domain:
        return "teamtailor"
    if "recruitee.com" in domain:
        return "recruitee"
    return ""


def _infer_job_board_provider(domain: str) -> str:
    if not domain:
        return ""
    if "linkedin.com" in domain:
        return "linkedin"
    if "indeed.com" in domain:
        return "indeed"
    if "glassdoor.com" in domain:
        return "glassdoor"
    if "ziprecruiter.com" in domain:
        return "ziprecruiter"
    if "monster.com" in domain:
        return "monster"
    if "simplyhired.com" in domain:
        return "simplyhired"
    return ""


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _has_salary_signal(posting: Mapping[str, Any], text: str) -> bool:
    salary_field = _clean(posting.get("salary_range", ""))
    if salary_field:
        return True
    return bool(
        re.search(
            r"\$\s?\d[\d,]*(\s*-\s*\$?\s?\d[\d,]*)?",
            text,
            flags=re.IGNORECASE,
        )
    )


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    cleaned = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _days_old(posting: Mapping[str, Any]) -> int | None:
    for field in ("posted_date", "publication_date", "pub_date", "date_posted"):
        dt = _parse_date(_clean(posting.get(field, "")))
        if dt is not None:
            delta = datetime.now(UTC) - dt
            return max(0, int(delta.total_seconds() // 86400))
    return None


def _format_pattern_names(patterns: list[str]) -> list[str]:
    cleaned: list[str] = []
    for pattern in patterns:
        label = pattern.replace("\\b", "")
        label = label.replace("\\s+", " ")
        label = label.replace("\\s*", " ")
        label = label.replace("\\", "")
        label = label.replace("(", "").replace(")", "")
        label = label.strip("^$")
        cleaned.append(label.strip())
    return sorted(set(filter(None, cleaned)))


def _benefit_categories_detected(text: str) -> list[str]:
    categories: list[str] = []
    for category, pattern in BENEFIT_CATEGORY_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            categories.append(category)
    return categories


def evaluate_job_posting(
    posting: Mapping[str, Any],
    ml_fraud_probability: float | None = None,
) -> dict[str, Any]:
    """Return trust/quality heuristic scoring and evidence for one posting.

    Scoring shape:
    - Trust score blends rule score + ML legitimacy score.
    - Quality score evaluates posting completeness and apply UX.
    - Final score ranks opportunities with trust-first weighting.
    """

    text = _extract_text(posting)
    detected_language, language_confidence, translation_applied = _language_context(posting)
    non_english_without_translation = (
        detected_language not in {"", "unknown", "en"}
        and language_confidence >= 0.60
        and not translation_applied
    )

    title = _clean(posting.get("title", ""))
    description = _clean(posting.get("description", ""))
    requirements = _clean(posting.get("requirements", ""))
    benefits = _clean(posting.get("benefits", ""))
    company = _clean(posting.get("company_profile", ""))
    employment_type = _clean(posting.get("employment_type", ""))
    location = _clean(posting.get("location", ""))

    apply_url = _clean(posting.get("apply_url", ""))
    if not apply_url:
        apply_url = _clean(posting.get("job_url", ""))

    apply_domain = _domain_of(apply_url)
    source_domain = _domain_of(_clean(posting.get("job_url", "")))

    is_known_ats = any(
        apply_domain == ats or apply_domain.endswith(f".{ats}")
        for ats in KNOWN_ATS_DOMAINS
    )
    is_known_board = any(
        apply_domain == trusted or apply_domain.endswith(f".{trusted}")
        for trusted in KNOWN_TRUSTED_BOARD_DOMAINS
    )
    is_major_job_board = any(
        apply_domain == board or apply_domain.endswith(f".{board}")
        for board in MAJOR_JOB_BOARD_DOMAINS
    )
    is_shortener = any(
        apply_domain == short or apply_domain.endswith(f".{short}")
        for short in URL_SHORTENER_DOMAINS
    )

    messaging_hits = _find_matches(text, MESSAGING_APP_TERMS)
    text_interview_hits = _find_matches(text, TEXT_INTERVIEW_TERMS)
    interview_over_messaging_hits = _find_matches(text, INTERVIEW_OVER_MESSAGING_TERMS)
    pay_hits = _find_matches(text, PAY_TO_GET_PAID_TERMS)
    fake_check_hits = _find_matches(text, FAKE_CHECK_TERMS)
    task_hits = _find_matches(text, TASK_SCAM_TERMS)
    bank_hits = _find_matches(text, BANK_INFO_TERMS)
    crypto_gift_hits = _find_matches(text, CRYPTO_OR_GIFTCARD_TERMS)
    mule_hits = _find_matches(text, MONEY_MULE_TERMS)
    recruit_hits = _find_matches(text, RECRUIT_OTHERS_TERMS)
    generic_hits = _find_matches(text, GENERIC_JOB_TERMS)
    urgency_hits = _find_matches(text, URGENCY_PRESSURE_TERMS)
    no_interview_hits = _find_matches(text, NO_INTERVIEW_TERMS)
    equipment_hits = _find_matches(text, EQUIPMENT_CHECK_TERMS)
    timezone_hits = _find_matches(text, TIMEZONE_OR_REGION_TERMS)
    specificity_hits = _find_matches(text, ROLE_SPECIFICITY_TERMS)
    experience_hits = _find_matches(text, YEARS_EXPERIENCE_TERMS)
    credential_hits = _find_matches(text, CREDENTIAL_OR_REQUIREMENT_TERMS)
    responsibility_hits = _find_matches(text, RESPONSIBILITY_STRUCTURE_TERMS)
    schedule_hits = _find_matches(text, SCHEDULE_OR_AVAILABILITY_TERMS)
    qualification_section_hits = _find_matches(text, QUALIFICATION_SECTION_TERMS)
    hiring_process_hits = _find_matches(text, HIRING_PROCESS_TERMS)
    remote_policy_hits = _find_matches(text, REMOTE_POLICY_CLARITY_TERMS)
    compensation_clarity_hits = _find_matches(text, COMPENSATION_CLARITY_TERMS)
    company_context_hits = _find_matches(text, COMPANY_CONTEXT_TERMS)
    benefit_categories = _benefit_categories_detected(text)
    benefit_category_count = len(benefit_categories)
    email_domains = _extract_email_domains(text)
    free_email_domains = [domain for domain in email_domains if domain in FREE_EMAIL_DOMAINS]
    messaging_interview_semantic = bool(
        messaging_hits
        and (
            text_interview_hits
            or interview_over_messaging_hits
            or (
                re.search(r"\binterview\b", text, flags=re.IGNORECASE)
                and re.search(r"\b(text|chat|sms)\b", text, flags=re.IGNORECASE)
            )
        )
    )

    positive_evidence: list[str] = []
    risk_flags: list[str] = []

    if non_english_without_translation:
        positive_evidence.append(
            "Non-English posting detected; translation unavailable, so language-sensitive rules were softened"
        )

    # Trust components (0-100, weighted later)
    identity_score = 0.0
    communication_score = 25.0
    monetary_score = 20.0
    company_score = 0.0

    # Identity/apply integrity (max 30)
    if apply_url:
        identity_score += 8
        if _normalize_url(apply_url).startswith("https://"):
            identity_score += 4
            positive_evidence.append("Apply link uses HTTPS")
        else:
            identity_score -= 3
            risk_flags.append("Apply link is not HTTPS")
        if is_known_ats:
            identity_score += 12
            positive_evidence.append(
                f"Apply domain matches known ATS infrastructure ({apply_domain})"
            )
        elif is_known_board:
            identity_score += 8
            positive_evidence.append(
                f"Apply domain is from a known job platform ({apply_domain})"
            )
        elif is_major_job_board:
            identity_score += 5
            positive_evidence.append(
                f"Apply domain is a major job board ({apply_domain})"
            )
        if source_domain and apply_domain and source_domain == apply_domain:
            identity_score += 6
            positive_evidence.append("Apply domain matches posting domain")
        elif source_domain and apply_domain and not (is_known_ats or is_known_board or is_major_job_board):
            identity_score -= 4
            risk_flags.append("Apply domain differs from posting domain")
        if is_shortener:
            identity_score -= 12
            risk_flags.append("Apply URL uses a shortener domain")
    else:
        risk_flags.append("No apply URL found")
        identity_score -= 10

    if "redirect=" in apply_url.lower() or "url=" in apply_url.lower():
        identity_score -= 4
        risk_flags.append("Apply link appears to include redirection parameters")

    identity_score = _clip(identity_score, 0, 30)

    # Communication safety (max 25)
    if messaging_hits:
        communication_score -= 12
        risk_flags.append("Messaging-app recruitment language detected")
    if text_interview_hits:
        communication_score -= 10
        risk_flags.append("Text-only interview language detected")
    if interview_over_messaging_hits:
        communication_score -= 8
        risk_flags.append("Interview via messaging-app channel language detected")
    if generic_hits:
        communication_score -= 6
        risk_flags.append("Generic high-pay/low-detail recruiting language detected")
    if urgency_hits:
        communication_score -= 4
        risk_flags.append("High-pressure urgency language detected")
    if no_interview_hits:
        communication_score -= 6
        risk_flags.append("No-interview or instant-hire language detected")
    if free_email_domains:
        communication_score -= 5
        risk_flags.append("Free webmail contact domain detected in posting text")

    communication_score = _clip(communication_score, 0, 25)

    # Monetary safety (max 20)
    if pay_hits:
        monetary_score -= 20
        risk_flags.append("Pay-to-work / upfront fee language detected")
    if fake_check_hits:
        monetary_score -= 18
        risk_flags.append("Fake-check pattern language detected")
    if task_hits:
        monetary_score -= 18
        risk_flags.append("Task-scam / product-boosting language detected")
    if bank_hits:
        monetary_score -= 8
        risk_flags.append("Early sensitive banking/identity request language detected")
    if crypto_gift_hits:
        monetary_score -= 8
        risk_flags.append("Crypto or gift-card payment language detected")
    if mule_hits:
        monetary_score -= 10
        risk_flags.append("Payment-forwarding/money-mule language detected")
    if recruit_hits:
        monetary_score -= 8
        risk_flags.append("Recruit-others / pyramid-style language detected")
    if equipment_hits:
        monetary_score -= 10
        risk_flags.append("Equipment-purchase/reimbursement-check language detected")

    monetary_score = _clip(monetary_score, 0, 20)

    # Company evidence (max 15)
    if company:
        company_score += 5
        positive_evidence.append("Company information is present")
    if location:
        company_score += 3
    if len(description.split()) >= 80:
        company_score += 4
    if len(requirements.split()) >= 30:
        company_score += 3

    company_score = _clip(company_score, 0, 15)

    rule_engine_score = identity_score + communication_score + monetary_score + company_score
    rule_engine_score = _clip(rule_engine_score, 0, 100)

    if ml_fraud_probability is None:
        ml_legitimacy_score = 50.0
    else:
        ml_legitimacy_score = _clip((1.0 - float(ml_fraud_probability)) * 100.0, 0, 100)
        # English-trained classifier confidence can degrade on out-of-language text.
        # Pull score toward neutral when non-English text could not be translated.
        if non_english_without_translation:
            ml_legitimacy_score = 0.5 * ml_legitimacy_score + 25.0

    trust_raw = (
        TRUST_BLEND_WEIGHTS["rule_engine_score"] * rule_engine_score
        + TRUST_BLEND_WEIGHTS["ml_legitimacy_score"] * ml_legitimacy_score
    )

    hard_caps: list[dict[str, Any]] = []
    trust_cap = 100.0

    if pay_hits:
        trust_cap = min(trust_cap, 10.0)
        hard_caps.append(
            {
                "cap": 10,
                "reason": "Upfront fee / pay-to-get-paid language detected",
                "matched_signals": _format_pattern_names(pay_hits),
            }
        )
    if fake_check_hits:
        trust_cap = min(trust_cap, 20.0)
        hard_caps.append(
            {
                "cap": 20,
                "reason": "Fake-check scam pattern detected",
                "matched_signals": _format_pattern_names(fake_check_hits),
            }
        )
    if task_hits:
        trust_cap = min(trust_cap, 20.0)
        hard_caps.append(
            {
                "cap": 20,
                "reason": "Task-scam pattern detected",
                "matched_signals": _format_pattern_names(task_hits),
            }
        )
    if messaging_interview_semantic:
        trust_cap = min(trust_cap, 25.0)
        hard_caps.append(
            {
                "cap": 25,
                "reason": "Messaging-app + interview-via-text/chat pattern detected",
                "matched_signals": _format_pattern_names(
                    messaging_hits + text_interview_hits + interview_over_messaging_hits
                ),
            }
        )
    if equipment_hits and fake_check_hits:
        trust_cap = min(trust_cap, 20.0)
        hard_caps.append(
            {
                "cap": 20,
                "reason": "Equipment reimbursement + fake-check pattern detected",
                "matched_signals": _format_pattern_names(equipment_hits + fake_check_hits),
            }
        )

    trust_score = min(trust_raw, trust_cap)

    # Quality components (100 total)
    transparency_score = 0.0  # max 30
    role_specificity_score = 0.0  # max 25
    remote_clarity_score = 0.0  # max 20
    apply_experience_score = 0.0  # max 15
    freshness_score = 0.0  # max 10

    # Transparency
    if _has_salary_signal(posting, text):
        transparency_score += 10
        positive_evidence.append("Salary signal is present")
    elif compensation_clarity_hits:
        transparency_score += 4
        positive_evidence.append("Compensation details are described")
    if employment_type:
        transparency_score += 5
    if location:
        transparency_score += 4
    if company:
        transparency_score += 6
    if company_context_hits:
        transparency_score += 2
        positive_evidence.append("Company context is clearly described")
    benefits_available = bool(benefits) or benefit_category_count >= 1
    if benefits_available:
        transparency_score += 5
        if benefits:
            positive_evidence.append("Benefits section is explicitly provided")
        else:
            positive_evidence.append("Benefits/perks are described in posting text")
    if benefit_category_count >= 3:
        transparency_score += 2
        positive_evidence.append("Benefit package covers multiple categories")
    transparency_score = _clip(transparency_score, 0, 30)

    # Role specificity
    desc_words = len(description.split())
    req_words = len(requirements.split())
    if desc_words >= 150:
        role_specificity_score += 10
    elif desc_words >= 80:
        role_specificity_score += 6
    else:
        if non_english_without_translation and desc_words >= 50:
            role_specificity_score += 4
        else:
            risk_flags.append("Description appears too short to evaluate role clearly")

    embedded_qualification_signal = bool(qualification_section_hits or experience_hits or credential_hits)

    if req_words >= 60:
        role_specificity_score += 8
    elif req_words >= 30:
        role_specificity_score += 5
    elif desc_words >= 320 and embedded_qualification_signal:
        role_specificity_score += 8
        positive_evidence.append("Qualification details are embedded in description")
    elif desc_words >= 220 and embedded_qualification_signal:
        role_specificity_score += 5
        positive_evidence.append("Qualification details are embedded in description")
    else:
        if non_english_without_translation and req_words >= 18:
            role_specificity_score += 4
        elif desc_words >= 300 and (
            responsibility_hits
            or experience_hits
            or len(specificity_hits) >= 2
            or qualification_section_hits
        ):
            role_specificity_score += 4
            positive_evidence.append("Detailed description offsets sparse requirements section")
        else:
            risk_flags.append("Requirements section is sparse")

    if re.search(r"\n\s*[-*•]", _clean(posting.get("requirements", ""))):
        role_specificity_score += 3

    specificity_signal_count = 0
    if len(specificity_hits) >= 2:
        specificity_signal_count += 1
    if experience_hits:
        specificity_signal_count += 1
    if credential_hits:
        specificity_signal_count += 1
    if responsibility_hits:
        specificity_signal_count += 1
    if schedule_hits:
        specificity_signal_count += 1

    if specificity_signal_count >= 2:
        role_specificity_score += 2
    if specificity_signal_count >= 3:
        role_specificity_score += 2
    if specificity_signal_count >= 4:
        role_specificity_score += 1
    if responsibility_hits and qualification_section_hits:
        role_specificity_score += 2
        positive_evidence.append("Role includes both responsibilities and qualification cues")
    if non_english_without_translation and specificity_signal_count == 0 and (desc_words >= 80 or req_words >= 30):
        role_specificity_score += 2

    role_specificity_score = _clip(role_specificity_score, 0, 25)

    # Remote clarity
    remote_hint = bool(
        re.search(r"\bremote\b|\btelecommut\w*\b|\bwork\s+from\s+home\b", text)
        or int(posting.get("telecommuting", 0) or 0) == 1
    )
    if remote_hint:
        remote_clarity_score += 10
    if timezone_hits:
        remote_clarity_score += 6
    if remote_policy_hits:
        remote_clarity_score += 3
    if location and remote_hint:
        remote_clarity_score += 4
    remote_clarity_score = _clip(remote_clarity_score, 0, 20)

    # Apply experience
    if apply_url:
        apply_experience_score += 5
        if _normalize_url(apply_url).startswith("https://"):
            apply_experience_score += 3
        else:
            apply_experience_score -= 2
        if is_known_ats:
            apply_experience_score += 5
        elif is_known_board:
            apply_experience_score += 3
        elif is_major_job_board:
            apply_experience_score += 2
        if is_shortener:
            apply_experience_score -= 4
    if hiring_process_hits:
        apply_experience_score += 2
        positive_evidence.append("Hiring process details are described")
    apply_experience_score = _clip(apply_experience_score, 0, 15)

    # Freshness
    age_days = _days_old(posting)
    if age_days is None:
        freshness_score += 3
    elif age_days <= 7:
        freshness_score += 10
    elif age_days <= 14:
        freshness_score += 8
    elif age_days <= 30:
        freshness_score += 6
    elif age_days <= 60:
        freshness_score += 3
    else:
        freshness_score += 1
        risk_flags.append("Posting appears stale based on publish date")
    freshness_score = _clip(freshness_score, 0, 10)

    quality_score = (
        transparency_score
        + role_specificity_score
        + remote_clarity_score
        + apply_experience_score
        + freshness_score
    )
    quality_score = _clip(quality_score, 0, 100)

    final_raw = (
        FINAL_BLEND_WEIGHTS["trust_score"] * trust_score
        + FINAL_BLEND_WEIGHTS["quality_score"] * quality_score
    )
    final_opportunity_score = final_raw
    if trust_score < 50:
        final_opportunity_score = min(final_opportunity_score, 49.0)

    if final_opportunity_score >= 85 and trust_score >= 75:
        badge = "Trusted Pick"
    elif final_opportunity_score >= 70:
        badge = "Promising"
    elif final_opportunity_score >= 50:
        badge = "Verify First"
    else:
        badge = "Avoid"

    if trust_score >= 75:
        fraud_risk_level = "low"
    elif trust_score >= 50:
        fraud_risk_level = "medium"
    else:
        fraud_risk_level = "high"

    if quality_score >= 75:
        quality_level = "high"
    elif quality_score >= 50:
        quality_level = "medium"
    else:
        quality_level = "low"

    return {
        "trust_score": round(trust_score, 2),
        "quality_score": round(quality_score, 2),
        "final_opportunity_score": round(final_opportunity_score, 2),
        "badge": badge,
        "fraud_risk_level": fraud_risk_level,
        "quality_level": quality_level,
        "rule_engine_score": round(rule_engine_score, 2),
        "ml_legitimacy_score": round(ml_legitimacy_score, 2),
        "trust_pre_cap_score": round(trust_raw, 2),
        "trust_cap": round(trust_cap, 2),
        "trust_cap_applied": bool(trust_score < trust_raw),
        "final_pre_cap_score": round(final_raw, 2),
        "final_cap_applied": bool(final_opportunity_score < final_raw),
        "weights": {
            "trust_blend": TRUST_BLEND_WEIGHTS,
            "final_blend": FINAL_BLEND_WEIGHTS,
            "trust_component_maxima": TRUST_COMPONENT_MAXIMA,
            "quality_component_maxima": QUALITY_COMPONENT_MAXIMA,
        },
        "hard_caps": hard_caps,
        "components": {
            "trust": {
                "identity_apply_integrity": round(identity_score, 2),
                "communication_safety": round(communication_score, 2),
                "monetary_safety": round(monetary_score, 2),
                "company_evidence": round(company_score, 2),
            },
            "quality": {
                "transparency": round(transparency_score, 2),
                "role_specificity": round(role_specificity_score, 2),
                "remote_clarity": round(remote_clarity_score, 2),
                "apply_experience": round(apply_experience_score, 2),
                "freshness": round(freshness_score, 2),
            },
        },
        "positive_evidence": sorted(set(positive_evidence)),
        "risk_flags": sorted(set(risk_flags)),
        "matched_signals": {
            "messaging_app": _format_pattern_names(messaging_hits),
            "text_only_interview": _format_pattern_names(text_interview_hits),
            "interview_over_messaging": _format_pattern_names(interview_over_messaging_hits),
            "pay_to_get_paid": _format_pattern_names(pay_hits),
            "fake_check": _format_pattern_names(fake_check_hits),
            "task_scam": _format_pattern_names(task_hits),
            "bank_info_request": _format_pattern_names(bank_hits),
            "crypto_or_gift_card": _format_pattern_names(crypto_gift_hits),
            "money_mule": _format_pattern_names(mule_hits),
            "recruit_others": _format_pattern_names(recruit_hits),
            "generic_job_text": _format_pattern_names(generic_hits),
            "urgency_or_pressure": _format_pattern_names(urgency_hits),
            "no_interview": _format_pattern_names(no_interview_hits),
            "equipment_check_pattern": _format_pattern_names(equipment_hits),
            "timezone_or_region": _format_pattern_names(timezone_hits),
            "remote_policy_terms": _format_pattern_names(remote_policy_hits),
            "role_specificity_terms": _format_pattern_names(specificity_hits),
            "years_experience_terms": _format_pattern_names(experience_hits),
            "credential_terms": _format_pattern_names(credential_hits),
            "responsibility_terms": _format_pattern_names(responsibility_hits),
            "schedule_terms": _format_pattern_names(schedule_hits),
            "qualification_section_terms": _format_pattern_names(qualification_section_hits),
            "hiring_process_terms": _format_pattern_names(hiring_process_hits),
            "compensation_clarity_terms": _format_pattern_names(compensation_clarity_hits),
            "company_context_terms": _format_pattern_names(company_context_hits),
            "benefit_categories": sorted(benefit_categories),
        },
        "contact_email_domains": email_domains,
        "free_email_domains": free_email_domains,
        "apply_domain": apply_domain,
        "is_known_ats_domain": bool(is_known_ats),
        "ats_provider": _infer_ats_provider(apply_domain),
        "is_major_job_board_domain": bool(is_major_job_board),
        "job_board_provider": _infer_job_board_provider(apply_domain),
        "posting_age_days": age_days,
        "detected_language": detected_language or "unknown",
        "language_confidence": round(language_confidence, 4),
        "translation_applied": bool(translation_applied),
        "language_normalization_softened": bool(non_english_without_translation),
        "benefit_category_count": benefit_category_count,
    }
