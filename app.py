"""Gradio dashboard for exploring live remote jobs with fraud/quality scoring."""

from __future__ import annotations

import ast
import base64
import html
import math
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_fraud_detector.inference import FraudDetector
from job_fraud_detector.live_sources import SOURCES, fetch_jobs_from_sources, score_live_jobs
from job_fraud_detector.rules import FINAL_BLEND_WEIGHTS, TRUST_BLEND_WEIGHTS

DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "emscad_light_model"
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "voting_soft")
DEFAULT_PAGE_SIZE = 6
PROGRESSIVE_RENDER_THRESHOLD = int(os.environ.get("PROGRESSIVE_RENDER_THRESHOLD", "120"))
PROGRESSIVE_SCORE_CHUNK_SIZE = int(os.environ.get("PROGRESSIVE_SCORE_CHUNK_SIZE", "36"))
BATCH_PREDICTION_SIZE = int(os.environ.get("BATCH_PREDICTION_SIZE", "128"))
RECOMMENDATION_BATCH_SIZE = int(os.environ.get("RECOMMENDATION_BATCH_SIZE", "64"))
COMPANY_MOTTO = (
    "We surface remote jobs from trusted public sources, score them for legitimacy and quality, "
    "and show users exactly why a job is safe, incomplete, or risky."
)

LOGO_CANDIDATE_PATHS = [
    ROOT / "assets" / "remote_trust_logo.png",
    Path("/Users/shahan/Downloads/ChatGPT Image May 24, 2026, 03_52_50 PM.png"),
]

MIND_MESH_LOGO_CANDIDATE_PATHS = [
    ROOT / "assets" / "mind_mesh_logo.png",
    Path("/Users/shahan/Downloads/ChatGPT Image May 24, 2026, 09_31_27 PM.png"),
]

DEV_PROFILE_CANDIDATE_PATHS: dict[str, list[Path]] = {
    "md_mohidul_hasan": [
        ROOT / "assets" / "md_mohidul_hasan.jpg",
        Path("/Users/shahan/Downloads/My Pics/25352170_1886572034716447_5262364254175945459_o.jpg"),
    ],
    "rashed_azad_chowdhury": [
        ROOT / "assets" / "rashed_azad_chowdhury.jpg",
    ],
}

ABOUT_SKILLS = [
    "Agentic AI and LLM-based workflows",
    "Retrieval-Augmented Generation (RAG) and Knowledge Graphs",
    "NLP, information retrieval, and model evaluation",
    "Public Health AI and policy intelligence",
    "MLOps, reproducible experimentation, and AI system design",
    "Python, PyTorch, TensorFlow, Hugging Face, LangChain, LangGraph, vLLM, and ChromaDB",
    "Software engineering across REST APIs, React/React Native, Java, SQL, and Laravel",
    "Teaching, mentorship, technical communication, and applied AI research",
]

MD_LINKEDIN_URL = os.environ.get(
    "MD_LINKEDIN_URL",
    "https://www.linkedin.com/in/shahan-hasan-141337142/",
).strip()
MD_GITHUB_URL = os.environ.get(
    "MD_GITHUB_URL",
    "https://github.com/shahanHasan",
).strip()
MD_STACKOVERFLOW_URL = os.environ.get(
    "MD_STACKOVERFLOW_URL",
    "https://stackoverflow.com/users/12205044/shahan-hasan",
).strip()
MD_PERSONAL_SITE_URL = os.environ.get(
    "MD_PERSONAL_SITE_URL",
    "https://shahanhasan.github.io/personal-site/",
).strip()
MD_EMAIL_PRIMARY = os.environ.get(
    "MD_EMAIL_PRIMARY",
    "shahan.hasan101294@gmail.com",
).strip()
MD_EMAIL_ACADEMIC = os.environ.get(
    "MD_EMAIL_ACADEMIC",
    "mhasan24@laurentian.ca",
).strip()

SOURCE_CITATIONS: dict[str, tuple[str, str]] = {
    "we_work_remotely_rss": ("We Work Remotely RSS", SOURCES.get("we_work_remotely_rss", "")),
    "jobicy_api": ("Jobicy API", SOURCES.get("jobicy_api", "")),
    "remotive_api": ("Remotive API", "https://remotive.com/api/remote-jobs"),
    "remotive_api_docs": ("Remotive API", "https://remotive.com/api/remote-jobs"),
    "usajobs_api": ("USAJOBS API", SOURCES.get("usajobs_api", "")),
    "synthetic_crude_scam": ("Synthetic Crude Scam Set", ""),
    "synthetic_sophisticated_scam": ("Synthetic Sophisticated Scam Set", ""),
}

SOURCE_OPTIONS = [
    ("We Work Remotely", "we_work_remotely_rss"),
    ("Jobicy", "jobicy_api"),
    ("Remotive", "remotive_api"),
    ("USAJOBS (Remote)", "usajobs_api"),
]

NEGATIVE_SOURCE_OPTIONS = [
    ("Crude Scam Samples", "synthetic_crude_scam"),
    ("Sophisticated Scam Samples", "synthetic_sophisticated_scam"),
]

RECOMMENDATION_SOURCE_OPTIONS = SOURCE_OPTIONS

BADGE_COLORS = {
    "Trusted Pick": "#0f9d58",
    "Promising": "#1a73e8",
    "Verify First": "#f29900",
    "Avoid": "#d93025",
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

TRUST_COMPONENT_LABELS = {
    "identity_apply_integrity": "apply link integrity",
    "communication_safety": "communication safety",
    "monetary_safety": "money safety",
    "company_evidence": "company evidence",
}

QUALITY_COMPONENT_LABELS = {
    "transparency": "transparency",
    "role_specificity": "role detail",
    "remote_clarity": "remote clarity",
    "apply_experience": "apply experience",
    "freshness": "freshness",
}

BADGE_GUIDANCE = {
    "Trusted Pick": "Strong option to pursue now.",
    "Promising": "Looks good overall, but do a quick verification pass.",
    "Verify First": "Mixed signals. Verify before applying.",
    "Avoid": "High risk profile. Do not apply without independent verification.",
}

SORT_OPTIONS = [
    "Final Score (high to low)",
    "Trust Score (high to low)",
    "Quality Score (high to low)",
    "Fraud Probability (high to low)",
    "Newest First",
]

RISK_FILTER_OPTIONS = [
    "All",
    "Hard Cap Triggered",
    "High Risk Only",
]

URL_PATTERN = re.compile(r"https?://[^\s<>'\"\\)]+", flags=re.IGNORECASE)
APPLY_HINT_TOKENS = (
    "apply",
    "application",
    "career",
    "careers",
    "jobs",
    "job",
    "greenhouse",
    "lever",
    "ashby",
    "ashbyhq",
    "myworkdayjobs",
    "workdayjobs",
    "smartrecruiters",
    "jobvite",
    "icims",
    "recruiting",
)
URL_SHORTENER_HOSTS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "rb.gy",
    "rebrand.ly",
}


def _load_detector() -> tuple[FraudDetector | None, str]:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    if not artifacts_dir.exists():
        return None, f"Artifacts directory not found: `{artifacts_dir}`"
    try:
        detector = FraudDetector.from_artifacts(artifacts_dir, model_name=DEFAULT_MODEL_NAME)
        return detector, (
            f"Loaded model `{detector.model_name or 'selected_champion'}` "
            f"with threshold `{detector.threshold:.4f}` from `{artifacts_dir}`."
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return None, f"Failed to load detector from `{artifacts_dir}`: {exc}"


DETECTOR, DETECTOR_STATUS = _load_detector()


def _load_image_data_uri(candidate_paths: list[Path]) -> str:
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    for path in candidate_paths:
        try:
            if not path.exists() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            mime_type = mime_by_suffix.get(suffix, "image/png")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        except Exception:
            continue
    return ""


LOGO_DATA_URI = _load_image_data_uri(LOGO_CANDIDATE_PATHS)
MIND_MESH_LOGO_DATA_URI = _load_image_data_uri(MIND_MESH_LOGO_CANDIDATE_PATHS)
MD_PROFILE_DATA_URI = _load_image_data_uri(DEV_PROFILE_CANDIDATE_PATHS["md_mohidul_hasan"])
RASHED_PROFILE_DATA_URI = _load_image_data_uri(DEV_PROFILE_CANDIDATE_PATHS["rashed_azad_chowdhury"])


def _build_intro_html(detector_status: str) -> str:
    clean_status = detector_status.replace("`", "")
    logo_html = ""
    if LOGO_DATA_URI:
        logo_html = (
            "<div class='hero-logo-wrap'>"
            f"<img class='hero-logo' src='{LOGO_DATA_URI}' alt='RemoteTrust logo' />"
            "</div>"
        )

    return (
        "<section class='hero-shell'>"
        f"{logo_html}"
        "<div class='hero-copy'>"
        "<p class='hero-kicker'>RemoteTrust</p>"
        "<h1 class='hero-title'>RemoteTrust Job Board Dashboard</h1>"
        f"<p class='hero-motto'>{html.escape(COMPANY_MOTTO)}</p>"
        "<p class='hero-subnote'>"
        "Discover remote jobs from <strong>We Work Remotely, Jobicy, Remotive, and USAJOBS</strong>. "
        "Every listing includes a <strong>Trust score</strong>, <strong>Quality score</strong>, and "
        "<strong>Final recommendation</strong> with plain-language reasons."
        "</p>"
        "<p class='hero-subnote'>"
        "Use filters to narrow jobs, then open each card's scoring breakdown when you want technical detail. "
        "Each card also has a <strong>Trust + Quality Rule Debug</strong> panel showing which rules fired "
        "and which did not."
        "</p>"
        f"<p class='hero-status'><strong>Model status:</strong> {html.escape(clean_status)}</p>"
        "</div>"
        "</section>"
    )


def _build_landing_html(detector_status: str) -> str:
    threshold_text = f"{DETECTOR.threshold:.4f}" if DETECTOR is not None else "validation-tuned threshold"
    logo_block = ""
    if LOGO_DATA_URI:
        logo_block = (
            "<div class='landing-logo-wrap'>"
            f"<img class='landing-logo' src='{LOGO_DATA_URI}' alt='RemoteTrust logo' />"
            "</div>"
        )

    source_kind_map = {
        "we_work_remotely_rss": "RSS",
        "jobicy_api": "API",
        "remotive_api": "API",
        "usajobs_api": "Gov API",
    }
    source_cards_html = "".join(
        (
            "<div class='landing-source-item'>"
            f"<span class='source-name'>{html.escape(label)}</span>"
            f"<span class='source-kind'>{html.escape(source_kind_map.get(source_key, 'Feed'))}</span>"
            "</div>"
        )
        for label, source_key in SOURCE_OPTIONS
    )
    mind_mesh_branding = (
        "<div class='landing-credit'>"
        f"<img class='landing-credit-logo' src='{MIND_MESH_LOGO_DATA_URI}' alt='Mind Mesh logo' />"
        "<p><strong>Made by Mind Mesh Team</strong> · Scale Without Borders Hackathon Submission 2026</p>"
        "</div>"
        if MIND_MESH_LOGO_DATA_URI
        else "<div class='landing-credit'><p><strong>Made by Mind Mesh Team</strong></p></div>"
    )

    return (
        "<section class='landing-shell'>"
        "<div class='landing-content'>"
        "<div class='landing-top-grid'>"
        "<div class='landing-copy'>"
        "<p class='landing-kicker'>RemoteTrust</p>"
        "<h1 class='landing-title'>Find Remote Jobs You Can Actually Trust</h1>"
        f"<p class='landing-motto'>{html.escape(COMPANY_MOTTO)}</p>"
        "<div class='landing-pill-row'>"
        "<span class='landing-pill'>Trust-First</span>"
        "<span class='landing-pill'>Explainable AI</span>"
        "<span class='landing-pill'>Live Source Monitoring</span>"
        "</div>"
        "<div class='landing-weight-grid'>"
        "<div class='weight-item'>"
        "<div class='weight-label'><span>Final Score Weight: Trust</span><strong>70%</strong></div>"
        "<div class='weight-track'><span class='weight-fill trust-fill'></span></div>"
        "</div>"
        "<div class='weight-item'>"
        "<div class='weight-label'><span>Final Score Weight: Quality</span><strong>30%</strong></div>"
        "<div class='weight-track'><span class='weight-fill quality-fill'></span></div>"
        "</div>"
        "<div class='weight-item'>"
        "<div class='weight-label'><span>Trust Blend: Rule Engine</span><strong>45%</strong></div>"
        "<div class='weight-track'><span class='weight-fill rules-fill'></span></div>"
        "</div>"
        "<div class='weight-item'>"
        "<div class='weight-label'><span>Trust Blend: ML Legitimacy</span><strong>55%</strong></div>"
        "<div class='weight-track'><span class='weight-fill ml-fill'></span></div>"
        "</div>"
        "</div>"
        "</div>"
        "<div class='landing-visual-panel'>"
        "<div class='signal-orbit'>"
        "<div class='signal-core'>"
        f"{logo_block}"
        "<p class='signal-core-label'>Opportunity Signal</p>"
        "<p class='signal-core-text'>Rules + Voting Ensemble</p>"
        "</div>"
        "</div>"
        "<div class='visual-kpi-grid'>"
        "<div class='visual-kpi'><span>Live Sources</span><strong>4</strong></div>"
        "<div class='visual-kpi'><span>Model Type</span><strong>Soft Voting</strong></div>"
        "<div class='visual-kpi'><span>Saved Threshold</span><strong>"
        f"{html.escape(threshold_text)}"
        "</strong></div>"
        "</div>"
        "</div>"
        "</div>"
        "<div class='landing-method-grid'>"
        "<div class='landing-point'><h3>Hybrid Innovation</h3><p>Traditional ML ensemble + expert trust/quality heuristics for practical job-seeker safety.</p></div>"
        "<div class='landing-point'><h3>Trust + Quality Scoring</h3><p>Dual scoring for legitimacy and posting quality, then a final recommendation.</p></div>"
        "<div class='landing-point'><h3>Rules-First Safety Caps</h3><p>High-severity scam patterns cap trust so polished scam copy cannot rank highly.</p></div>"
        "<div class='landing-point'><h3>Explainable Decisions</h3><p>Rule-level fired/not-fired evidence and plain-language reasons for each score.</p></div>"
        "</div>"
        "<div class='landing-formula-grid'>"
        "<div class='landing-formula'><h4>Trust Blend</h4><p><code>Trust = 0.45 x RuleEngine + 0.55 x ML Legitimacy</code></p></div>"
        "<div class='landing-formula'><h4>Final Opportunity</h4><p><code>Final = 0.70 x Trust + 0.30 x Quality</code></p></div>"
        "<div class='landing-formula'><h4>Guardrail</h4><p><code>If Trust &lt; 50, Final score is capped below 50</code></p></div>"
        "</div>"
        "<details class='landing-tech-details'>"
        "<summary>Technical replication details</summary>"
        "<div class='landing-repro'>"
        "<p class='landing-repro-lead'>"
        "We use a <strong>soft-voting ensemble classifier</strong> (LogReg + SGD + Calibrated LinearSVC + ComplementNB) "
        "with TF-IDF / BoW features, recall-first validation tuning, and saved threshold "
        f"<code>{html.escape(threshold_text)}</code>."
        "</p>"
        "<ul class='landing-repro-list'>"
        "<li>Dataset baseline: EMSCAD fake job postings.</li>"
        "<li>Text features: n-grams over title, company, description, requirements, and metadata.</li>"
        "<li>Inference blend: ML legitimacy + rules-based trust + quality scoring + hard-cap logic.</li>"
        "<li>Live feeds: We Work Remotely, Jobicy, Remotive, and USAJOBS (remote-filtered).</li>"
        "</ul>"
        "</div>"
        "</details>"
        "<div class='landing-sources-card'>"
        "<div class='landing-sources-head'>"
        "<h4>Live Sources (Remote)</h4>"
        "<p><strong>4 feeds</strong> with attribution-first linking and periodic refresh.</p>"
        "</div>"
        f"<div class='landing-source-grid'>{source_cards_html}</div>"
        "</div>"
        f"{mind_mesh_branding}"
        "</div>"
        "</section>"
    )


def _profile_media_html(data_uri: str, alt_text: str, initials: str) -> str:
    if data_uri:
        return (
            "<div class='about-photo-wrap'>"
            f"<img class='about-photo' src='{data_uri}' alt='{html.escape(alt_text)}' />"
            "</div>"
        )
    return f"<div class='about-photo-wrap about-photo-fallback'>{html.escape(initials)}</div>"


def _build_about_html() -> str:
    focus_tags = [
        "Trust + Quality Dual Scoring",
        "Rules-First Scam Guardrails",
        "Voting-Classifier Legitimacy Model",
        "LIME-Based Explainability",
        "Live Remote Job Feed Aggregation",
    ]
    focus_tags_html = "".join(f"<span class='about-tag'>{html.escape(tag)}</span>" for tag in focus_tags)

    md_contact_items: list[str] = []
    if MD_EMAIL_PRIMARY:
        md_contact_items.append(
            "<li><i class='fa-solid fa-envelope'></i>"
            f"<a href='mailto:{html.escape(MD_EMAIL_PRIMARY)}'>{html.escape(MD_EMAIL_PRIMARY)}</a></li>"
        )
    if MD_EMAIL_ACADEMIC:
        md_contact_items.append(
            "<li><i class='fa-solid fa-graduation-cap'></i>"
            f"<a href='mailto:{html.escape(MD_EMAIL_ACADEMIC)}'>{html.escape(MD_EMAIL_ACADEMIC)}</a></li>"
        )
    if MD_GITHUB_URL:
        md_contact_items.append(
            "<li><i class='fa-brands fa-github'></i>"
            f"<a href='{html.escape(MD_GITHUB_URL)}' target='_blank' rel='noopener noreferrer'>GitHub</a></li>"
        )
    if MD_LINKEDIN_URL:
        md_contact_items.append(
            "<li><i class='fa-brands fa-linkedin'></i>"
            f"<a href='{html.escape(MD_LINKEDIN_URL)}' target='_blank' rel='noopener noreferrer'>LinkedIn</a></li>"
        )
    if MD_STACKOVERFLOW_URL:
        md_contact_items.append(
            "<li><i class='fa-brands fa-stack-overflow'></i>"
            f"<a href='{html.escape(MD_STACKOVERFLOW_URL)}' target='_blank' rel='noopener noreferrer'>Stack Overflow</a></li>"
        )
    if MD_PERSONAL_SITE_URL:
        md_contact_items.append(
            "<li><i class='fa-solid fa-globe'></i>"
            f"<a href='{html.escape(MD_PERSONAL_SITE_URL)}' target='_blank' rel='noopener noreferrer'>Personal Site</a></li>"
        )
    md_contacts_html = "<ul class='about-link-list'>" + "".join(md_contact_items) + "</ul>" if md_contact_items else ""

    md_media = _profile_media_html(
        data_uri=MD_PROFILE_DATA_URI,
        alt_text="Md Mohidul Hasan",
        initials="MH",
    )
    rashed_media = _profile_media_html(
        data_uri=RASHED_PROFILE_DATA_URI,
        alt_text="Rashed Azad Chowdhury",
        initials="RA",
    )
    mind_mesh_about_brand = (
        "<div class='about-mindmesh'>"
        f"<img class='about-mindmesh-logo' src='{MIND_MESH_LOGO_DATA_URI}' alt='Mind Mesh logo' />"
        "<div class='about-mindmesh-copy'>"
        "<p class='about-mindmesh-kicker'>Built by</p>"
        "<p class='about-mindmesh-title'>Mind Mesh Team</p>"
        "</div>"
        "</div>"
        if MIND_MESH_LOGO_DATA_URI
        else ""
    )

    return (
        "<section class='about-shell'>"
        "<div class='about-banner'>"
        "<p class='about-kicker'>Scale Without Borders Hackathon 2026 Submission</p>"
        "<h2>Team MindMesh</h2>"
        "<p class='about-intro'>"
        "RemoteTrust is our trust-first remote job intelligence app: we score legitimacy and quality, then explain every decision in plain language."
        "</p>"
        f"{mind_mesh_about_brand}"
        "<div class='about-tag-row'>"
        f"{focus_tags_html}"
        "</div>"
        "</div>"
        "<div class='about-grid'>"
        "<article class='about-card'>"
        f"{rashed_media}"
        "<div class='about-card-copy'>"
        "<h3>Rashed Azad Chowdhury</h3>"
        "<p class='about-role'>IT Systems Administration & PMP-Certified Project Management Professional</p>"
        "<p class='about-pronouns'>He/Him</p>"
        "<p class='about-location'><i class='fa-solid fa-location-dot'></i> Greater Sudbury, Ontario, Canada</p>"
        "<ul class='about-quick-list'>"
        "<li><i class='fa-solid fa-briefcase'></i><span>12+ years across regulated financial and institutional IT.</span></li>"
        "<li><i class='fa-solid fa-diagram-project'></i><span>Delivered core banking, infrastructure, and cloud/data migration initiatives.</span></li>"
        "<li><i class='fa-solid fa-shield-halved'></i><span>Microsoft 365, Azure AD, CRM, MSSQL, ISO 27001 / ISMS 27001.</span></li>"
        "<li><i class='fa-solid fa-circle-check'></i><span>Eligible to work in Canada and available immediately.</span></li>"
        "</ul>"
        "</div>"
        "</article>"
        "<article class='about-card'>"
        f"{md_media}"
        "<div class='about-card-copy'>"
        "<h3>Md Mohidul (Shahan) Hasan</h3>"
        "<p class='about-role'>AI/ML Engineer | AI4PH Intern @ Fraser Health Authority</p>"
        "<p class='about-pronouns'>He/Him</p>"
        "<p class='about-location'><i class='fa-solid fa-location-dot'></i> Greater Sudbury, Ontario, Canada</p>"
        "<ul class='about-quick-list'>"
        "<li><i class='fa-solid fa-graduation-cap'></i><span>MSc Computational Science candidate at Laurentian University.</span></li>"
        "<li><i class='fa-solid fa-brain'></i><span>Focus: Agentic AI, LLM evaluation, hybrid retrieval, and Graph-RAG.</span></li>"
        "<li><i class='fa-solid fa-heart-pulse'></i><span>Applied AI-for-public-health work in responsible deployment settings.</span></li>"
        "<li><i class='fa-solid fa-code'></i><span>Stack: Python, PyTorch, TensorFlow, Hugging Face, LangChain, LangGraph, vLLM, ChromaDB.</span></li>"
        "</ul>"
        "<h4>Contact & Profiles</h4>"
        f"{md_contacts_html}"
        "</div>"
        "</article>"
        "</div>"
        "<p class='about-opportunities'>"
        "<strong>Open to collaboration:</strong> AI/ML Engineering, Applied AI Research, NLP, Public Health AI, MLOps, and trustworthy AI products."
        "</p>"
        "</section>"
    )


def _score_color(score: float) -> str:
    if score >= 85:
        return "#0f9d58"
    if score >= 70:
        return "#1a73e8"
    if score >= 50:
        return "#f29900"
    return "#d93025"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _dedupe_text(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _friendly_value(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _source_meta(source_key: str) -> tuple[str, str]:
    return SOURCE_CITATIONS.get(source_key, (source_key, ""))


def _source_map_from_selection(selected_sources: list[str] | None) -> dict[str, str]:
    if selected_sources:
        selected = set(str(value).strip() for value in selected_sources if str(value).strip())
    else:
        selected = {value for _, value in SOURCE_OPTIONS}

    mapped: dict[str, str] = {}
    for key in selected:
        if key == "remotive_api":
            if "remotive_api" in SOURCES:
                mapped["remotive_api"] = SOURCES["remotive_api"]
            elif "remotive_api_docs" in SOURCES:
                mapped["remotive_api_docs"] = SOURCES["remotive_api_docs"]
            continue
        if key in SOURCES:
            mapped[key] = SOURCES[key]

    if mapped:
        return mapped
    return dict(SOURCES)


def _build_negative_samples() -> list[dict[str, Any]]:
    return [
        {
            "source": "synthetic_crude_scam",
            "job_url": "http://quick-income-now.example/jobs/phone-data-entry",
            "apply_url": "http://bit.ly/start-earning-now",
            "posted_date": "2026-05-22T11:10:00+00:00",
            "title": "URGENT Data Entry - Earn $500 Daily From Your Phone",
            "location": "Remote - Worldwide",
            "company_profile": "FastStart Recruiting Group",
            "description": (
                "No interview. Immediate start today. Work from your phone and get guaranteed income. "
                "Reply YES on WhatsApp to reserve one of 20 spots. Daily pay."
            ),
            "requirements": (
                "No experience needed. Pay a $49 activation fee and onboarding fee to unlock your account."
            ),
            "benefits": "Fast payouts and bonuses.",
            "employment_type": "Contract",
            "required_experience": "Entry",
            "required_education": "",
            "industry": "Marketing",
            "function": "Data Entry",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
        },
        {
            "source": "synthetic_crude_scam",
            "job_url": "https://hiring-center-checks.example/job/assistant",
            "apply_url": "https://tinyurl.com/home-office-check",
            "posted_date": "2026-05-20T09:30:00+00:00",
            "title": "Remote Administrative Assistant - Equipment Reimbursement",
            "location": "US, CA",
            "company_profile": "NorthField Office Solutions",
            "description": (
                "We will mail you a check for your home office kit. Mobile deposit it right away and send part "
                "to our approved vendor to secure your workstation."
            ),
            "requirements": (
                "Must have bank account details ready. No interview, instant hire once deposit is confirmed."
            ),
            "benefits": "Weekly pay.",
            "employment_type": "Full-time",
            "required_experience": "1 year",
            "required_education": "High school",
            "industry": "Administrative Services",
            "function": "Operations",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
        },
        {
            "source": "synthetic_crude_scam",
            "job_url": "https://boost-work-platform.example/tasks",
            "apply_url": "https://boost-work-platform.example/tasks",
            "posted_date": "2026-05-18T16:20:00+00:00",
            "title": "Product Boosting Specialist - Online Tasks",
            "location": "Remote",
            "company_profile": "BrightScale Talent Hub",
            "description": (
                "Complete online tasks to boost products and ratings. Earn commission per click and daily pay. "
                "Task optimization team available on Telegram."
            ),
            "requirements": (
                "Start with a security deposit to unlock high-value tasks. Crypto payouts available in USDT."
            ),
            "benefits": "Performance bonus.",
            "employment_type": "Part-time",
            "required_experience": "",
            "required_education": "",
            "industry": "E-commerce",
            "function": "Operations",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
        },
        {
            "source": "synthetic_crude_scam",
            "job_url": "https://mlm-hiring-network.example/remote-sales",
            "apply_url": "https://rb.gy/join-our-team",
            "posted_date": "2026-05-17T14:05:00+00:00",
            "title": "Remote Sales Partner - Unlimited Earnings",
            "location": "Worldwide",
            "company_profile": "Legacy Growth Alliance",
            "description": (
                "Guaranteed income and immediate hiring. Build your downline and recruit others for high residuals."
            ),
            "requirements": (
                "Registration fee required. Interview over chat only on Signal. Limited slots available."
            ),
            "benefits": "Residual commissions.",
            "employment_type": "Commission",
            "required_experience": "No experience needed",
            "required_education": "",
            "industry": "Sales",
            "function": "Business Development",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
        },
        {
            "source": "synthetic_sophisticated_scam",
            "job_url": "https://careers-novalytics.example/roles/senior-ops-analyst",
            "apply_url": "https://careers-novalytics.example/roles/senior-ops-analyst",
            "posted_date": "2026-05-23T12:45:00+00:00",
            "title": "Senior Remote Operations Analyst",
            "location": "Remote - North America",
            "company_profile": "NovaLytics is a distributed operations intelligence company helping enterprise teams modernize workflows.",
            "description": (
                "You will own performance reporting, cross-functional project coordination, and process optimization. "
                "Compensation range is $92,000-$118,000 plus health coverage and PTO. "
                "Final-stage interview process is conducted via Telegram text interview for speed."
            ),
            "requirements": (
                "5+ years operations analytics experience, SQL proficiency, dashboard tooling, and stakeholder communication."
            ),
            "benefits": "Medical, dental, PTO, learning budget.",
            "employment_type": "Full-time",
            "required_experience": "5 years",
            "required_education": "Bachelor's degree",
            "industry": "Technology",
            "function": "Operations",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        },
        {
            "source": "synthetic_sophisticated_scam",
            "job_url": "https://talent-bridge-consulting.example/careers/finance-coordinator",
            "apply_url": "https://talent-bridge-consulting.example/careers/finance-coordinator",
            "posted_date": "2026-05-21T18:00:00+00:00",
            "title": "Remote Finance Coordinator",
            "location": "Remote - US",
            "company_profile": "Talent Bridge Consulting supports accounting transformation programs for global clients.",
            "description": (
                "This role manages reconciliation workflows, monthly close tasks, and vendor communication in a remote environment. "
                "Compensation is $82,000-$96,000 with retirement and wellness benefits."
            ),
            "requirements": (
                "4+ years finance operations, strong spreadsheet skills, and comfort with distributed teams. "
                "During onboarding, a refundable compliance deposit is required before system provisioning."
            ),
            "benefits": "Health coverage, PTO, retirement plan, wellness stipend.",
            "employment_type": "Full-time",
            "required_experience": "4 years",
            "required_education": "Bachelor's degree",
            "industry": "Finance",
            "function": "Accounting",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        },
        {
            "source": "synthetic_sophisticated_scam",
            "job_url": "https://altacore-digital.example/openings/customer-success-manager",
            "apply_url": "https://altacore-digital.example/openings/customer-success-manager",
            "posted_date": "2026-05-24T07:15:00+00:00",
            "title": "Customer Success Manager (Remote)",
            "location": "Remote - US and Canada",
            "company_profile": "AltaCore Digital provides retention operations services to SaaS companies.",
            "description": (
                "Lead success strategy, KPI reviews, and quarterly plans with enterprise customers. "
                "Base salary $95,000-$120,000, bonus, and comprehensive benefits."
            ),
            "requirements": (
                "6+ years customer success leadership, CRM fluency, and strong communication. "
                "Interview process includes WhatsApp follow-up for final verification and document review."
            ),
            "benefits": "Medical, dental, vision, PTO, parental leave, development budget.",
            "employment_type": "Full-time",
            "required_experience": "6 years",
            "required_education": "Bachelor's degree",
            "industry": "SaaS",
            "function": "Customer Success",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        },
        {
            "source": "synthetic_sophisticated_scam",
            "job_url": "https://northlane-ai.example/jobs/program-manager",
            "apply_url": "https://northlane-ai.example/jobs/program-manager",
            "posted_date": "2026-05-24T13:40:00+00:00",
            "title": "AI Program Manager (Remote)",
            "location": "Remote - EU/NA overlap",
            "company_profile": "Northlane AI builds workflow automation systems for regulated organizations.",
            "description": (
                "Drive multi-team program execution, roadmap alignment, and executive reporting. "
                "Compensation range $120,000-$148,000 with equity and comprehensive benefits."
            ),
            "requirements": (
                "7+ years in program management, cross-functional leadership, risk management, and technical fluency. "
                "Home-office setup is reimbursed via mailed check that must be deposited and forwarded to our equipment vendor."
            ),
            "benefits": "Health, dental, PTO, equity, retirement, learning budget.",
            "employment_type": "Full-time",
            "required_experience": "7 years",
            "required_education": "Bachelor's degree",
            "industry": "AI Software",
            "function": "Program Management",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
        },
    ]


def _recommendation_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("title", ""),
        row.get("company_profile", ""),
        row.get("location", ""),
        row.get("description", ""),
        row.get("requirements", ""),
        row.get("benefits", ""),
        row.get("employment_type", ""),
        row.get("required_experience", ""),
        row.get("required_education", ""),
        row.get("industry", ""),
        row.get("function", ""),
    ]
    text = " ".join(_friendly_value(value) for value in fields if _friendly_value(value))
    return re.sub(r"\s+", " ", text).strip()


def _rank_cv_recommendations(
    cv_text: str,
    rows: list[dict[str, Any]],
    similarity_threshold: float,
    min_trust_score: float,
    max_results: int,
) -> list[dict[str, Any]]:
    clean_cv = re.sub(r"\s+", " ", (cv_text or "").strip())
    if not clean_cv or not rows:
        return []

    candidate_rows = [row for row in rows if _safe_float(row.get("trust_score")) >= min_trust_score]
    if not candidate_rows:
        return []

    job_texts = [_recommendation_text(row) for row in candidate_rows]
    if not any(job_texts):
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=40000,
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform([clean_cv, *job_texts])
    cv_vec = matrix[0:1]
    job_vecs = matrix[1:]
    similarities = cosine_similarity(cv_vec, job_vecs).ravel()

    ranked: list[dict[str, Any]] = []
    for row, similarity in zip(candidate_rows, similarities, strict=False):
        sim_value = float(similarity)
        if sim_value < similarity_threshold:
            continue
        enriched = dict(row)
        enriched["cv_similarity"] = round(sim_value, 4)
        ranked.append(enriched)

    ranked.sort(key=lambda item: _safe_float(item.get("cv_similarity")), reverse=True)
    return ranked[: max(1, int(max_results))]


def _filter_recommendation_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_lc = (query or "").strip().lower()
    if not query_lc:
        return list(rows)

    filtered: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join(
            [
                _friendly_value(row.get("title", "")),
                _friendly_value(row.get("company_profile", "")),
                _friendly_value(row.get("location", "")),
                _friendly_value(row.get("description", "")),
                _friendly_value(row.get("requirements", "")),
            ]
        ).lower()
        if query_lc in haystack:
            filtered.append(row)
    return filtered


def _normalize_href(url: str) -> str:
    clean = _friendly_value(url)
    if not clean:
        return ""
    if clean.lower().startswith(("http://", "https://")):
        return clean
    return f"https://{clean.lstrip('/')}"


def _domain_from_url(url: str) -> str:
    clean = _normalize_href(url)
    if not clean:
        return ""
    try:
        hostname = (urlparse(clean).hostname or "").lower().strip()
    except Exception:
        return ""
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _clean_listing_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_candidate_urls(*values: Any) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for value in values:
        text = _clean_listing_text(value)
        if not text:
            continue
        for match in URL_PATTERN.findall(text):
            candidate = _normalize_href(match.rstrip(".,);]}>"))
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(candidate)
    return urls


def _score_apply_candidate(candidate_url: str, job_domain: str, source_domain: str) -> float:
    score = 0.0
    candidate_lc = candidate_url.lower()
    domain = _domain_from_url(candidate_url)
    if not domain:
        return score

    if candidate_lc.startswith("https://"):
        score += 2.0

    if any(token in candidate_lc for token in APPLY_HINT_TOKENS):
        score += 4.0

    if domain in URL_SHORTENER_HOSTS:
        score -= 5.0

    if source_domain and domain != source_domain:
        score += 3.0

    if job_domain and domain == job_domain:
        score += 1.0

    if any(
        ats_token in domain
        for ats_token in ("greenhouse", "lever", "ashby", "workday", "smartrecruiters", "jobvite", "icims")
    ):
        score += 2.5

    return score


def _best_apply_url(row: dict[str, Any]) -> str:
    source_key = str(row.get("source", ""))
    _source_label, source_feed_url = _source_meta(source_key)
    source_domain = _domain_from_url(source_feed_url)

    job_url = _normalize_href(str(row.get("job_url", "")).strip())
    apply_url = _normalize_href(str(row.get("apply_url", "")).strip())
    job_domain = _domain_from_url(job_url)

    if apply_url and job_url and apply_url != job_url:
        return apply_url
    if apply_url and not job_url:
        return apply_url

    candidates = _extract_candidate_urls(
        row.get("description", ""),
        row.get("requirements", ""),
        row.get("benefits", ""),
        row.get("original_description", ""),
        row.get("original_requirements", ""),
    )
    if not candidates:
        return apply_url or job_url

    ranked = sorted(
        candidates,
        key=lambda url: _score_apply_candidate(
            candidate_url=url,
            job_domain=job_domain,
            source_domain=source_domain,
        ),
        reverse=True,
    )
    top_candidate = ranked[0]
    if not top_candidate:
        return apply_url or job_url
    return top_candidate


def _listing_section_html(title: str, text: str) -> str:
    clean = _clean_listing_text(text)
    if not clean:
        return ""
    chunks: list[str] = []
    for block in re.split(r"\n+", clean):
        block = block.strip()
        if not block:
            continue
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\\[])|\\s+[-*•]\\s+", block)
        for part in sentence_parts:
            piece = part.strip(" -•\t")
            if not piece:
                continue
            if len(piece) > 170:
                wrapped = textwrap.wrap(piece, width=150, break_long_words=False, replace_whitespace=False)
                chunks.extend(chunk.strip() for chunk in wrapped if chunk.strip())
            else:
                chunks.append(piece)

    chunks = chunks[:22]
    if not chunks:
        chunks = [clean]
    items_html = "".join(f"<li>{html.escape(chunk)}</li>" for chunk in chunks)
    return (
        "<section class='listing-section'>"
        f"<h5>{html.escape(title)}</h5>"
        f"<ul class='listing-bullets'>{items_html}</ul>"
        "</section>"
    )


def _listing_overview_line(row: dict[str, Any]) -> str:
    items: list[str] = []
    salary = _friendly_value(row.get("salary_range", ""))
    employment_type = _friendly_value(row.get("employment_type", ""))
    department = _friendly_value(row.get("department", ""))
    experience = _friendly_value(row.get("required_experience", ""))
    education = _friendly_value(row.get("required_education", ""))
    if salary:
        items.append(f"Salary: {salary}")
    if employment_type:
        items.append(f"Type: {employment_type}")
    if department:
        items.append(f"Department: {department}")
    if experience:
        items.append(f"Experience: {experience}")
    if education:
        items.append(f"Education: {education}")
    return " | ".join(items)


def _score_meaning_html() -> str:
    return (
        "<details class='score-meaning-details'>"
        "<summary>What these scores mean</summary>"
        "<div class='score-meaning-grid'>"
        "<article class='score-meaning-item'>"
        "<h5>Trust (0-100)</h5>"
        "<p>How safe and legitimate the posting appears after blending rules and model legitimacy.</p>"
        "</article>"
        "<article class='score-meaning-item'>"
        "<h5>Quality (0-100)</h5>"
        "<p>How complete and useful the post is: compensation clarity, role detail, remote clarity, apply UX, and freshness.</p>"
        "</article>"
        "<article class='score-meaning-item'>"
        "<h5>Final Opportunity (0-100)</h5>"
        "<p>Combined ranking score: <strong>0.70 x Trust + 0.30 x Quality</strong> with trust guardrails for scam signals.</p>"
        "</article>"
        "<article class='score-meaning-item'>"
        "<h5>Fraud Risk %</h5>"
        "<p>Model-estimated fraud probability from text and metadata, shown as a percentage. Higher means more suspicious.</p>"
        "</article>"
        "</div>"
        "</details>"
    )


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return -1.0
    parsed = pd.to_datetime(str(value), errors="coerce", utc=True)
    if pd.isna(parsed):
        return -1.0
    return float(parsed.value)


def _apply_filters(
    rows: list[dict[str, Any]],
    query: str,
    selected_sources: list[str] | None,
    badge_filter: str,
    risk_filter: str,
    sort_by: str,
) -> list[dict[str, Any]]:
    query_lc = (query or "").strip().lower()
    selected_sources = selected_sources or []
    filtered: list[dict[str, Any]] = []

    for row in rows:
        source = str(row.get("source", ""))
        if selected_sources and source not in selected_sources:
            continue

        if badge_filter and badge_filter != "All" and row.get("badge") != badge_filter:
            continue

        if risk_filter == "Hard Cap Triggered" and not _to_list(row.get("hard_caps")):
            continue
        if risk_filter == "High Risk Only" and row.get("fraud_risk_level") != "high":
            continue

        if query_lc:
            haystack = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("company_profile", "")),
                    str(row.get("location", "")),
                    str(row.get("description", "")),
                    str(row.get("requirements", "")),
                ]
            ).lower()
            if query_lc not in haystack:
                continue

        filtered.append(row)

    if sort_by == "Trust Score (high to low)":
        filtered.sort(key=lambda item: _safe_float(item.get("trust_score")), reverse=True)
    elif sort_by == "Quality Score (high to low)":
        filtered.sort(key=lambda item: _safe_float(item.get("quality_score")), reverse=True)
    elif sort_by == "Fraud Probability (high to low)":
        filtered.sort(key=lambda item: _safe_float(item.get("fraud_probability")), reverse=True)
    elif sort_by == "Newest First":
        filtered.sort(key=lambda item: _parse_timestamp(item.get("posted_date")), reverse=True)
    else:
        filtered.sort(key=lambda item: _safe_float(item.get("final_opportunity_score")), reverse=True)

    return filtered


def _friendly_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        name_value = value.get("Name") or value.get("name")
        if name_value:
            return _friendly_value(name_value)
        ordered_keys = ["Value", "value", "Type", "type", "Code", "code"]
        ordered_parts = [_friendly_value(value.get(key, "")) for key in ordered_keys]
        ordered_parts = [part for part in ordered_parts if part]
        if ordered_parts:
            return " ".join(ordered_parts)
        parts: list[str] = []
        for key, item in value.items():
            clean_item = _friendly_value(item)
            if clean_item:
                parts.append(f"{key}: {clean_item}")
        return ", ".join(parts)
    if isinstance(value, (list, tuple, set)):
        items = [_friendly_value(item) for item in value]
        items = [item for item in items if item]
        deduped = list(dict.fromkeys(items))
        return ", ".join(deduped)

    text = str(value).strip()
    if not text:
        return ""
    if text[0] in {"{", "["} and text[-1] in {"}", "]"}:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if parsed is not None:
            reparsed = _friendly_value(parsed)
            if reparsed:
                return reparsed
    return re.sub(r"\s+", " ", text)


def _friendly_posted_date(row: dict[str, Any]) -> str:
    posted_raw = _friendly_value(row.get("posted_date", ""))
    age_days = row.get("posting_age_days")
    formatted = posted_raw or "Unknown"
    if posted_raw:
        parsed = pd.to_datetime(posted_raw, errors="coerce", utc=True)
        if not pd.isna(parsed):
            formatted = parsed.strftime("%b %d, %Y")
    if isinstance(age_days, int):
        if age_days == 0:
            formatted = f"{formatted} (today)"
        elif age_days == 1:
            formatted = f"{formatted} (1 day ago)"
        elif age_days > 1:
            formatted = f"{formatted} ({age_days} days ago)"
    return formatted


def _score_band(score: float) -> str:
    if score >= 85:
        return "very strong"
    if score >= 70:
        return "strong"
    if score >= 50:
        return "mixed"
    return "high risk"


def _component_notes(
    components: dict[str, Any],
    maxima: dict[str, float],
    labels: dict[str, str],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    watchouts: list[str] = []
    for key, max_value in maxima.items():
        raw_value = _safe_float(components.get(key, 0.0))
        ratio = raw_value / max_value if max_value > 0 else 0.0
        label = labels.get(key, key.replace("_", " "))
        if ratio >= 0.75:
            strengths.append(f"{label.title()} is strong ({raw_value:.0f}/{max_value:.0f})")
        elif ratio <= 0.45:
            watchouts.append(f"{label.title()} is weak ({raw_value:.0f}/{max_value:.0f})")
    return strengths[:2], watchouts[:2]


def _bullet_list_html(items: list[str], empty_label: str, css_class: str) -> str:
    safe_items = [item.strip() for item in items if item and item.strip()]
    if not safe_items:
        safe_items = [empty_label]
    list_items = "".join(f"<li>{html.escape(item)}</li>" for item in safe_items[:3])
    return f"<ul class='{css_class}'>{list_items}</ul>"


def _signal_has_matches(matched_signals: dict[str, Any], key: str) -> bool:
    value = matched_signals.get(key, [])
    if isinstance(value, list):
        return any(bool(_friendly_value(item)) for item in value)
    return bool(_friendly_value(value))


def _has_phrase(items: list[str], phrase: str) -> bool:
    needle = phrase.lower()
    return any(needle in item.lower() for item in items)


def _rule_list_html(items: list[str], empty_label: str, css_class: str) -> str:
    if not items:
        items = [empty_label]
    list_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<ul class='rule-list {css_class}'>{list_items}</ul>"


def _rule_display_text(rule_label: str) -> str:
    clean = re.sub(r"^\[(BOOST|RISK|QUALITY|BONUS)\]\s*", "", rule_label, flags=re.IGNORECASE).strip()
    return clean[:1].upper() + clean[1:] if clean else rule_label


def _top_rule_text(rule_labels: list[str], include_prefix: str, limit: int = 2) -> list[str]:
    prefix = include_prefix.upper()
    selected: list[str] = []
    for label in rule_labels:
        if label.upper().startswith(prefix):
            selected.append(_rule_display_text(label))
        if len(selected) >= limit:
            break
    return selected


def _collect_trust_rule_debug(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    matched_signals = row.get("matched_signals", {})
    if not isinstance(matched_signals, dict):
        matched_signals = {}

    positive_evidence = [_friendly_value(item) for item in _to_list(row.get("positive_evidence"))]
    apply_url = _normalize_href(_friendly_value(row.get("apply_url", "")))
    company = _friendly_value(row.get("company_profile", ""))

    trust_checks = [
        ("[BOOST] Apply link uses HTTPS", apply_url.startswith("https://")),
        ("[BOOST] Apply domain is known ATS", bool(row.get("is_known_ats_domain", False))),
        ("[BOOST] Apply domain is major job board", bool(row.get("is_major_job_board_domain", False))),
        (
            "[BOOST] Apply domain matches posting domain",
            _has_phrase(positive_evidence, "apply domain matches posting domain"),
        ),
        ("[BOOST] Company information is present", bool(company)),
        ("[RISK] Messaging-app recruitment signal", _signal_has_matches(matched_signals, "messaging_app")),
        (
            "[RISK] Interview-by-text/chat signal",
            _signal_has_matches(matched_signals, "text_only_interview")
            or _signal_has_matches(matched_signals, "interview_over_messaging"),
        ),
        ("[RISK] Pay-to-get-paid signal", _signal_has_matches(matched_signals, "pay_to_get_paid")),
        ("[RISK] Fake-check signal", _signal_has_matches(matched_signals, "fake_check")),
        ("[RISK] Task-scam signal", _signal_has_matches(matched_signals, "task_scam")),
        (
            "[RISK] Early banking/ID request signal",
            _signal_has_matches(matched_signals, "bank_info_request"),
        ),
        ("[RISK] Crypto or gift-card payment signal", _signal_has_matches(matched_signals, "crypto_or_gift_card")),
        ("[RISK] URL shortener signal", _has_phrase([_friendly_value(item) for item in _to_list(row.get("risk_flags"))], "shortener")),
        ("[RISK] Hard cap triggered", bool(row.get("trust_cap_applied", False))),
    ]

    fired = [label for label, state in trust_checks if state]
    not_fired = [label for label, state in trust_checks if not state]
    return fired, not_fired


def _collect_quality_rule_debug(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    matched_signals = row.get("matched_signals", {})
    if not isinstance(matched_signals, dict):
        matched_signals = {}

    positive_evidence = [_friendly_value(item) for item in _to_list(row.get("positive_evidence"))]
    description = _friendly_value(row.get("description", ""))
    requirements = _friendly_value(row.get("requirements", ""))
    employment_type = _friendly_value(row.get("employment_type", ""))
    location = _friendly_value(row.get("location", ""))
    company = _friendly_value(row.get("company_profile", ""))
    apply_url = _normalize_href(_friendly_value(row.get("apply_url", "")))
    posting_age_days = row.get("posting_age_days")

    desc_words = len(description.split())
    req_words = len(requirements.split())
    benefit_category_count = int(_safe_float(row.get("benefit_category_count", 0)))

    benefits_explicit = _has_phrase(positive_evidence, "benefits section is explicitly provided")
    benefits_embedded = _has_phrase(positive_evidence, "benefits/perks are described in posting text")
    benefits_available = benefits_explicit or benefits_embedded

    qualification_embedded = _has_phrase(positive_evidence, "qualification details are embedded in description")
    sparse_req_offset = _has_phrase(
        positive_evidence,
        "detailed description offsets sparse requirements section",
    )
    requirements_or_qualification_present = req_words >= 30 or qualification_embedded or sparse_req_offset

    responsibilities_plus_qualification = _has_phrase(
        positive_evidence,
        "role includes both responsibilities and qualification cues",
    )
    structured_role_cues_present = _signal_has_matches(
        matched_signals,
        "responsibility_terms",
    ) or _signal_has_matches(
        matched_signals,
        "qualification_section_terms",
    )

    quality_checks = [
        (
            "[QUALITY] Salary/compensation clarity present",
            _has_phrase(positive_evidence, "salary signal is present")
            or _has_phrase(positive_evidence, "compensation details are described"),
        ),
        ("[QUALITY] Employment type provided", bool(employment_type)),
        ("[QUALITY] Location provided", bool(location)),
        ("[QUALITY] Company context provided", bool(company)),
        ("[QUALITY] Benefits information present (field or description)", benefits_available),
        ("[QUALITY] Multi-category benefit package", benefit_category_count >= 3),
        (
            "[QUALITY] Company mission/culture context described",
            _has_phrase(positive_evidence, "company context is clearly described"),
        ),
        ("[QUALITY] Description has substantial detail", desc_words >= 150),
        (
            "[QUALITY] Requirements/qualification detail present (field or embedded)",
            requirements_or_qualification_present,
        ),
        (
            "[QUALITY] Structured role cues present (responsibilities/qualifications)",
            structured_role_cues_present,
        ),
        (
            "[QUALITY] Remote-policy clarity terms present",
            _signal_has_matches(matched_signals, "remote_policy_terms"),
        ),
        (
            "[QUALITY] Timezone/region clarity terms present",
            _signal_has_matches(matched_signals, "timezone_or_region"),
        ),
        ("[QUALITY] Apply link uses HTTPS", apply_url.startswith("https://")),
        (
            "[QUALITY] Hiring process details described",
            _has_phrase(positive_evidence, "hiring process details are described"),
        ),
        (
            "[QUALITY] Posting freshness acceptable (<=30 days)",
            isinstance(posting_age_days, int) and posting_age_days <= 30,
        ),
        ("[BONUS] Benefits provided in dedicated benefits field", benefits_explicit),
        ("[BONUS] Benefits inferred from description text", benefits_embedded),
        ("[BONUS] Qualification details embedded in description", qualification_embedded),
        ("[BONUS] Long description offset sparse requirements", sparse_req_offset),
        ("[BONUS] Responsibilities + qualification cue combo", responsibilities_plus_qualification),
    ]

    fired = [label for label, state in quality_checks if state]
    not_fired = [label for label, state in quality_checks if not state]
    return fired, not_fired


def _build_rule_debug_html(row: dict[str, Any], trust_score: float, quality_score: float) -> str:
    trust_fired, trust_not_fired = _collect_trust_rule_debug(row)
    quality_fired, quality_not_fired = _collect_quality_rule_debug(row)

    trust_band = _score_band(trust_score)
    quality_band = _score_band(quality_score)

    return (
        "<details class='rule-debug-details'>"
        "<summary>Trust + Quality Rule Debug</summary>"
        "<div class='rule-debug-overview'>"
        f"<span>Trust profile: <strong>{html.escape(trust_band)}</strong> ({trust_score:.1f}/100)</span>"
        f"<span>Quality profile: <strong>{html.escape(quality_band)}</strong> ({quality_score:.1f}/100)</span>"
        "</div>"
        "<div class='rule-debug-grid'>"
        "<div class='rule-debug-col'>"
        f"<h5>Trust Rules Fired ({len(trust_fired)})</h5>"
        f"{_rule_list_html(trust_fired, 'No trust rules fired.', 'rule-fired')}"
        f"<h5>Trust Rules Not Fired ({len(trust_not_fired)})</h5>"
        f"{_rule_list_html(trust_not_fired, 'All trust rules fired.', 'rule-not-fired')}"
        "</div>"
        "<div class='rule-debug-col'>"
        f"<h5>Quality Rules Fired ({len(quality_fired)})</h5>"
        f"{_rule_list_html(quality_fired, 'No quality rules fired.', 'rule-fired')}"
        f"<h5>Quality Rules Not Fired ({len(quality_not_fired)})</h5>"
        f"{_rule_list_html(quality_not_fired, 'All quality rules fired.', 'rule-not-fired')}"
        "</div>"
        "</div>"
        "</details>"
    )


def _build_explanation_html(row: dict[str, Any]) -> str:
    trust_score = _safe_float(row.get("trust_score"))
    quality_score = _safe_float(row.get("quality_score"))
    final_score = _safe_float(row.get("final_opportunity_score"))
    badge = str(row.get("badge", "")).strip()
    guidance = BADGE_GUIDANCE.get(badge, "Review the posting carefully before applying.")

    hard_caps = _to_list(row.get("hard_caps"))
    hard_cap_reasons = [
        _friendly_value(item.get("reason", ""))
        for item in hard_caps
        if isinstance(item, dict) and item.get("reason")
    ]
    risk_flags = [_friendly_value(item) for item in _to_list(row.get("risk_flags"))]
    positives = [_friendly_value(item) for item in _to_list(row.get("positive_evidence"))]

    trust_fired_rules, _ = _collect_trust_rule_debug(row)
    quality_fired_rules, _ = _collect_quality_rule_debug(row)
    trust_boost_notes = [
        f"Trust rule confirmed: {item}" for item in _top_rule_text(trust_fired_rules, "[BOOST]", limit=2)
    ]
    trust_risk_notes = [
        f"Trust risk rule triggered: {item}" for item in _top_rule_text(trust_fired_rules, "[RISK]", limit=3)
    ]
    quality_notes = [
        f"Quality rule confirmed: {item}" for item in _top_rule_text(quality_fired_rules, "[QUALITY]", limit=3)
    ]
    quality_bonus_notes = [
        f"Quality bonus: {item}" for item in _top_rule_text(quality_fired_rules, "[BONUS]", limit=2)
    ]

    trust_components = row.get("components", {}).get("trust", {})
    quality_components = row.get("components", {}).get("quality", {})
    trust_strengths, trust_watchouts = _component_notes(
        components=trust_components,
        maxima=TRUST_COMPONENT_MAXIMA,
        labels=TRUST_COMPONENT_LABELS,
    )
    quality_strengths, quality_watchouts = _component_notes(
        components=quality_components,
        maxima=QUALITY_COMPONENT_MAXIMA,
        labels=QUALITY_COMPONENT_LABELS,
    )

    primary_risks = _dedupe_text(hard_cap_reasons + trust_risk_notes + trust_watchouts + quality_watchouts + risk_flags)
    primary_positives = _dedupe_text(
        trust_boost_notes + quality_notes + quality_bonus_notes + trust_strengths + quality_strengths + positives
    )

    trust_band = _score_band(trust_score)
    quality_band = _score_band(quality_score)
    final_band = _score_band(final_score)
    fraud_probability = _safe_float(row.get("fraud_probability"))
    fraud_risk_percent = fraud_probability * 100.0
    summary_sentence = (
        f"Recommendation: {guidance} "
        f"Current signal mix looks {final_band}, with {trust_band} trust and {quality_band} quality."
    )

    rule_engine = _safe_float(row.get("rule_engine_score"))
    ml_legit = _safe_float(row.get("ml_legitimacy_score"))
    trust_pre_cap = _safe_float(row.get("trust_pre_cap_score"))
    formula_trust = (
        f"Trust = {TRUST_BLEND_WEIGHTS['rule_engine_score']:.2f} x Rules ({rule_engine:.1f}) "
        f"+ {TRUST_BLEND_WEIGHTS['ml_legitimacy_score']:.2f} x ML legitimacy ({ml_legit:.1f}) "
        f"= {trust_score:.1f}"
    )
    if trust_pre_cap > trust_score:
        formula_trust += f" after hard-cap adjustment from {trust_pre_cap:.1f}"
    formula_final = (
        f"Final = {FINAL_BLEND_WEIGHTS['trust_score']:.2f} x Trust ({trust_score:.1f}) "
        f"+ {FINAL_BLEND_WEIGHTS['quality_score']:.2f} x Quality ({quality_score:.1f}) "
        f"= {final_score:.1f}"
    )
    formula_fraud_risk = (
        f"Fraud Risk % = ML fraud probability ({fraud_probability:.4f}) x 100 = {fraud_risk_percent:.1f}%"
    )
    formula_legitimacy = f"ML legitimacy = (1 - fraud probability) x 100 = {ml_legit:.1f}"

    trust_breakdown = (
        f"Trust components: identity/apply {trust_components.get('identity_apply_integrity', 0):.1f}/30, "
        f"communication {trust_components.get('communication_safety', 0):.1f}/25, "
        f"monetary safety {trust_components.get('monetary_safety', 0):.1f}/20, "
        f"company evidence {trust_components.get('company_evidence', 0):.1f}/15."
    )
    quality_breakdown = (
        f"Quality components: transparency {quality_components.get('transparency', 0):.1f}/30, "
        f"role detail {quality_components.get('role_specificity', 0):.1f}/25, "
        f"remote clarity {quality_components.get('remote_clarity', 0):.1f}/20, "
        f"apply experience {quality_components.get('apply_experience', 0):.1f}/15, "
        f"freshness {quality_components.get('freshness', 0):.1f}/10."
    )

    positives_html = _bullet_list_html(
        items=primary_positives,
        empty_label="Basic legitimacy signals are present, but supporting evidence is limited.",
        css_class="evidence-list positive-list",
    )
    risks_html = _bullet_list_html(
        items=primary_risks,
        empty_label="No major scam red flags were triggered by current rules.",
        css_class="evidence-list risk-list",
    )

    rule_debug_html = _build_rule_debug_html(
        row=row,
        trust_score=trust_score,
        quality_score=quality_score,
    )
    score_meaning_html = _score_meaning_html()

    return (
        "<div class='reasoning-block'>"
        f"<p class='reasoning-summary'>{html.escape(summary_sentence)}</p>"
        "<div class='reasoning-columns'>"
        "<article class='reasoning-col'>"
        "<h4>Why this scored well</h4>"
        f"{positives_html}"
        "</article>"
        "<article class='reasoning-col'>"
        "<h4>What to review before applying</h4>"
        f"{risks_html}"
        "</article>"
        "</div>"
        "<details class='formula-details'>"
        "<summary>How this score was computed</summary>"
        "<div class='formula-grid'>"
        "<article class='formula-item'>"
        "<h5>Trust Blend</h5>"
        f"<p>{html.escape(formula_trust)}</p>"
        f"<p>{html.escape(formula_legitimacy)}</p>"
        "</article>"
        "<article class='formula-item'>"
        "<h5>Final Opportunity Blend</h5>"
        f"<p>{html.escape(formula_final)}</p>"
        "</article>"
        "<article class='formula-item'>"
        "<h5>Fraud Risk Calculation</h5>"
        f"<p>{html.escape(formula_fraud_risk)}</p>"
        "</article>"
        "<article class='formula-item'>"
        "<h5>Component Breakdown</h5>"
        f"<p>{html.escape(trust_breakdown)}</p>"
        f"<p>{html.escape(quality_breakdown)}</p>"
        "</article>"
        "</div>"
        "</details>"
        f"{score_meaning_html}"
        f"{rule_debug_html}"
        "</div>"
    )


def _risk_color(risk_percent: float) -> str:
    if risk_percent <= 10:
        return "#0f9d58"
    if risk_percent <= 25:
        return "#f29900"
    return "#d93025"


def _score_chip(label: str, value: float, color: str | None = None) -> str:
    chip_color = color or _score_color(value)
    return (
        "<span class='score-chip' style='border-color:{c}; color:{c};'>"
        "{label}: {value:.1f}</span>"
    ).format(c=chip_color, label=html.escape(label), value=value)


def _score_bar(label: str, value: float, color: str, inverse: bool = False) -> str:
    normalized = max(0.0, min(100.0, value))
    width = 100.0 - normalized if inverse else normalized
    return (
        "<div class='score-bar'>"
        f"<div class='score-bar-label'>{html.escape(label)} <span>{value:.1f}</span></div>"
        "<div class='score-bar-track'>"
        f"<div class='score-bar-fill' style='background:{color}; width:{width:.1f}%;'></div>"
        "</div>"
        "</div>"
    )


def _render_job_card(row: dict[str, Any], row_number: int) -> str:
    title = html.escape(_friendly_value(row.get("title", "Untitled role")) or "Untitled role")
    company = html.escape(_friendly_value(row.get("company_profile", "Unknown company")) or "Unknown company")
    location = html.escape(_friendly_value(row.get("location", "Unknown location")) or "Unknown location")
    employment_type = html.escape(_friendly_value(row.get("employment_type", "")))
    posted_date = html.escape(_friendly_posted_date(row))
    salary_range = html.escape(_friendly_value(row.get("salary_range", "")))
    badge = str(row.get("badge", ""))
    badge_color = BADGE_COLORS.get(badge, "#5f6368")
    badge_html = (
        f"<span class='badge-pill' style='background:{badge_color};'>{html.escape(badge)}</span>"
        if badge
        else ""
    )

    trust = _safe_float(row.get("trust_score"))
    quality = _safe_float(row.get("quality_score"))
    final_score = _safe_float(row.get("final_opportunity_score"))
    fraud_prob = _safe_float(row.get("fraud_probability")) * 100.0
    cv_similarity = _safe_float(row.get("cv_similarity")) * 100.0

    fraud_chip_color = _risk_color(fraud_prob)
    score_chips = [
        _score_chip("Trust", trust),
        _score_chip("Quality", quality),
        _score_chip("Final", final_score),
        _score_chip("Fraud Risk %", fraud_prob, color=fraud_chip_color),
    ]
    if "cv_similarity" in row:
        score_chips.append(_score_chip("CV Match %", cv_similarity, color=_score_color(cv_similarity)))
    score_row = " ".join(score_chips)
    score_bars = "".join(
        [
            _score_bar("Trust", trust, _score_color(trust)),
            _score_bar("Quality", quality, _score_color(quality)),
            _score_bar("Final", final_score, _score_color(final_score)),
            _score_bar("Fraud Risk", fraud_prob, _risk_color(fraud_prob), inverse=True),
        ]
    )

    source_key = str(row.get("source", ""))
    source_label, source_url = _source_meta(source_key)
    source_url = _normalize_href(source_url)
    source_link = (
        f"<a href='{html.escape(source_url)}' target='_blank' rel='noopener noreferrer'>{html.escape(source_label)}</a>"
        if source_url
        else html.escape(source_label)
    )
    job_url = _normalize_href(str(row.get("job_url", "")).strip())
    apply_url = _best_apply_url(row)

    apply_button_html = (
        "<a class='apply-cta-btn' "
        f"href='{html.escape(apply_url)}' target='_blank' rel='noopener noreferrer'>Apply Now</a>"
        if apply_url
        else ""
    )
    header_actions = "".join(
        f"<span class='header-action'>{button_html}</span>"
        for button_html in [apply_button_html]
        if button_html
    )
    header_actions_html = f"<div class='job-header-actions'>{header_actions}</div>" if header_actions else ""

    reasons = _build_explanation_html(row)

    detected_language = _friendly_value(row.get("detected_language", "unknown")).lower() or "unknown"
    language_confidence = _safe_float(row.get("language_confidence", 0.0))
    translation_applied = bool(row.get("translation_applied", False))
    translation_error = _friendly_value(row.get("translation_error", ""))
    language_meta_html = ""
    if detected_language not in {"unknown", "en"}:
        language_state = "translated for scoring" if translation_applied else "not translated"
        if translation_error:
            language_state = f"{language_state} ({translation_error[:80]})"
        language_meta_html = (
            "<span class='lang-note'><strong>Language:</strong> "
            f"{html.escape(detected_language)} ({language_confidence:.2f}) - {html.escape(language_state)}</span>"
        )

    trust_cap_applied = bool(row.get("trust_cap_applied", False))
    hard_caps = _to_list(row.get("hard_caps"))
    cap_reason = ""
    if hard_caps and isinstance(hard_caps[0], dict):
        cap_reason = _friendly_value(hard_caps[0].get("reason", ""))
    cap_notice = (
        "<div class='cap-note'>Trust was capped due to a high-severity scam pattern"
        + (f": {html.escape(cap_reason)}" if cap_reason else ".")
        + "</div>"
        if trust_cap_applied
        else ""
    )

    listing_overview = _listing_overview_line(row)
    listing_sections = [
        _listing_section_html("Description", row.get("description", "")),
        _listing_section_html("Requirements / Qualifications", row.get("requirements", "")),
        _listing_section_html("Benefits / Perks", row.get("benefits", "")),
    ]
    original_description = _clean_listing_text(row.get("original_description", ""))
    original_requirements = _clean_listing_text(row.get("original_requirements", ""))
    if bool(row.get("translation_applied", False)) and original_description:
        original_lang = _friendly_value(row.get("detected_language", "")).upper() or "Original"
        listing_sections.append(
            _listing_section_html(
                f"Original Description ({original_lang})",
                original_description,
            )
        )
    if bool(row.get("translation_applied", False)) and original_requirements:
        original_lang = _friendly_value(row.get("detected_language", "")).upper() or "Original"
        listing_sections.append(
            _listing_section_html(
                f"Original Requirements ({original_lang})",
                original_requirements,
            )
        )
    listing_sections_html = "".join(section for section in listing_sections if section)
    if not listing_sections_html:
        listing_sections_html = (
            "<section class='listing-section'><p>No full listing text is available from this source.</p></section>"
        )

    listing_notes: list[str] = []
    if listing_overview:
        listing_notes.append(listing_overview)
    if apply_url and job_url and apply_url != job_url:
        listing_notes.append("Apply link appears to be a direct application destination.")
    elif apply_url:
        listing_notes.append("Direct external apply link was not available, so source listing link is used.")
    listing_notes_html = ""
    if listing_notes or job_url:
        note_items = [f"<li>{html.escape(note)}</li>" for note in listing_notes]
        if job_url:
            note_items.append(
                "<li>Source listing URL: "
                f"<a href='{html.escape(job_url)}' target='_blank' rel='noopener noreferrer'>{html.escape(job_url)}</a>"
                "</li>"
            )
        listing_notes_html = "<ul class='listing-note-list'>" + "".join(note_items) + "</ul>"

    listing_details_html = (
        "<details class='listing-details'>"
        "<summary>View full listing text and metadata</summary>"
        f"{listing_notes_html}"
        f"{listing_sections_html}"
        "</details>"
    )

    meta_line_parts = [f"{company}", f"{location}"]
    if employment_type:
        meta_line_parts.append(employment_type)
    if salary_range:
        meta_line_parts.append(f"Salary: {salary_range}")
    meta_line = " | ".join(part for part in meta_line_parts if part)

    return f"""
    <div class="job-card">
      <div class="job-card-header">
        <div class="job-title-block">
          <div class="job-rank">#{row_number}</div>
          <h3>{title}</h3>
        </div>
        <div class="job-header-right">
          {badge_html}
          {header_actions_html}
        </div>
      </div>
      <div class="job-meta">{meta_line}</div>
      <div class="job-submeta">
        <span><strong>Source:</strong> {source_link}</span>
        <span><strong>Posted:</strong> {posted_date or "Unknown"}</span>
        {"<span><strong>CV Match:</strong> " + f"{cv_similarity:.1f}%" + "</span>" if "cv_similarity" in row else ""}
        {language_meta_html}
      </div>
      <div class="score-row">{score_row}</div>
      <div class="score-bars">{score_bars}</div>
      {cap_notice}
      <div class="reasoning">{reasons}</div>
      {listing_details_html}
    </div>
    """


def _render_page(
    filtered_rows: list[dict[str, Any]],
    page: int,
    page_size: int,
) -> tuple[int, str, str]:
    total = len(filtered_rows)
    if total == 0:
        empty_html = (
            "<div class='empty-state'>No jobs match your current filters.<br>"
            "Try broadening the source filter, clearing search, or lowering risk strictness.</div>"
        )
        return 1, "Page 0 / 0", empty_html

    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = filtered_rows[start:end]

    cards = []
    for idx, row in enumerate(page_rows, start=start + 1):
        cards.append(_render_job_card(row=row, row_number=idx))

    page_label = f"Page {page} / {total_pages}  |  Showing {start + 1}-{min(end, total)} of {total}"
    return page, page_label, "\n".join(cards)


def _overview_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "Source",
                "Source Feed URL",
                "Role",
                "Company",
                "Location",
                "Posted Date",
                "Trust",
                "Quality",
                "Final",
                "Badge",
                "Fraud Risk %",
                "Listing URL",
                "Apply URL",
            ]
        )

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        source_key = str(row.get("source", ""))
        source_label, source_url = _source_meta(source_key)
        listing_url = _normalize_href(_friendly_value(row.get("job_url", "")))
        apply_url = _best_apply_url(row)
        table_rows.append(
            {
                "Source": source_label,
                "Source Feed URL": _normalize_href(source_url),
                "Role": _friendly_value(row.get("title", "")),
                "Company": _friendly_value(row.get("company_profile", "")),
                "Location": _friendly_value(row.get("location", "")),
                "Posted Date": _friendly_posted_date(row),
                "Trust": round(_safe_float(row.get("trust_score")), 2),
                "Quality": round(_safe_float(row.get("quality_score")), 2),
                "Final": round(_safe_float(row.get("final_opportunity_score")), 2),
                "Badge": _friendly_value(row.get("badge", "")),
                "Fraud Risk %": round(_safe_float(row.get("fraud_probability")) * 100.0, 2),
                "Listing URL": listing_url,
                "Apply URL": apply_url,
            }
        )
    return pd.DataFrame(table_rows)


def _build_overview_csv(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    table_df = _overview_table(rows)
    if table_df.empty:
        return None
    return _build_csv_from_dataframe(table_df, prefix="remote_trust_live_jobs_")


def _build_csv_from_dataframe(table_df: pd.DataFrame, prefix: str) -> str | None:
    if table_df.empty:
        return None
    export_dir = ROOT / "artifacts" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=prefix,
        dir=str(export_dir),
        delete=False,
        encoding="utf-8",
    )
    temp_file_path = Path(temp_file.name)
    temp_file.close()
    table_df.to_csv(temp_file_path, index=False)
    return str(temp_file_path)


def _overview_table_recommendations(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "Source",
                "Role",
                "Company",
                "Location",
                "CV Match %",
                "Trust",
                "Quality",
                "Final",
                "Fraud Risk %",
                "Badge",
                "Listing URL",
                "Apply URL",
            ]
        )

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        source_key = str(row.get("source", ""))
        source_label, _source_url = _source_meta(source_key)
        listing_url = _normalize_href(_friendly_value(row.get("job_url", "")))
        apply_url = _best_apply_url(row)
        table_rows.append(
            {
                "Source": source_label,
                "Role": _friendly_value(row.get("title", "")),
                "Company": _friendly_value(row.get("company_profile", "")),
                "Location": _friendly_value(row.get("location", "")),
                "CV Match %": round(_safe_float(row.get("cv_similarity")) * 100.0, 2),
                "Trust": round(_safe_float(row.get("trust_score")), 2),
                "Quality": round(_safe_float(row.get("quality_score")), 2),
                "Final": round(_safe_float(row.get("final_opportunity_score")), 2),
                "Fraud Risk %": round(_safe_float(row.get("fraud_probability")) * 100.0, 2),
                "Badge": _friendly_value(row.get("badge", "")),
                "Listing URL": listing_url,
                "Apply URL": apply_url,
            }
        )
    return pd.DataFrame(table_rows)


def _status_text(total_raw: int, total_filtered: int, source_count: int) -> str:
    source_word = "source" if source_count == 1 else "sources"
    return (
        f"Pulled **{total_raw}** jobs from **{source_count}** live {source_word}. "
        f"After filters, **{total_filtered}** jobs are shown."
    )


def _negative_status_text(total_raw: int, total_filtered: int, source_count: int) -> str:
    source_word = "group" if source_count == 1 else "groups"
    return (
        f"Loaded **{total_raw}** adversarial scam samples across **{source_count}** {source_word}. "
        f"After filters, **{total_filtered}** samples are shown."
    )


def _recommendation_status_text(
    total_raw: int,
    total_filtered: int,
    source_count: int,
    similarity_threshold: float,
    min_trust_score: float,
) -> str:
    source_word = "source" if source_count == 1 else "sources"
    return (
        f"Scanned **{total_raw}** jobs from **{source_count}** {source_word}. "
        f"Found **{total_filtered}** recommendations with CV similarity >= **{similarity_threshold:.2f}** "
        f"and Trust >= **{min_trust_score:.1f}**."
    )


def refresh_dashboard(
    per_source: int,
    query: str,
    selected_sources: list[str] | None,
    badge_filter: str,
    risk_filter: str,
    sort_by: str,
    page_size: int,
):
    if DETECTOR is None:
        yield (
            [],
            [],
            1,
            f"### Error\n{DETECTOR_STATUS}",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>Model not loaded. Check artifacts and restart.</div>",
            None,
        )
        return

    per_source_value = int(per_source)
    selected_source_map = _source_map_from_selection(selected_sources)
    selected_source_count = len(selected_source_map)
    expected_max = per_source_value * selected_source_count

    yield (
        [],
        [],
        1,
        (
            f"Fetching from **{selected_source_count}** selected source(s), up to "
            f"**{expected_max}** jobs..."
        ),
        pd.DataFrame(),
        "Page 0 / 0",
        "<div class='empty-state'>Fetching live postings...</div>",
        None,
    )

    jobs = fetch_jobs_from_sources(
        selected_source_map,
        per_source=per_source_value,
        fail_fast=False,
        parallel=True,
    )

    if not jobs:
        yield (
            [],
            [],
            1,
            "No jobs were fetched from the selected sources. Try refreshing again or reducing source constraints.",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>No jobs loaded yet.</div>",
            None,
        )
        return

    use_progressive_mode = len(jobs) >= PROGRESSIVE_RENDER_THRESHOLD

    if not use_progressive_mode:
        scored = score_live_jobs(
            detector=DETECTOR,
            live_jobs=jobs,
            with_explanations=False,
            with_heuristics=True,
            batch_size=BATCH_PREDICTION_SIZE,
        )

        filtered = _apply_filters(
            rows=scored,
            query=query,
            selected_sources=selected_sources,
            badge_filter=badge_filter,
            risk_filter=risk_filter,
            sort_by=sort_by,
        )

        page = 1
        page, page_text, cards_html = _render_page(filtered_rows=filtered, page=page, page_size=int(page_size))
        overview = _overview_table(filtered)
        overview_csv_path = _build_overview_csv(filtered)
        source_count = len({str(row.get("source", "")).strip() for row in scored if row.get("source")})
        status = _status_text(total_raw=len(scored), total_filtered=len(filtered), source_count=source_count)
        yield scored, filtered, page, status, overview, page_text, cards_html, overview_csv_path
        return

    scored_accumulated: list[dict[str, Any]] = []
    total_jobs = len(jobs)
    chunk_size = max(1, min(PROGRESSIVE_SCORE_CHUNK_SIZE, total_jobs))
    total_chunks = max(1, math.ceil(total_jobs / chunk_size))

    for chunk_idx, start in enumerate(range(0, total_jobs, chunk_size), start=1):
        end = min(start + chunk_size, total_jobs)
        chunk_jobs = jobs[start:end]
        chunk_scored = score_live_jobs(
            detector=DETECTOR,
            live_jobs=chunk_jobs,
            with_explanations=False,
            with_heuristics=True,
            batch_size=BATCH_PREDICTION_SIZE,
        )
        scored_accumulated.extend(chunk_scored)

        filtered = _apply_filters(
            rows=scored_accumulated,
            query=query,
            selected_sources=selected_sources,
            badge_filter=badge_filter,
            risk_filter=risk_filter,
            sort_by=sort_by,
        )
        page = 1
        page, page_text, cards_html = _render_page(
            filtered_rows=filtered,
            page=page,
            page_size=int(page_size),
        )
        overview = _overview_table(filtered)
        overview_csv_path = _build_overview_csv(filtered)
        source_count = len(
            {str(row.get("source", "")).strip() for row in scored_accumulated if row.get("source")}
        )
        status = (
            _status_text(
                total_raw=len(scored_accumulated),
                total_filtered=len(filtered),
                source_count=source_count,
            )
            + f"  \nProcessing chunk **{chunk_idx}/{total_chunks}**..."
        )

        if chunk_idx == total_chunks:
            status = _status_text(
                total_raw=len(scored_accumulated),
                total_filtered=len(filtered),
                source_count=source_count,
            )

        yield (
            scored_accumulated,
            filtered,
            page,
            status,
            overview,
            page_text,
            cards_html,
            overview_csv_path,
        )


def refilter_dashboard(
    jobs_state: list[dict[str, Any]],
    query: str,
    selected_sources: list[str] | None,
    badge_filter: str,
    risk_filter: str,
    sort_by: str,
    page_size: int,
):
    jobs_state = jobs_state or []
    filtered = _apply_filters(
        rows=jobs_state,
        query=query,
        selected_sources=selected_sources,
        badge_filter=badge_filter,
        risk_filter=risk_filter,
        sort_by=sort_by,
    )
    page = 1
    page, page_text, cards_html = _render_page(filtered_rows=filtered, page=page, page_size=int(page_size))
    overview = _overview_table(filtered)
    overview_csv_path = _build_overview_csv(filtered)
    source_count = len({str(row.get("source", "")).strip() for row in jobs_state if row.get("source")})
    status = _status_text(total_raw=len(jobs_state), total_filtered=len(filtered), source_count=source_count)
    return filtered, page, status, overview, page_text, cards_html, overview_csv_path


def refresh_negative_samples(
    query: str,
    selected_sources: list[str] | None,
    badge_filter: str,
    risk_filter: str,
    sort_by: str,
    page_size: int,
):
    if DETECTOR is None:
        return (
            [],
            [],
            1,
            f"### Error\n{DETECTOR_STATUS}",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>Model not loaded. Check artifacts and restart.</div>",
            None,
        )

    negative_samples = _build_negative_samples()
    scored = score_live_jobs(
        detector=DETECTOR,
        live_jobs=negative_samples,
        with_explanations=False,
        with_heuristics=True,
        enable_i18n=False,
        batch_size=BATCH_PREDICTION_SIZE,
    )

    filtered = _apply_filters(
        rows=scored,
        query=query,
        selected_sources=selected_sources,
        badge_filter=badge_filter,
        risk_filter=risk_filter,
        sort_by=sort_by,
    )

    page = 1
    page, page_text, cards_html = _render_page(filtered_rows=filtered, page=page, page_size=int(page_size))
    overview = _overview_table(filtered)
    overview_csv_path = _build_overview_csv(filtered)
    source_count = len({str(row.get("source", "")).strip() for row in scored if row.get("source")})
    status = _negative_status_text(
        total_raw=len(scored),
        total_filtered=len(filtered),
        source_count=source_count,
    )
    return scored, filtered, page, status, overview, page_text, cards_html, overview_csv_path


def refresh_recommendations(
    per_source: int,
    selected_sources: list[str] | None,
    cv_text: str,
    similarity_threshold: float,
    min_trust_score: float,
    max_results: int,
    query: str,
    page_size: int,
):
    if DETECTOR is None:
        return (
            [],
            [],
            1,
            f"### Error\n{DETECTOR_STATUS}",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>Model not loaded. Check artifacts and restart.</div>",
            None,
        )

    clean_cv = re.sub(r"\s+", " ", (cv_text or "").strip())
    if not clean_cv:
        return (
            [],
            [],
            1,
            "Paste CV/resume text to generate recommendations.",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>No CV text provided yet.</div>",
            None,
        )

    per_source_value = int(per_source)
    selected_source_map = _source_map_from_selection(selected_sources)
    jobs = fetch_jobs_from_sources(
        selected_source_map,
        per_source=per_source_value,
        fail_fast=False,
        parallel=True,
    )
    if not jobs:
        return (
            [],
            [],
            1,
            "No jobs were fetched from selected sources. Try refreshing or lowering constraints.",
            pd.DataFrame(),
            "Page 0 / 0",
            "<div class='empty-state'>No source jobs available for recommendation.</div>",
            None,
        )

    scored = score_live_jobs(
        detector=DETECTOR,
        live_jobs=jobs,
        with_explanations=False,
        with_heuristics=True,
        batch_size=RECOMMENDATION_BATCH_SIZE,
    )
    recommended = _rank_cv_recommendations(
        cv_text=clean_cv,
        rows=scored,
        similarity_threshold=float(similarity_threshold),
        min_trust_score=float(min_trust_score),
        max_results=int(max_results),
    )
    filtered = _filter_recommendation_rows(recommended, query=query)

    page = 1
    page, page_text, cards_html = _render_page(filtered_rows=filtered, page=page, page_size=int(page_size))
    overview = _overview_table_recommendations(filtered)
    overview_csv_path = _build_csv_from_dataframe(
        overview,
        prefix="remote_trust_recommendations_",
    )
    source_count = len({str(row.get("source", "")).strip() for row in scored if row.get("source")})
    status = _recommendation_status_text(
        total_raw=len(scored),
        total_filtered=len(filtered),
        source_count=source_count,
        similarity_threshold=float(similarity_threshold),
        min_trust_score=float(min_trust_score),
    )
    return recommended, filtered, page, status, overview, page_text, cards_html, overview_csv_path


def refilter_recommendations(
    recommended_state: list[dict[str, Any]],
    query: str,
    page_size: int,
):
    recommended_state = recommended_state or []
    filtered = _filter_recommendation_rows(recommended_state, query=query)
    page = 1
    page, page_text, cards_html = _render_page(
        filtered_rows=filtered,
        page=page,
        page_size=int(page_size),
    )
    overview = _overview_table_recommendations(filtered)
    overview_csv_path = _build_csv_from_dataframe(
        overview,
        prefix="remote_trust_recommendations_",
    )
    status = (
        f"Loaded **{len(recommended_state)}** recommendations. "
        f"After search, **{len(filtered)}** are shown."
    )
    return filtered, page, status, overview, page_text, cards_html, overview_csv_path


def go_prev_page(filtered_state: list[dict[str, Any]], page: int, page_size: int):
    filtered_state = filtered_state or []
    target_page = max(1, int(page) - 1)
    page, page_text, cards_html = _render_page(
        filtered_rows=filtered_state,
        page=target_page,
        page_size=int(page_size),
    )
    return page, page_text, cards_html


def go_next_page(filtered_state: list[dict[str, Any]], page: int, page_size: int):
    filtered_state = filtered_state or []
    target_page = max(1, int(page) + 1)
    page, page_text, cards_html = _render_page(
        filtered_rows=filtered_state,
        page=target_page,
        page_size=int(page_size),
    )
    return page, page_text, cards_html


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

:root {
  --rt-bg: var(--body-background-fill, #f4f7fb);
  --rt-secondary: var(--background-fill-secondary, #f8fafc);
  --rt-panel: var(--block-background-fill, #ffffff);
  --rt-panel-soft: color-mix(in srgb, var(--rt-secondary) 88%, var(--body-background-fill, #f4f7fb) 12%);
  --rt-panel-strong: color-mix(in srgb, var(--rt-secondary) 72%, var(--body-background-fill, #f4f7fb) 28%);
  --rt-border: var(--block-border-color, #d6dee8);
  --rt-text: var(--body-text-color, #0f172a);
  --rt-text-muted: var(--body-text-color-subdued, #5f6f86);
  --rt-heading: var(--body-text-color, #0f172a);
  --rt-accent: var(--color-accent);
  --rt-accent-2: color-mix(in srgb, var(--color-accent) 55%, #ffd45a 45%);
  --rt-bg-orb-1: color-mix(in srgb, var(--color-accent) 28%, transparent);
  --rt-bg-orb-2: color-mix(in srgb, var(--rt-accent-2) 22%, transparent);
  --rt-grad-left: color-mix(in srgb, var(--color-accent-soft, #eaf5ff) 55%, var(--rt-bg) 45%);
  --rt-grad-mid: color-mix(in srgb, var(--rt-bg) 94%, #ffffff 6%);
  --rt-grad-right: color-mix(in srgb, var(--rt-accent-2) 14%, var(--rt-bg) 86%);
  --rt-shadow: var(--shadow-drop-lg);
}

body.dark .gradio-container,
.dark .gradio-container,
[data-theme="dark"] .gradio-container,
.gradio-container.dark,
.gradio-container[data-theme="dark"] {
  --rt-bg: var(--body-background-fill-dark, #0b1220);
  --rt-secondary: var(--background-fill-secondary-dark, #111c2f);
  --rt-panel: var(--block-background-fill-dark, #0f1a2b);
  --rt-panel-soft: color-mix(in srgb, var(--rt-secondary) 88%, var(--body-background-fill-dark, #0b1220) 12%);
  --rt-panel-strong: color-mix(in srgb, var(--rt-secondary) 72%, var(--body-background-fill-dark, #0b1220) 28%);
  --rt-border: var(--block-border-color-dark, #2a3b55);
  --rt-text: var(--body-text-color-dark, #e7efff);
  --rt-text-muted: var(--body-text-color-subdued-dark, #a8b8d0);
  --rt-heading: var(--body-text-color-dark, #f4f8ff);
  --rt-accent: var(--color-accent);
  --rt-accent-2: color-mix(in srgb, var(--color-accent) 58%, #ffb45d 42%);
  --rt-bg-orb-1: color-mix(in srgb, var(--color-accent) 36%, transparent);
  --rt-bg-orb-2: color-mix(in srgb, var(--rt-accent-2) 30%, transparent);
  --rt-grad-left: color-mix(in srgb, var(--color-accent) 24%, var(--rt-bg) 76%);
  --rt-grad-mid: color-mix(in srgb, var(--rt-bg) 96%, #ffffff 4%);
  --rt-grad-right: color-mix(in srgb, var(--rt-accent-2) 20%, var(--rt-bg) 80%);
  --rt-shadow: 0 16px 36px rgba(0, 0, 0, 0.35);
}

.gradio-container,
.gradio-container * {
  box-sizing: border-box;
}

.gradio-container {
  background:
    linear-gradient(
      90deg,
      var(--rt-grad-left) 0%,
      var(--rt-grad-mid) 48%,
      var(--rt-grad-right) 100%
    ),
    var(--rt-bg);
  color: var(--rt-text);
  font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
  max-width: 1220px !important;
  margin: 0 auto;
  padding: 12px 16px 30px;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose strong,
.gradio-container .prose span,
.gradio-container .prose li,
.gradio-container .prose a,
.gradio-container label,
.gradio-container legend,
.gradio-container p,
.gradio-container li,
.gradio-container span,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5 {
  color: var(--rt-text) !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
  font-family: "Space Grotesk", "Manrope", sans-serif;
  color: var(--rt-heading) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container option {
  color: var(--rt-text) !important;
  background: var(--rt-panel-soft) !important;
  border-color: var(--rt-border) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--rt-text-muted) !important;
  opacity: 0.9 !important;
}

.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-box {
  background: color-mix(in srgb, var(--rt-panel) 90%, transparent);
  border: 1px solid var(--rt-border);
  border-radius: 14px;
}

#rt-tabs button[role="tab"] {
  border-radius: 999px !important;
  border: 1px solid var(--rt-border) !important;
  background: color-mix(in srgb, var(--rt-panel) 92%, transparent) !important;
  color: var(--rt-text) !important;
  font-weight: 700 !important;
}

#rt-tabs button[role="tab"][aria-selected="true"] {
  background: linear-gradient(92deg, var(--rt-accent), var(--rt-accent-2)) !important;
  color: #ffffff !important;
  border-color: transparent !important;
  box-shadow: 0 8px 18px color-mix(in srgb, var(--rt-accent) 42%, transparent);
}

.landing-shell {
  position: relative;
  overflow: hidden;
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--rt-panel) 96%, transparent) 0%,
    color-mix(in srgb, var(--rt-panel-soft) 94%, transparent) 50%,
    color-mix(in srgb, var(--rt-panel-strong) 94%, transparent) 100%
  );
  border: 1px solid var(--rt-border);
  border-radius: 22px;
  padding: 24px;
  min-height: calc(100vh - 132px);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--rt-shadow);
}

.landing-content {
  position: relative;
  width: 100%;
  max-width: 1020px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  text-align: center;
  z-index: 1;
}

.landing-top-grid {
  width: 100%;
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 16px;
  align-items: stretch;
}

.landing-copy {
  text-align: left;
  padding: 6px 2px;
  animation: rtFadeUp 560ms ease-out both;
}

.landing-logo-wrap {
  width: 200px;
  height: 200px;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid var(--rt-border);
  box-shadow: 0 10px 24px rgba(14, 34, 58, 0.18);
  margin: 0 auto 2px;
}

.landing-logo {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.landing-kicker {
  margin: 0 0 6px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.8rem;
  color: color-mix(in srgb, var(--rt-accent) 72%, var(--rt-text));
  font-weight: 800;
}

.landing-title {
  margin: 0 0 10px;
  font-size: clamp(1.7rem, 3vw, 2.4rem);
  line-height: 1.08;
}

.landing-motto {
  margin: 0 0 9px;
  font-size: 0.98rem;
  line-height: 1.55;
  color: var(--rt-text);
  font-weight: 600;
  max-width: 560px;
}

.landing-pill-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 0 0 10px;
}

.landing-pill {
  border: 1px solid color-mix(in srgb, var(--rt-border) 70%, var(--rt-accent) 30%);
  background: color-mix(in srgb, var(--rt-panel-soft) 88%, var(--rt-accent) 12%);
  color: var(--rt-text);
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  padding: 4px 10px;
}

.landing-weight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 10px;
}

.weight-item {
  border: 1px solid var(--rt-border);
  background: color-mix(in srgb, var(--rt-panel-soft) 92%, transparent);
  border-radius: 11px;
  padding: 7px 8px;
}

.weight-label {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 5px;
  font-size: 0.75rem;
  color: var(--rt-text);
}

.weight-label strong {
  font-size: 0.74rem;
}

.weight-track {
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--rt-panel-strong) 82%, transparent);
  overflow: hidden;
}

.weight-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--rt-accent), var(--rt-accent-2));
  transform-origin: left;
  animation: rtBarGrow 900ms ease-out both;
}

.weight-fill.trust-fill { width: 70%; }
.weight-fill.quality-fill { width: 30%; }
.weight-fill.rules-fill { width: 45%; }
.weight-fill.ml-fill { width: 55%; }

.landing-visual-panel {
  border: 1px solid var(--rt-border);
  background: linear-gradient(
    130deg,
    color-mix(in srgb, var(--rt-panel-soft) 84%, var(--rt-accent) 16%),
    color-mix(in srgb, var(--rt-panel) 92%, var(--rt-accent-2) 8%)
  );
  border-radius: 16px;
  padding: 12px;
  display: grid;
  gap: 10px;
  align-content: start;
  animation: rtFadeUp 680ms ease-out both;
}

.signal-orbit {
  margin: 0 auto;
  width: 220px;
  height: 220px;
  border-radius: 999px;
  background: conic-gradient(
    from 0deg,
    color-mix(in srgb, var(--rt-accent) 85%, transparent),
    color-mix(in srgb, var(--rt-accent-2) 84%, transparent),
    color-mix(in srgb, var(--rt-accent) 85%, transparent)
  );
  padding: 2px;
  animation: rtSpin 14s linear infinite;
}

.signal-core {
  height: 100%;
  border-radius: inherit;
  border: 1px solid color-mix(in srgb, var(--rt-border) 74%, var(--rt-accent) 26%);
  background: color-mix(in srgb, var(--rt-panel) 90%, transparent);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  text-align: center;
  padding: 12px;
}

.signal-core .landing-logo-wrap {
  width: 90px;
  height: 90px;
  border-radius: 14px;
  margin: 0;
}

.signal-core-label {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--rt-text-muted) !important;
  font-weight: 700;
}

.signal-core-text {
  margin: 0;
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--rt-text) !important;
}

.visual-kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.visual-kpi {
  border: 1px solid var(--rt-border);
  background: color-mix(in srgb, var(--rt-panel) 90%, transparent);
  border-radius: 11px;
  padding: 8px;
  text-align: center;
}

.visual-kpi span {
  display: block;
  font-size: 0.68rem;
  color: var(--rt-text-muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.visual-kpi strong {
  display: block;
  margin-top: 3px;
  font-size: 0.8rem;
  color: var(--rt-text) !important;
}

.landing-method-grid {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin: 2px 0 2px;
}

.landing-point {
  border: 1px solid var(--rt-border);
  background: color-mix(in srgb, var(--rt-panel-soft) 92%, transparent);
  border-radius: 12px;
  padding: 11px;
  text-align: left;
  animation: rtFadeUp 560ms ease-out both;
}

.landing-point h3 {
  margin: 0 0 3px;
  font-size: 0.9rem;
}

.landing-point p {
  margin: 0;
  font-size: 0.82rem;
  color: color-mix(in srgb, var(--rt-text) 82%, var(--rt-text-muted) 18%) !important;
}

.landing-footnote {
  margin: 0 0 8px;
  font-size: 0.88rem;
  color: color-mix(in srgb, var(--rt-text) 74%, var(--rt-text-muted) 26%) !important;
}

.landing-sources-card {
  width: 100%;
  border: 1px solid var(--rt-border);
  border-radius: 12px;
  background: linear-gradient(
    95deg,
    color-mix(in srgb, var(--rt-panel-soft) 92%, transparent) 0%,
    color-mix(in srgb, var(--rt-panel) 90%, var(--rt-accent) 10%) 100%
  );
  padding: 10px 11px;
  text-align: left;
  animation: rtFadeUp 740ms ease-out both;
}

.landing-credit {
  margin-top: 10px;
  width: 100%;
  border: 1px solid color-mix(in srgb, var(--rt-border) 74%, var(--rt-accent) 26%);
  border-radius: 12px;
  background: linear-gradient(
    95deg,
    color-mix(in srgb, var(--rt-panel-soft) 90%, transparent) 0%,
    color-mix(in srgb, var(--rt-panel) 86%, var(--rt-accent) 14%) 100%
  );
  padding: 9px 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  text-align: center;
}

.landing-credit-logo {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  object-fit: cover;
  border: 1px solid var(--rt-border);
  background: var(--rt-panel);
}

.landing-credit p {
  margin: 0;
  font-size: 0.82rem;
  color: var(--rt-text) !important;
}

.landing-sources-head h4 {
  margin: 0;
  font-size: 0.92rem;
}

.landing-sources-head p {
  margin: 4px 0 8px;
  font-size: 0.8rem;
  color: var(--rt-text-muted) !important;
}

.landing-source-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 8px;
}

.landing-source-item {
  border: 1px solid color-mix(in srgb, var(--rt-border) 78%, var(--rt-accent) 22%);
  border-radius: 10px;
  background: color-mix(in srgb, var(--rt-panel) 92%, transparent);
  padding: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.source-name {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--rt-text) !important;
}

.source-kind {
  font-size: 0.67rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 1px solid color-mix(in srgb, var(--rt-border) 70%, var(--rt-accent) 30%);
  border-radius: 999px;
  padding: 2px 7px;
  color: color-mix(in srgb, var(--rt-accent) 72%, var(--rt-text)) !important;
  background: color-mix(in srgb, var(--rt-panel-soft) 84%, var(--rt-accent) 16%);
}

.landing-formula-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
}

.landing-formula {
  border: 1px solid var(--rt-border);
  background: color-mix(in srgb, var(--rt-panel) 90%, transparent);
  border-radius: 12px;
  padding: 10px;
  text-align: left;
}

.landing-formula h4 {
  margin: 0 0 5px;
  font-size: 0.87rem;
}

.landing-formula p {
  margin: 0;
  font-size: 0.8rem;
  line-height: 1.4;
}

.landing-formula code {
  font-size: 0.76rem;
  color: var(--rt-text) !important;
}

.landing-tech-details {
  width: 100%;
  border: 1px dashed color-mix(in srgb, var(--rt-border) 88%, transparent);
  background: color-mix(in srgb, var(--rt-panel-soft) 88%, transparent);
  border-radius: 12px;
  padding: 8px 10px;
}

.landing-tech-details summary {
  cursor: pointer;
  font-size: 0.87rem;
  font-weight: 700;
  color: var(--rt-heading) !important;
}

.landing-repro {
  width: 100%;
  text-align: left;
  border: none;
  background: transparent;
  border-radius: 0;
  padding: 8px 2px 0;
}

.landing-repro h3 {
  margin: 0 0 6px;
  font-size: 1rem;
}

.landing-repro-lead {
  margin: 0 0 8px;
  font-size: 0.87rem;
  line-height: 1.5;
}

.landing-repro-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 4px;
  font-size: 0.82rem;
  line-height: 1.45;
}

@keyframes rtFadeUp {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes rtBarGrow {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}

@keyframes rtSpin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .landing-copy,
  .landing-point,
  .landing-visual-panel,
  .weight-fill,
  .signal-orbit {
    animation: none !important;
  }
}

.hero-shell {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 14px;
  align-items: center;
  background: linear-gradient(120deg, color-mix(in srgb, var(--rt-panel) 94%, transparent), color-mix(in srgb, var(--rt-panel-strong) 95%, transparent));
  border: 1px solid var(--rt-border);
  border-radius: 18px;
  padding: 14px;
  box-shadow: var(--rt-shadow);
}

.hero-logo-wrap {
  width: 148px;
  height: 148px;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--rt-border);
}

.hero-logo {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.hero-kicker {
  margin: 0 0 4px;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: color-mix(in srgb, var(--rt-accent) 72%, var(--rt-text)) !important;
  font-weight: 800;
}

.hero-title {
  margin: 0 0 8px;
  font-size: clamp(1.45rem, 2.5vw, 1.9rem);
}

.hero-motto,
.hero-subnote,
.hero-copy p {
  color: var(--rt-text) !important;
}

.hero-motto {
  margin: 0 0 7px;
  font-size: 0.96rem;
  line-height: 1.5;
  font-weight: 600;
}

.hero-subnote {
  margin: 0 0 6px;
  font-size: 0.88rem;
  line-height: 1.5;
  color: var(--rt-text-muted) !important;
}

.hero-status {
  margin: 8px 0 0;
  padding: 8px 10px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--rt-panel-soft) 92%, transparent);
  border: 1px solid var(--rt-border);
  font-size: 0.82rem;
  color: var(--rt-text) !important;
}

.landing-status {
  margin-top: 10px;
}

.about-shell {
  background: linear-gradient(120deg, color-mix(in srgb, var(--rt-panel) 95%, transparent), color-mix(in srgb, var(--rt-panel-strong) 95%, transparent));
  border: 1px solid var(--rt-border);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--rt-shadow);
}

.about-banner {
  border: 1px solid color-mix(in srgb, var(--rt-border) 72%, var(--rt-accent) 28%);
  border-radius: 14px;
  padding: 12px;
  margin-bottom: 12px;
  background: linear-gradient(
    95deg,
    color-mix(in srgb, var(--rt-panel-soft) 84%, var(--rt-accent) 16%),
    color-mix(in srgb, var(--rt-panel) 90%, var(--rt-accent-2) 10%)
  );
}

.about-kicker {
  margin: 0 0 4px;
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: color-mix(in srgb, var(--rt-accent) 72%, var(--rt-text)) !important;
}

.about-shell h2 {
  margin: 0 0 6px;
  font-size: 1.6rem;
}

.about-intro {
  margin: 0;
  color: var(--rt-text-muted) !important;
}

.about-mindmesh {
  margin-top: 10px;
  border: 1px solid color-mix(in srgb, var(--rt-border) 74%, var(--rt-accent) 26%);
  background: color-mix(in srgb, var(--rt-panel) 90%, var(--rt-accent) 10%);
  border-radius: 12px;
  padding: 8px 10px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.about-mindmesh-logo {
  width: 56px;
  height: 56px;
  border-radius: 10px;
  border: 1px solid var(--rt-border);
  object-fit: cover;
  background: var(--rt-panel);
}

.about-mindmesh-copy {
  display: grid;
  gap: 2px;
}

.about-mindmesh-kicker {
  margin: 0;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--rt-text-muted) !important;
  font-weight: 700;
}

.about-mindmesh-title {
  margin: 0;
  font-size: 0.94rem;
  font-weight: 800;
  color: var(--rt-text) !important;
}

.about-tag-row {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.about-tag {
  border: 1px solid color-mix(in srgb, var(--rt-border) 68%, var(--rt-accent) 32%);
  background: color-mix(in srgb, var(--rt-panel) 88%, var(--rt-accent) 12%);
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--rt-text) !important;
}

.about-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
  gap: 14px;
}

.about-card {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 12px;
  border: 1px solid var(--rt-border);
  border-radius: 14px;
  padding: 12px;
  background: color-mix(in srgb, var(--rt-panel-soft) 92%, transparent);
}

.about-photo-wrap {
  width: 110px;
  height: 110px;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid var(--rt-border);
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(130deg, color-mix(in srgb, var(--rt-accent) 45%, transparent), color-mix(in srgb, var(--rt-accent-2) 35%, transparent));
}

.about-photo {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center 30%;
}

.about-photo-fallback {
  color: #ffffff;
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: 0.04em;
}

.about-card-copy h3 {
  margin: 0;
  font-size: 1.12rem;
}

.about-card-copy h4 {
  margin: 10px 0 6px;
  font-size: 0.95rem;
}

.about-role {
  margin: 3px 0 2px;
  font-weight: 700;
}

.about-pronouns {
  margin: 0 0 6px;
  color: var(--rt-text-muted) !important;
}

.about-location {
  margin: 0 0 8px;
  font-weight: 700;
}

.about-location i {
  margin-right: 6px;
  color: color-mix(in srgb, var(--rt-accent) 74%, var(--rt-text));
}

.about-card-copy p {
  margin: 6px 0;
  font-size: 0.88rem;
  line-height: 1.45;
}

.about-link a,
.about-link-list a {
  color: #2a7fff !important;
  font-weight: 700;
}

.about-link-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 6px;
  font-size: 0.84rem;
  line-height: 1.42;
}

.about-link-list li {
  list-style: none;
  margin-left: -18px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.about-link-list i {
  width: 16px;
  text-align: center;
  color: color-mix(in srgb, var(--rt-accent) 70%, var(--rt-text));
}

.about-quick-list {
  margin: 0;
  padding-left: 0;
  display: grid;
  gap: 7px;
}

.about-quick-list li {
  list-style: none;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 0.84rem;
  line-height: 1.45;
}

.about-quick-list i {
  margin-top: 2px;
  color: color-mix(in srgb, var(--rt-accent-2) 68%, var(--rt-text));
}

.about-skill-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 4px;
  font-size: 0.84rem;
}

.about-opportunities {
  margin-top: 8px;
  font-weight: 600;
}

.gr-button-primary {
  background: linear-gradient(
    92deg,
    color-mix(in srgb, var(--color-accent) 92%, #ffffff 8%) 0%,
    color-mix(in srgb, var(--rt-accent-2) 86%, var(--color-accent) 14%) 100%
  ) !important;
  border: none !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 18px color-mix(in srgb, var(--color-accent) 35%, transparent) !important;
}

.gr-button-secondary {
  background: linear-gradient(
    92deg,
    color-mix(in srgb, var(--rt-secondary) 82%, var(--color-accent-soft) 18%) 0%,
    color-mix(in srgb, var(--rt-secondary) 74%, var(--color-accent-soft) 26%) 100%
  ) !important;
  color: var(--rt-text) !important;
  border: 1px solid color-mix(in srgb, var(--rt-border) 78%, var(--color-accent) 22%) !important;
  font-weight: 700 !important;
}

.job-card {
  background: var(--rt-panel);
  border: 1px solid var(--rt-border);
  border-radius: 18px;
  padding: 16px;
  margin-bottom: 14px;
  box-shadow: var(--rt-shadow);
  position: relative;
  overflow: hidden;
}

.job-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(
    115deg,
    color-mix(in srgb, var(--rt-accent) 6%, transparent) 0%,
    transparent 38%,
    color-mix(in srgb, var(--rt-accent-2) 8%, transparent) 100%
  );
  pointer-events: none;
}

.job-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.job-title-block {
  display: flex;
  align-items: center;
  gap: 10px;
}

.job-title-block h3 {
  margin: 0;
  font-size: 1.17rem;
  line-height: 1.22;
  color: var(--rt-heading) !important;
}

.job-header-right {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.job-header-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 8px;
}

.header-action {
  display: inline-flex;
}

.job-rank {
  color: color-mix(in srgb, var(--rt-text) 82%, #4d87ff);
  font-weight: 800;
  font-size: 0.85rem;
  background: color-mix(in srgb, var(--rt-panel-soft) 88%, transparent);
  border: 1px solid var(--rt-border);
  border-radius: 999px;
  padding: 3px 10px;
}

.badge-pill {
  color: #ffffff !important;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 0.75rem;
  font-weight: 700;
  white-space: nowrap;
}

.apply-cta-btn,
.listing-cta-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  text-decoration: none !important;
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 0.77rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  border: 1px solid transparent;
}

.apply-cta-btn {
  background: linear-gradient(94deg, var(--rt-accent), var(--rt-accent-2));
  color: #ffffff !important;
  box-shadow: 0 8px 16px color-mix(in srgb, var(--rt-accent) 35%, transparent);
}

.apply-cta-btn:hover {
  filter: brightness(1.05);
}

.listing-cta-btn {
  background: color-mix(in srgb, var(--rt-panel-soft) 86%, transparent);
  border-color: color-mix(in srgb, var(--rt-border) 76%, var(--rt-accent) 24%);
  color: var(--rt-text) !important;
}

.job-meta,
.job-submeta {
  color: var(--rt-text-muted) !important;
  font-size: 0.9rem;
  margin-top: 6px;
  line-height: 1.42;
}

.job-submeta {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.job-submeta strong {
  color: var(--rt-text) !important;
}

.lang-note {
  color: var(--rt-text) !important;
  background: color-mix(in srgb, var(--rt-panel-soft) 90%, transparent);
  border: 1px solid var(--rt-border);
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 0.78rem;
}

.score-row {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.score-chip {
  border: 1.5px solid;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 0.8rem;
  font-weight: 700;
  background: color-mix(in srgb, var(--rt-panel-soft) 72%, transparent);
}

.score-bars {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 8px;
  margin-top: 10px;
}

.score-bar {
  background: var(--rt-panel-soft);
  border: 1px solid var(--rt-border);
  border-radius: 11px;
  padding: 7px 8px;
}

.score-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--rt-text) !important;
}

.score-bar-track {
  margin-top: 4px;
  height: 7px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--rt-panel-strong) 88%, transparent);
  overflow: hidden;
}

.score-bar-fill {
  height: 100%;
  border-radius: 999px;
}

.cap-note {
  margin-top: 10px;
  background: color-mix(in srgb, #ffecce 92%, transparent);
  border: 1px solid #ffd08c;
  color: #7a4800;
  border-radius: 10px;
  padding: 9px 10px;
  font-size: 0.84rem;
  font-weight: 600;
}

body.dark .gradio-container .cap-note,
.dark .gradio-container .cap-note,
[data-theme="dark"] .gradio-container .cap-note {
  color: #ffdcb0;
  background: rgba(118, 66, 0, 0.28);
  border-color: rgba(255, 174, 68, 0.42);
}

.reasoning {
  margin-top: 10px;
}

.reasoning-summary {
  margin: 0 0 9px;
  padding: 10px 12px;
  border-radius: 12px;
  background: linear-gradient(
    94deg,
    color-mix(in srgb, var(--rt-panel-soft) 86%, transparent),
    color-mix(in srgb, var(--rt-panel) 94%, var(--rt-accent) 6%)
  );
  border: 1px solid color-mix(in srgb, var(--rt-border) 72%, var(--rt-accent) 28%);
  font-weight: 700;
  color: var(--rt-text) !important;
}

.reasoning-columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}

.reasoning-col {
  background: color-mix(in srgb, var(--rt-panel-soft) 90%, transparent);
  border: 1px solid var(--rt-border);
  border-radius: 12px;
  padding: 10px;
}

.reasoning-col h4 {
  margin: 0 0 8px;
  font-size: 0.9rem;
  color: var(--rt-heading) !important;
}

.evidence-list {
  margin: 0;
  padding-left: 18px;
  font-size: 0.86rem;
  line-height: 1.43;
  color: var(--rt-text) !important;
}

.evidence-list li {
  margin-bottom: 5px;
}

.positive-list li::marker {
  color: #15a46d;
}

.risk-list li::marker {
  color: #e64b3c;
}

.formula-details,
.rule-debug-details {
  margin-top: 10px;
  border: 1px dashed color-mix(in srgb, var(--rt-border) 94%, transparent);
  border-radius: 10px;
  padding: 8px 10px;
  background: color-mix(in srgb, var(--rt-panel-soft) 88%, transparent);
}

.formula-details summary,
.rule-debug-details summary {
  cursor: pointer;
  font-weight: 700;
  color: var(--rt-heading) !important;
}

.formula-details p {
  margin: 8px 0 0;
  font-size: 0.84rem;
  color: var(--rt-text-muted) !important;
}

.formula-grid {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
}

.formula-item {
  border: 1px solid var(--rt-border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--rt-panel) 92%, transparent);
  padding: 9px 10px;
}

.formula-item h5 {
  margin: 0 0 5px;
  font-size: 0.84rem;
  color: var(--rt-heading) !important;
}

.formula-item p {
  margin: 0 0 5px;
}

.score-meaning-details {
  margin-top: 10px;
  border: 1px dashed color-mix(in srgb, var(--rt-border) 92%, transparent);
  border-radius: 10px;
  padding: 8px 10px;
  background: color-mix(in srgb, var(--rt-panel-soft) 90%, transparent);
}

.score-meaning-details summary {
  cursor: pointer;
  font-weight: 700;
  color: var(--rt-heading) !important;
}

.score-meaning-grid {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
}

.score-meaning-item {
  border: 1px solid var(--rt-border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--rt-panel) 93%, transparent);
  padding: 8px 10px;
}

.score-meaning-item h5 {
  margin: 0 0 5px;
  font-size: 0.84rem;
  color: var(--rt-heading) !important;
}

.score-meaning-item p {
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.45;
  color: var(--rt-text) !important;
}

.rule-debug-overview {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 0.82rem;
  color: var(--rt-text-muted) !important;
}

.rule-debug-grid {
  margin-top: 8px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 10px;
}

.rule-debug-col {
  background: var(--rt-panel);
  border: 1px solid var(--rt-border);
  border-radius: 10px;
  padding: 8px 10px;
}

.rule-debug-col h5 {
  margin: 0 0 6px;
  font-size: 0.84rem;
  color: var(--rt-heading) !important;
}

.rule-list {
  margin: 0 0 8px;
  padding-left: 18px;
  font-size: 0.8rem;
  line-height: 1.4;
  color: var(--rt-text) !important;
}

.rule-list li {
  margin-bottom: 4px;
}

.rule-list.rule-fired li::marker {
  color: #12a468;
}

.rule-list.rule-not-fired li::marker {
  color: #d58509;
}

.listing-details {
  margin-top: 10px;
  border: 1px dashed color-mix(in srgb, var(--rt-border) 90%, transparent);
  border-radius: 12px;
  padding: 8px 10px;
  background: color-mix(in srgb, var(--rt-panel-soft) 86%, transparent);
}

.listing-details summary {
  cursor: pointer;
  font-weight: 800;
  color: var(--rt-heading) !important;
}

.listing-note-list {
  margin: 8px 0 4px;
  padding-left: 18px;
  font-size: 0.82rem;
  line-height: 1.45;
  color: var(--rt-text-muted) !important;
}

.listing-note-list li {
  margin-bottom: 4px;
}

.listing-note-list a {
  color: #2d7eff !important;
  word-break: break-all;
}

.listing-section {
  border: 1px solid var(--rt-border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--rt-panel) 93%, transparent);
  padding: 9px 10px;
  margin-top: 8px;
}

.listing-section h5 {
  margin: 0 0 4px;
  font-size: 0.84rem;
  color: var(--rt-heading) !important;
}

.listing-section p {
  margin: 0;
  font-size: 0.83rem;
  line-height: 1.45;
  color: var(--rt-text) !important;
}

.listing-bullets {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 5px;
}

.listing-bullets li {
  font-size: 0.83rem;
  line-height: 1.5;
  max-width: 72ch;
  text-wrap: pretty;
}

.board-gap {
  height: 12px;
}

.search-hint {
  margin: 4px 0 8px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid color-mix(in srgb, var(--rt-border) 78%, var(--rt-accent) 22%);
  background: linear-gradient(
    94deg,
    color-mix(in srgb, var(--rt-panel-soft) 86%, transparent),
    color-mix(in srgb, var(--rt-panel) 94%, var(--rt-accent) 6%)
  );
  font-size: 0.84rem;
  line-height: 1.45;
  color: var(--rt-text) !important;
}

.empty-state {
  background: var(--rt-panel);
  border: 1px dashed var(--rt-border);
  border-radius: 14px;
  padding: 18px;
  color: var(--rt-text-muted) !important;
  text-align: center;
}

/* Keep dataframe headers readable in narrow layouts/scroll containers. */
#overview-table-live table,
#overview-table-negative table,
#overview-table-recommendations table {
  table-layout: auto !important;
}

#overview-table-live th,
#overview-table-negative th,
#overview-table-recommendations th {
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  text-overflow: ellipsis;
}

#overview-table-live td,
#overview-table-negative td,
#overview-table-recommendations td {
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}

@media (max-width: 960px) {
  .gradio-container {
    padding: 10px;
  }
  .landing-shell {
    min-height: auto;
    padding: 16px;
  }
  .landing-top-grid {
    grid-template-columns: 1fr;
  }
  .landing-copy {
    text-align: center;
  }
  .landing-pill-row {
    justify-content: center;
  }
  .landing-weight-grid {
    grid-template-columns: 1fr;
  }
  .landing-visual-panel {
    width: 100%;
  }
  .signal-orbit {
    width: 180px;
    height: 180px;
  }
  .signal-core .landing-logo-wrap {
    width: 74px;
    height: 74px;
  }
  .visual-kpi-grid {
    grid-template-columns: 1fr;
  }
  .landing-credit {
    flex-direction: column;
    text-align: center;
  }
  .landing-credit-logo {
    width: 52px;
    height: 52px;
  }
  .hero-shell,
  .about-card {
    grid-template-columns: 1fr;
  }
  .about-mindmesh {
    width: 100%;
    justify-content: center;
  }
  .landing-logo-wrap {
    width: 140px;
    height: 140px;
  }
  .hero-logo-wrap {
    width: 130px;
    height: 130px;
  }
  .about-photo-wrap {
    width: 100px;
    height: 100px;
  }
  .job-card {
    padding: 13px;
  }
  .job-card-header {
    flex-direction: column;
    gap: 10px;
  }
  .job-header-right {
    width: 100%;
    align-items: flex-start;
    justify-content: flex-start;
  }
  .job-header-actions {
    width: 100%;
    justify-content: flex-start;
  }
  .apply-cta-btn,
  .listing-cta-btn {
    width: 100%;
  }
  .score-bars {
    grid-template-columns: 1fr;
  }
}
"""


GRADIO_MAJOR_VERSION = int(gr.__version__.split(".")[0]) if gr.__version__ else 0
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
)
_blocks_kwargs = {"title": "RemoteTrust Job Board"}
_launch_kwargs: dict[str, Any] = {}
if GRADIO_MAJOR_VERSION >= 6:
    _launch_kwargs["css"] = CSS
    _launch_kwargs["theme"] = THEME
else:
    _blocks_kwargs["css"] = CSS
    _blocks_kwargs["theme"] = THEME

if os.environ.get("GRADIO_SHARE", "1").strip().lower() in {"1", "true", "yes"}:
    _launch_kwargs["share"] = True

# Bind to all interfaces so devices on the same LAN can access the app
# via http://<your-local-ip>:7860.
_launch_kwargs["server_name"] = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0").strip() or "0.0.0.0"


with gr.Blocks(**_blocks_kwargs) as demo:
    with gr.Tabs(elem_id="rt-tabs"):
        with gr.Tab("Home"):
            gr.HTML(_build_landing_html(DETECTOR_STATUS))

        with gr.Tab("Live Job Board"):
            jobs_state = gr.State([])
            filtered_state = gr.State([])
            page_state = gr.State(1)

            with gr.Row():
                per_source = gr.Slider(
                    minimum=3,
                    maximum=100,
                    value=5,
                    step=1,
                    label="Jobs Per Source",
                    info="Pull this many jobs from each of the 4 feeds.",
                )
                page_size = gr.Dropdown(
                    choices=[4, 6, 8, 10],
                    value=DEFAULT_PAGE_SIZE,
                    label="Cards Per Page",
                )
            with gr.Row():
                source_filter = gr.Dropdown(
                    choices=SOURCE_OPTIONS,
                    value=[value for _, value in SOURCE_OPTIONS],
                    multiselect=True,
                    label="Sources",
                    info="Pick one or more sources to include.",
                )
                badge_filter = gr.Dropdown(
                    choices=["All", "Trusted Pick", "Promising", "Verify First", "Avoid"],
                    value="All",
                    label="Badge Filter",
                )
                risk_filter = gr.Dropdown(
                    choices=RISK_FILTER_OPTIONS,
                    value="All",
                    label="Risk Filter",
                )
                sort_by = gr.Dropdown(
                    choices=SORT_OPTIONS,
                    value="Final Score (high to low)",
                    label="Sort By",
                )

            refresh_btn = gr.Button("Refresh Live Jobs", variant="primary", size="lg")
            status_md = gr.Markdown("Loading live postings...")
            with gr.Accordion("Detailed table view", open=False):
                overview_df = gr.Dataframe(
                    label="Filtered Jobs Overview",
                    interactive=False,
                    wrap=False,
                    elem_id="overview-table-live",
                )
            csv_download = gr.DownloadButton(
                label="Download Filtered Table (CSV)",
                value=None,
                variant="secondary",
            )
            gr.HTML("<div class='board-gap'></div>")
            gr.HTML(
                "<div class='search-hint'>"
                "<strong>Filter Pulled Jobs:</strong> Search and filters below work on jobs already pulled into the board. "
                "Use <strong>Refresh Live Jobs</strong> whenever you want newly fetched posts."
                "</div>"
            )
            query = gr.Textbox(
                label="Search Pulled Jobs",
                placeholder="Search title, company, location, description, or requirements in pulled jobs...",
            )
            page_info = gr.Markdown("Page 0 / 0")
            cards_html = gr.HTML("<div class='empty-state'>No jobs loaded yet.</div>")

            with gr.Row():
                prev_btn = gr.Button("Previous Page", variant="secondary")
                next_btn = gr.Button("Next Page", variant="secondary")

            refresh_btn.click(
                fn=refresh_dashboard,
                inputs=[per_source, query, source_filter, badge_filter, risk_filter, sort_by, page_size],
                outputs=[jobs_state, filtered_state, page_state, status_md, overview_df, page_info, cards_html, csv_download],
            )

            for filter_component in [query, source_filter, badge_filter, risk_filter, sort_by, page_size]:
                filter_component.change(
                    fn=refilter_dashboard,
                    inputs=[jobs_state, query, source_filter, badge_filter, risk_filter, sort_by, page_size],
                    outputs=[filtered_state, page_state, status_md, overview_df, page_info, cards_html, csv_download],
                )

            prev_btn.click(
                fn=go_prev_page,
                inputs=[filtered_state, page_state, page_size],
                outputs=[page_state, page_info, cards_html],
            )
            next_btn.click(
                fn=go_next_page,
                inputs=[filtered_state, page_state, page_size],
                outputs=[page_state, page_info, cards_html],
            )

            demo.load(
                fn=refresh_dashboard,
                inputs=[per_source, query, source_filter, badge_filter, risk_filter, sort_by, page_size],
                outputs=[jobs_state, filtered_state, page_state, status_md, overview_df, page_info, cards_html, csv_download],
            )

        with gr.Tab("CV Recommendations"):
            rec_jobs_state = gr.State([])
            rec_filtered_state = gr.State([])
            rec_page_state = gr.State(1)

            cv_text = gr.Textbox(
                label="Paste CV / Resume Text",
                placeholder="Paste your CV text here. We'll rank live jobs by semantic similarity + trust threshold.",
                lines=12,
            )
            with gr.Row():
                rec_per_source = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=25,
                    step=1,
                    label="Jobs Per Source",
                )
                rec_similarity_threshold = gr.Slider(
                    minimum=0.02,
                    maximum=0.85,
                    value=0.10,
                    step=0.01,
                    label="Minimum CV Similarity",
                )
                rec_min_trust = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=60,
                    step=1,
                    label="Minimum Trust Score",
                )
                rec_max_results = gr.Slider(
                    minimum=5,
                    maximum=120,
                    value=40,
                    step=1,
                    label="Max Recommendations",
                )
            with gr.Row():
                rec_page_size = gr.Dropdown(
                    choices=[4, 6, 8, 10],
                    value=DEFAULT_PAGE_SIZE,
                    label="Cards Per Page",
                )
                rec_source_filter = gr.Dropdown(
                    choices=RECOMMENDATION_SOURCE_OPTIONS,
                    value=[value for _, value in RECOMMENDATION_SOURCE_OPTIONS],
                    multiselect=True,
                    label="Sources",
                )

            rec_run_btn = gr.Button("Generate Recommendations", variant="primary", size="lg")
            rec_status_md = gr.Markdown("Paste CV text and click Generate Recommendations.")
            with gr.Accordion("Detailed table view", open=False):
                rec_overview_df = gr.Dataframe(
                    label="Recommended Jobs Overview",
                    interactive=False,
                    wrap=False,
                    elem_id="overview-table-recommendations",
                )
            rec_csv_download = gr.DownloadButton(
                label="Download Recommendations (CSV)",
                value=None,
                variant="secondary",
            )
            gr.HTML("<div class='board-gap'></div>")
            rec_query = gr.Textbox(
                label="Search Recommended Jobs",
                placeholder="Filter the recommended set by title, company, location, description...",
            )
            rec_page_info = gr.Markdown("Page 0 / 0")
            rec_cards_html = gr.HTML("<div class='empty-state'>No recommendations yet.</div>")

            with gr.Row():
                rec_prev_btn = gr.Button("Previous Page", variant="secondary")
                rec_next_btn = gr.Button("Next Page", variant="secondary")

            rec_run_btn.click(
                fn=refresh_recommendations,
                inputs=[
                    rec_per_source,
                    rec_source_filter,
                    cv_text,
                    rec_similarity_threshold,
                    rec_min_trust,
                    rec_max_results,
                    rec_query,
                    rec_page_size,
                ],
                outputs=[
                    rec_jobs_state,
                    rec_filtered_state,
                    rec_page_state,
                    rec_status_md,
                    rec_overview_df,
                    rec_page_info,
                    rec_cards_html,
                    rec_csv_download,
                ],
            )

            rec_query.change(
                fn=refilter_recommendations,
                inputs=[rec_jobs_state, rec_query, rec_page_size],
                outputs=[
                    rec_filtered_state,
                    rec_page_state,
                    rec_status_md,
                    rec_overview_df,
                    rec_page_info,
                    rec_cards_html,
                    rec_csv_download,
                ],
            )

            rec_page_size.change(
                fn=refilter_recommendations,
                inputs=[rec_jobs_state, rec_query, rec_page_size],
                outputs=[
                    rec_filtered_state,
                    rec_page_state,
                    rec_status_md,
                    rec_overview_df,
                    rec_page_info,
                    rec_cards_html,
                    rec_csv_download,
                ],
            )

            rec_prev_btn.click(
                fn=go_prev_page,
                inputs=[rec_filtered_state, rec_page_state, rec_page_size],
                outputs=[rec_page_state, rec_page_info, rec_cards_html],
            )
            rec_next_btn.click(
                fn=go_next_page,
                inputs=[rec_filtered_state, rec_page_state, rec_page_size],
                outputs=[rec_page_state, rec_page_info, rec_cards_html],
            )

        with gr.Tab("Negative Samples"):
            neg_jobs_state = gr.State([])
            neg_filtered_state = gr.State([])
            neg_page_state = gr.State(1)

            with gr.Row():
                neg_page_size = gr.Dropdown(
                    choices=[4, 6, 8, 10],
                    value=DEFAULT_PAGE_SIZE,
                    label="Cards Per Page",
                )
                neg_source_filter = gr.Dropdown(
                    choices=NEGATIVE_SOURCE_OPTIONS,
                    value=[value for _, value in NEGATIVE_SOURCE_OPTIONS],
                    multiselect=True,
                    label="Sample Groups",
                )
                neg_badge_filter = gr.Dropdown(
                    choices=["All", "Trusted Pick", "Promising", "Verify First", "Avoid"],
                    value="All",
                    label="Badge Filter",
                )
                neg_risk_filter = gr.Dropdown(
                    choices=RISK_FILTER_OPTIONS,
                    value="All",
                    label="Risk Filter",
                )
                neg_sort_by = gr.Dropdown(
                    choices=SORT_OPTIONS,
                    value="Final Score (high to low)",
                    label="Sort By",
                )

            run_negative_btn = gr.Button("Run Synthetic Scam Benchmark", variant="primary", size="lg")
            neg_status_md = gr.Markdown("Loading synthetic scam samples...")
            with gr.Accordion("Detailed table view", open=False):
                neg_overview_df = gr.Dataframe(
                    label="Filtered Synthetic Samples Overview",
                    interactive=False,
                    wrap=False,
                    elem_id="overview-table-negative",
                )
            neg_csv_download = gr.DownloadButton(
                label="Download Filtered Table (CSV)",
                value=None,
                variant="secondary",
            )
            gr.HTML("<div class='board-gap'></div>")
            gr.HTML(
                "<div class='search-hint'>"
                "<strong>Synthetic validation mode:</strong> This tab uses intentionally fake postings (crude and sophisticated) "
                "to stress-test trust caps, risk flags, and explanation quality."
                "</div>"
            )
            neg_query = gr.Textbox(
                label="Search Synthetic Samples",
                placeholder="Search by title, company, location, description, or requirements...",
            )
            neg_page_info = gr.Markdown("Page 0 / 0")
            neg_cards_html = gr.HTML("<div class='empty-state'>No samples loaded yet.</div>")

            with gr.Row():
                neg_prev_btn = gr.Button("Previous Page", variant="secondary")
                neg_next_btn = gr.Button("Next Page", variant="secondary")

            run_negative_btn.click(
                fn=refresh_negative_samples,
                inputs=[neg_query, neg_source_filter, neg_badge_filter, neg_risk_filter, neg_sort_by, neg_page_size],
                outputs=[
                    neg_jobs_state,
                    neg_filtered_state,
                    neg_page_state,
                    neg_status_md,
                    neg_overview_df,
                    neg_page_info,
                    neg_cards_html,
                    neg_csv_download,
                ],
            )

            for neg_filter_component in [
                neg_query,
                neg_source_filter,
                neg_badge_filter,
                neg_risk_filter,
                neg_sort_by,
                neg_page_size,
            ]:
                neg_filter_component.change(
                    fn=refilter_dashboard,
                    inputs=[
                        neg_jobs_state,
                        neg_query,
                        neg_source_filter,
                        neg_badge_filter,
                        neg_risk_filter,
                        neg_sort_by,
                        neg_page_size,
                    ],
                    outputs=[
                        neg_filtered_state,
                        neg_page_state,
                        neg_status_md,
                        neg_overview_df,
                        neg_page_info,
                        neg_cards_html,
                        neg_csv_download,
                    ],
                )

            neg_prev_btn.click(
                fn=go_prev_page,
                inputs=[neg_filtered_state, neg_page_state, neg_page_size],
                outputs=[neg_page_state, neg_page_info, neg_cards_html],
            )
            neg_next_btn.click(
                fn=go_next_page,
                inputs=[neg_filtered_state, neg_page_state, neg_page_size],
                outputs=[neg_page_state, neg_page_info, neg_cards_html],
            )

            demo.load(
                fn=refresh_negative_samples,
                inputs=[neg_query, neg_source_filter, neg_badge_filter, neg_risk_filter, neg_sort_by, neg_page_size],
                outputs=[
                    neg_jobs_state,
                    neg_filtered_state,
                    neg_page_state,
                    neg_status_md,
                    neg_overview_df,
                    neg_page_info,
                    neg_cards_html,
                    neg_csv_download,
                ],
            )

        with gr.Tab("About"):
            gr.HTML(_build_about_html())


if __name__ == "__main__":
    demo.queue().launch(**_launch_kwargs)
