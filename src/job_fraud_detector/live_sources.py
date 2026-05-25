"""Fetch and score live remote jobs from public sources."""

from __future__ import annotations

import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import pandas as pd

from .constants import MODEL_FEATURE_COLUMNS, USAJOBS_API_KEY, USAJOBS_USER_AGENT
from .features import prepare_features
from .i18n import normalize_posting_language
from .rules import evaluate_job_posting

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SOURCES = {
    "we_work_remotely_rss": "https://weworkremotely.com/remote-jobs.rss",
    "jobicy_api": "https://jobicy.com/api/v2/remote-jobs?count=100",
    "remotive_api_docs": "https://remotive.com/remote-jobs/api",
    "usajobs_api": "https://data.usajobs.gov/api/search",
}

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if item)
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def _to_comma_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        preferred = _clean_text(value.get("Name", "")) or _clean_text(value.get("name", ""))
        if preferred:
            return preferred
        parts = [_clean_text(v) for v in value.values()]
        parts = [part for part in parts if part]
        return ", ".join(parts)
    if isinstance(value, list):
        return ", ".join(_clean_text(v) for v in value if _clean_text(v))
    return _clean_text(value)


def _build_location(*parts: Any) -> str:
    cleaned = [_clean_text(part) for part in parts]
    cleaned = [part for part in cleaned if part]
    return ", ".join(cleaned)


def _request_url(
    url: str,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[bytes, str]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }
    if extra_headers:
        headers.update(dict(extra_headers))

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type", "")
    return content, content_type


def _extract_wwr_company_title(raw_title: str) -> tuple[str, str]:
    title = _clean_text(raw_title)
    if ":" not in title:
        return "", title
    company, role = title.split(":", 1)
    company = company.strip()
    role = role.strip()
    if not company or not role:
        return "", title
    return company, role


def _parse_we_work_remotely_rss(content: bytes, limit: int) -> list[dict[str, Any]]:
    root = ET.fromstring(content)
    items = root.findall("./channel/item")

    parsed: list[dict[str, Any]] = []
    for item in items[:limit]:
        company_name, title = _extract_wwr_company_title(item.findtext("title"))
        location = _build_location(
            item.findtext("region"),
            item.findtext("state"),
            item.findtext("country"),
        )

        logo_url = ""
        for child in item:
            if child.tag.endswith("content"):
                logo_url = _clean_text(child.attrib.get("url", ""))
                break

        link = _clean_text(item.findtext("link"))
        parsed.append(
            {
                "source": "we_work_remotely_rss",
                "job_url": link,
                "apply_url": link,
                "posted_date": _clean_text(item.findtext("pubDate")),
                "title": title,
                "location": location,
                "department": "",
                "salary_range": "",
                "company_profile": company_name,
                "description": _clean_text(item.findtext("description")),
                "requirements": _clean_text(item.findtext("skills")),
                "benefits": "",
                "telecommuting": 1,
                "has_company_logo": int(bool(logo_url)),
                "has_questions": 0,
                "employment_type": _clean_text(item.findtext("type")),
                "required_experience": "",
                "required_education": "",
                "industry": _clean_text(item.findtext("category")),
                "function": "",
            }
        )
    return parsed


def _parse_jobicy(content: bytes, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(content.decode("utf-8"))
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []

    parsed: list[dict[str, Any]] = []
    for job in jobs[:limit]:
        if not isinstance(job, Mapping):
            continue
        logo_url = _clean_text(job.get("companyLogo", ""))
        link = _clean_text(job.get("url", ""))
        parsed.append(
            {
                "source": "jobicy_api",
                "job_url": link,
                "apply_url": link,
                "posted_date": _clean_text(job.get("pubDate", "")),
                "title": _clean_text(job.get("jobTitle", "")),
                "location": _clean_text(job.get("jobGeo", "")),
                "department": "",
                "salary_range": _clean_text(job.get("annualSalaryMin", "")),
                "company_profile": _clean_text(job.get("companyName", "")),
                "description": _clean_text(job.get("jobDescription", "")),
                "requirements": _clean_text(job.get("jobExcerpt", "")),
                "benefits": "",
                "telecommuting": 1,
                "has_company_logo": int(bool(logo_url)),
                "has_questions": 0,
                "employment_type": _to_comma_list(job.get("jobType", "")),
                "required_experience": _to_comma_list(job.get("jobLevel", "")),
                "required_education": "",
                "industry": _to_comma_list(job.get("jobIndustry", "")),
                "function": "",
            }
        )
    return parsed


def _parse_remotive(content: bytes, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(content.decode("utf-8"))
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []

    parsed: list[dict[str, Any]] = []
    for job in jobs[:limit]:
        if not isinstance(job, Mapping):
            continue
        logo_url = _clean_text(job.get("company_logo", ""))
        link = _clean_text(job.get("url", ""))
        parsed.append(
            {
                "source": "remotive_api",
                "job_url": link,
                "apply_url": link,
                "posted_date": _clean_text(job.get("publication_date", "")),
                "title": _clean_text(job.get("title", "")),
                "location": _clean_text(job.get("candidate_required_location", "")),
                "department": "",
                "salary_range": _clean_text(job.get("salary", "")),
                "company_profile": _clean_text(job.get("company_name", "")),
                "description": _clean_text(job.get("description", "")),
                "requirements": _to_comma_list(job.get("tags", "")),
                "benefits": "",
                "telecommuting": 1,
                "has_company_logo": int(bool(logo_url)),
                "has_questions": 0,
                "employment_type": _clean_text(job.get("job_type", "")),
                "required_experience": "",
                "required_education": "",
                "industry": _clean_text(job.get("category", "")),
                "function": "",
            }
        )
    return parsed


def _salary_from_usajobs(details: Mapping[str, Any]) -> str:
    remunerations = details.get("PositionRemuneration", [])
    if not isinstance(remunerations, list) or not remunerations:
        return ""

    first = remunerations[0]
    if not isinstance(first, Mapping):
        return ""

    minimum = _clean_text(first.get("MinimumRange", ""))
    maximum = _clean_text(first.get("MaximumRange", ""))
    rate = _clean_text(first.get("RateIntervalCode", ""))

    if minimum and maximum:
        return f"${minimum} - ${maximum} {rate}".strip()
    if minimum:
        return f"${minimum} {rate}".strip()
    return ""


def _extract_usajobs_description(details: Mapping[str, Any]) -> tuple[str, str, str]:
    description = _clean_text(details.get("JobSummary", ""))
    duties = _clean_text(details.get("MajorDuties", ""))
    qualification = _clean_text(details.get("QualificationSummary", ""))
    requirements = _clean_text(details.get("Education", ""))

    if duties:
        description = f"{description}\n\nDuties: {duties}".strip()
    if qualification:
        requirements = f"{requirements}\n\nQualification summary: {qualification}".strip()

    benefits = _clean_text(details.get("Benefits", ""))
    return description, requirements, benefits


def _parse_usajobs(content: bytes, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(content.decode("utf-8"))
    search_result = payload.get("SearchResult", {}) if isinstance(payload, Mapping) else {}
    items = search_result.get("SearchResultItems", []) if isinstance(search_result, Mapping) else []

    parsed: list[dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, Mapping):
            continue

        descriptor = item.get("MatchedObjectDescriptor", {})
        if not isinstance(descriptor, Mapping):
            continue

        apply_uris = descriptor.get("ApplyURI", [])
        apply_url = ""
        if isinstance(apply_uris, list) and apply_uris:
            apply_url = _clean_text(apply_uris[0])

        details = descriptor.get("UserArea", {})
        if isinstance(details, Mapping):
            details = details.get("Details", {}) if isinstance(details.get("Details", {}), Mapping) else details
        if not isinstance(details, Mapping):
            details = {}

        description, requirements, benefits = _extract_usajobs_description(details)
        employment_type = _to_comma_list(descriptor.get("PositionSchedule", ""))
        if not employment_type:
            employment_type = _to_comma_list(details.get("PositionSchedule", ""))

        posted_date = _clean_text(descriptor.get("PublicationStartDate", ""))
        if not posted_date:
            posted_date = _clean_text(details.get("PublicationStartDate", ""))

        job_url = _clean_text(descriptor.get("PositionURI", ""))

        parsed.append(
            {
                "source": "usajobs_api",
                "job_url": job_url,
                "apply_url": apply_url or job_url,
                "posted_date": posted_date,
                "title": _clean_text(descriptor.get("PositionTitle", "")),
                "location": _clean_text(descriptor.get("PositionLocationDisplay", "")),
                "department": _clean_text(descriptor.get("DepartmentName", "")),
                "salary_range": _salary_from_usajobs(details),
                "company_profile": _clean_text(descriptor.get("OrganizationName", "")),
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
                "telecommuting": 1,
                "has_company_logo": 1,
                "has_questions": 0,
                "employment_type": employment_type,
                "required_experience": _clean_text(details.get("LowGrade", "")),
                "required_education": _clean_text(details.get("EducationalRequirements", "")),
                "industry": "Government",
                "function": "Public Service",
            }
        )

    return parsed


def _resolve_url(source_name: str, source_url: str) -> str:
    # The provided Remotive docs URL points to docs, not JSON feed.
    if source_name == "remotive_api_docs":
        return "https://remotive.com/api/remote-jobs"
    return source_url


def _fetch_one_source_jobs(
    source_name: str,
    source_url: str,
    per_source: int,
    timeout: int,
    user_agent: str,
) -> list[dict[str, Any]]:
    if source_name == "usajobs_api":
        return _fetch_usajobs_remote_jobs(
            source_url=source_url,
            per_source=per_source,
            timeout=timeout,
        )

    resolved_url = _resolve_url(source_name, source_url)
    content, _content_type = _request_url(
        resolved_url,
        timeout=timeout,
        user_agent=user_agent,
    )

    if source_name == "we_work_remotely_rss":
        return _parse_we_work_remotely_rss(content, limit=per_source)
    if source_name == "jobicy_api":
        return _parse_jobicy(content, limit=per_source)
    if source_name in {"remotive_api_docs", "remotive_api"}:
        return _parse_remotive(content, limit=per_source)
    raise ValueError(
        f"Unsupported source '{source_name}'. Known sources: {list(SOURCES.keys())}"
    )


def _fetch_usajobs_remote_jobs(
    source_url: str,
    per_source: int,
    timeout: int,
) -> list[dict[str, Any]]:
    if not USAJOBS_API_KEY:
        raise ValueError("USAJOBS_API_KEY is missing; cannot fetch USAJOBS jobs.")

    params = {
        "RemoteIndicator": "True",
        "ResultsPerPage": str(max(per_source, 25)),
        "Page": "1",
        "WhoMayApply": "public",
        "SortField": "openingdate",
        "SortDirection": "Desc",
        "Fields": "full",
    }
    request_url = f"{source_url}?{urllib.parse.urlencode(params)}"
    content, _content_type = _request_url(
        request_url,
        timeout=timeout,
        user_agent=USAJOBS_USER_AGENT,
        extra_headers={
            "Host": "data.usajobs.gov",
            "Authorization-Key": USAJOBS_API_KEY,
        },
    )
    return _parse_usajobs(content, limit=per_source)


def fetch_jobs_from_sources(
    sources: Mapping[str, str] | None = None,
    per_source: int = 5,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    fail_fast: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch and normalize remote jobs from known sources."""
    if per_source < 1:
        raise ValueError("per_source must be >= 1")

    sources_to_use = list(dict(sources or SOURCES).items())
    collected: list[dict[str, Any]] = []

    # For large pulls, source calls run concurrently to reduce end-to-end latency.
    if parallel and len(sources_to_use) > 1:
        workers = max_workers or min(4, len(sources_to_use))
        jobs_by_index: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _fetch_one_source_jobs,
                    source_name,
                    source_url,
                    per_source,
                    timeout,
                    user_agent,
                ): idx
                for idx, (source_name, source_url) in enumerate(sources_to_use)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    jobs_by_index[idx] = future.result()
                except Exception:
                    if fail_fast:
                        raise
        for idx in range(len(sources_to_use)):
            collected.extend(jobs_by_index.get(idx, []))
        return collected

    for source_name, source_url in sources_to_use:
        try:
            jobs = _fetch_one_source_jobs(
                source_name=source_name,
                source_url=source_url,
                per_source=per_source,
                timeout=timeout,
                user_agent=user_agent,
            )
            collected.extend(jobs)
        except Exception:
            if fail_fast:
                raise

    return collected


def model_payload_from_live_job(live_job: Mapping[str, Any]) -> dict[str, Any]:
    """Create a model-compatible posting payload from one normalized live job."""
    payload: dict[str, Any] = {}
    for column in MODEL_FEATURE_COLUMNS:
        if column in {"telecommuting", "has_company_logo", "has_questions"}:
            payload[column] = int(live_job.get(column, 0) or 0)
        else:
            payload[column] = _clean_text(live_job.get(column, ""))
    return payload


def score_live_jobs(
    detector: Any,
    live_jobs: list[dict[str, Any]],
    with_explanations: bool = False,
    with_heuristics: bool = True,
    num_features: int = 10,
    num_samples: int = 1500,
    enable_i18n: bool = True,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    """Score normalized live jobs with ML + optional heuristic trust/quality layer."""
    if not live_jobs:
        return []

    if batch_size < 1:
        batch_size = 1

    scored: list[dict[str, Any]] = []
    normalized_items: list[dict[str, Any]] = []

    for job in live_jobs:
        i18n = normalize_posting_language(job, enable_translation=enable_i18n)
        scoring_job = i18n.posting
        normalized_items.append(
            {
                "job": job,
                "scoring_job": scoring_job,
                "i18n": i18n,
                "payload": model_payload_from_live_job(scoring_job),
            }
        )

    # Prefer batched inference when explanations are off and detector supports dataframe mode.
    batch_predictions: list[dict[str, Any]] = []
    can_batch_predict = (
        not with_explanations
        and hasattr(detector, "predict_dataframe")
        and callable(getattr(detector, "predict_dataframe"))
    )
    if can_batch_predict:
        for start in range(0, len(normalized_items), batch_size):
            chunk = normalized_items[start:start + batch_size]
            chunk_payload_df = pd.DataFrame([item["payload"] for item in chunk])
            prepared = prepare_features(chunk_payload_df, include_target=False)
            chunk_predictions = detector.predict_dataframe(prepared)
            for _, pred_row in chunk_predictions.iterrows():
                batch_predictions.append(
                    {
                        "fraud_probability": float(pred_row["fraud_probability"]),
                        "prediction": int(pred_row["prediction"]),
                        "threshold": float(getattr(detector, "threshold", 0.5)),
                    }
                )

    for idx, item in enumerate(normalized_items):
        job = item["job"]
        scoring_job = item["scoring_job"]
        i18n = item["i18n"]

        if can_batch_predict:
            result = batch_predictions[idx]
        else:
            result = detector.predict_posting(
                item["payload"],
                with_explanation=with_explanations,
                num_features=num_features,
                num_samples=num_samples,
            )

        row = {
            "source": job.get("source", ""),
            "job_url": job.get("job_url", ""),
            "apply_url": job.get("apply_url", "") or job.get("job_url", ""),
            "posted_date": job.get("posted_date", ""),
            "title": scoring_job.get("title", ""),
            "company_profile": scoring_job.get("company_profile", ""),
            "location": scoring_job.get("location", ""),
            "department": scoring_job.get("department", ""),
            "description": scoring_job.get("description", ""),
            "requirements": scoring_job.get("requirements", ""),
            "benefits": scoring_job.get("benefits", ""),
            "salary_range": scoring_job.get("salary_range", ""),
            "employment_type": scoring_job.get("employment_type", ""),
            "required_experience": scoring_job.get("required_experience", ""),
            "required_education": scoring_job.get("required_education", ""),
            "industry": scoring_job.get("industry", ""),
            "function": scoring_job.get("function", ""),
            "telecommuting": int(job.get("telecommuting", 0) or 0),
            "has_company_logo": int(job.get("has_company_logo", 0) or 0),
            "has_questions": int(job.get("has_questions", 0) or 0),
            "fraud_probability": float(result["fraud_probability"]),
            "prediction": int(result["prediction"]),
            "threshold": float(result["threshold"]),
            "detected_language": i18n.language,
            "language_confidence": round(i18n.language_confidence, 4),
            "language_detector": i18n.language_detector,
            "translation_applied": bool(i18n.translation_applied),
            "translation_provider": i18n.translation_provider,
            "translation_error": i18n.translation_error,
            "original_title": job.get("title", ""),
            "original_description": job.get("description", ""),
            "original_requirements": job.get("requirements", ""),
        }

        if with_heuristics:
            heuristic = evaluate_job_posting(
                posting=scoring_job,
                ml_fraud_probability=float(result["fraud_probability"]),
            )
            row.update(heuristic)

        if "lime_explanation" in result:
            row["lime_explanation"] = result["lime_explanation"]
        scored.append(row)

    return scored
