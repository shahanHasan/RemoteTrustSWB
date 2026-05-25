"""Lightweight EMSCAD fraud detection package."""

from .features import prepare_single_posting
from .inference import FraudDetector
from .live_sources import SOURCES, fetch_jobs_from_sources, score_live_jobs
from .rules import evaluate_job_posting

__all__ = [
    "FraudDetector",
    "prepare_single_posting",
    "SOURCES",
    "fetch_jobs_from_sources",
    "score_live_jobs",
    "evaluate_job_posting",
]
