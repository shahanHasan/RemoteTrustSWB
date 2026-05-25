"""Project constants for EMSCAD fraud detection."""

import os

TARGET_COLUMN = "fraudulent"
ID_COLUMN = "job_id"

TEXT_COLUMNS = [
    "title",
    "location",
    "department",
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

NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

MODEL_FEATURE_COLUMNS = [*TEXT_COLUMNS, *NUMERIC_COLUMNS]
MODEL_INPUT_COLUMNS = ["combined_text", *NUMERIC_COLUMNS]

CLASS_NAMES = ["legitimate", "fraudulent"]

DEFAULT_C_GRID = [0.5, 1.0, 2.0]
USAJOBS_API_KEY = os.environ.get(
    "USAJOBS_API_KEY",
    "IdDUFcfqs9Mr4/Pjjwy+UcCEtbYkaqT3ZX5r9mi9ET8=",
)
# USAJOBS requires this to be the email used when requesting the API key.
USAJOBS_USER_AGENT = os.environ.get("USAJOBS_USER_AGENT", "shahan.hasan101294@gmail.com")
