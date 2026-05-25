# RemoteTrust Comprehensive Presentation Report

Prepared for: **Scale Without Borders Hackathon 2026 Submission**  
Team: **VectorSpace**  
Date: **May 24, 2026**  
Project: **RemoteTrust - Trust-ranked remote job intelligence with transparent fraud and quality scoring**

---

## 1) Executive Summary

RemoteTrust is a practical, explainable remote-job intelligence system that:

1. Aggregates live remote postings from public, documented sources.
2. Scores each posting with a dual-score system:
   - **Trust Score** (legitimacy/safety likelihood)
   - **Quality Score** (job-post completeness/utility)
3. Produces a **Final Opportunity Score** and action-oriented badge.
4. Explains results using both:
   - **ML probability + thresholding**
   - **Rules-first reasoning with visible fired/not-fired checks**
   - **LIME local text explanations (artifact + optional inference path)**

This design directly addresses job-seeker safety and usefulness, not only binary fraud detection.

---

## 2) Problem Framing and Product Positioning

### Core problem
Job seekers need to answer two distinct questions:

1. Is this posting likely legitimate and safe?
2. Even if legitimate, is this posting high quality and worth pursuing?

### Product stance
RemoteTrust intentionally separates these with a trust-first guardrail:

- A polished but suspicious post should not rank highly.
- A legitimate but incomplete post should not be over-recommended.

---

## 3) Implemented System Architecture (Code-Accurate)

### 3.1 Data ingestion layer
Live sources in code (`src/job_fraud_detector/live_sources.py`):

- We Work Remotely RSS
- Jobicy API
- Remotive API
- USAJOBS API (remote-filtered via `RemoteIndicator=True`)

### 3.2 Normalization layer
Each source is normalized into a common schema:

- text fields: title, company_profile, description, requirements, benefits, etc.
- numeric flags: `telecommuting`, `has_company_logo`, `has_questions`
- source metadata: `source`, `job_url`, `apply_url`, `posted_date`

### 3.3 Language normalization layer
`src/job_fraud_detector/i18n.py`:

- Detects language from title + description + requirements + benefits.
- Uses script heuristics + Latin keyword heuristics + `langdetect`.
- Optionally translates non-English content via `deep-translator` Google backend.
- If confident non-English text is not translated, downstream rule/model confidence is softened.

### 3.4 ML inference layer
`src/job_fraud_detector/inference.py`:

- Model outputs fraud probability `p_fraud`.
- Prediction: `fraudulent = 1 if p_fraud >= threshold else 0`.
- LIME explanation support via `LimeTextExplainer`.

### 3.5 Rules and scoring layer
`src/job_fraud_detector/rules.py`:

- Computes rule-engine trust signals.
- Blends rule trust with ML legitimacy.
- Computes quality score.
- Applies hard caps for severe scam patterns.
- Produces badge + evidence.

### 3.6 UX layer
`app.py`:

- 3 tabs: Home, Job Board, About.
- Job cards with:
  - Apply CTA
  - Trust / Quality / Final / Fraud Risk
  - Scoring breakdown
  - Rule debug panels
  - Expandable full listing view
- CSV export for filtered table.

---

## 4) Data, Features, and Model Inputs

### 4.1 Training dataset
EMSCAD fake job postings dataset.

Current artifact metadata (`artifacts/emscad_light_model/metrics.json`):

- Total rows: **17,880**
- Fraud rate: **0.048434** (~4.84%)
- Split sizes:
  - Train: 12,516
  - Validation: 2,682
  - Test: 2,682

### 4.2 Required model fields
From `src/job_fraud_detector/constants.py` and `features.py`:

- Text columns (12):
  - `title`, `location`, `department`, `company_profile`, `description`, `requirements`, `benefits`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`
- Numeric passthrough columns (3):
  - `telecommuting`, `has_company_logo`, `has_questions`

### 4.3 Feature construction
`prepare_features()`:

- Cleans/normalizes text.
- Coerces numeric flags to integers.
- Builds:
  - `combined_text = " ".join(text columns)`
- Final model input:
  - `combined_text` + 3 numeric columns.

---

## 5) Traditional ML Benchmark + Ensemble

Implemented in `src/job_fraud_detector/modeling.py` and `train.py`.

## 5.1 Candidate model families

1. `logreg_tfidf`
   - TF-IDF (1-2 grams, sublinear TF)
   - LogisticRegression (`class_weight='balanced'`, `liblinear`)
2. `sgd_log_tfidf`
   - TF-IDF
   - SGDClassifier (`loss='log_loss'`, balanced)
3. `linsvc_cal_tfidf`
   - TF-IDF
   - LinearSVC (balanced) + CalibratedClassifierCV
4. `cnb_bow`
   - CountVectorizer (BoW, 1-2 grams)
   - ComplementNB
5. `voting_soft`
   - Soft VotingClassifier over tuned base models

## 5.2 Split strategy

- Stratified split with seed 42
- Train/Val/Test = 70/15/15

## 5.3 Threshold selection objective (recall-first)

`select_threshold_for_recall()`:

1. Evaluate all candidate thresholds from validation probabilities.
2. If recall constraint is met (`recall >= target_recall`, default 0.90):
   - choose threshold with highest precision.
3. If constraint is not met:
   - fallback to max-recall threshold (tie-breaker: higher precision).

## 5.4 Champion selection among model families

Selection key in `train.py` prioritizes:

1. Recall constraint met status (binary priority)
2. Validation precision
3. Validation PR-AUC
4. Validation recall
5. Validation F1

## 5.5 Stored artifacts

- `model.joblib` (champion)
- `candidate_models/*.joblib`
- `config.json` (thresholds and model registry)
- `metrics.json` (per-model + champion metrics)
- `model_comparison.csv`
- `confusion_matrix_test.png`
- `example_explanations.json`

---

## 6) Current Artifact Metrics Snapshot (Exact)

From `artifacts/emscad_light_model/metrics.json`:

- Selected training champion: **`linsvc_cal_tfidf`**
- Selected threshold: **0.424533** (`target_recall_met`)

Validation (champion):

- Precision: 0.9597
- Recall: 0.9154
- F1: 0.9370
- ROC-AUC: 0.9978
- PR-AUC: 0.9718

Test (champion):

- Precision: 0.9407
- Recall: 0.8538
- F1: 0.8952
- ROC-AUC: 0.9954
- PR-AUC: 0.9584

Important deployment note:

- `app.py` defaults `MODEL_NAME` to **`voting_soft`**, so deployed Job Board can intentionally run the voting model threshold (`0.661507`) even if the artifact champion in `metrics.json` is `linsvc_cal_tfidf`.

---

## 7) Scoring Mathematics (Presentation-Ready)

## 7.1 ML fraud probability

- Model outputs `p_fraud` in [0, 1].
- UI fraud risk:
  - `Fraud Risk % = 100 * p_fraud`

## 7.2 ML legitimacy score

- `MLLegitimacyScore = 100 * (1 - p_fraud)` (clipped to [0,100])
- If confident non-English and translation not applied:
  - softening toward neutral:
  - `MLLegitimacyScore = 0.5 * MLLegitimacyScore + 25`

## 7.3 Trust blend

- `TrustRaw = 0.45 * RuleEngineScore + 0.55 * MLLegitimacyScore`
- `TrustScore = min(TrustRaw, TrustCap)` where TrustCap is from hard-cap rules.

## 7.4 Quality score

- `QualityScore = Transparency + RoleSpecificity + RemoteClarity + ApplyExperience + Freshness`
- Each component clipped to its max.

## 7.5 Final opportunity score

- `FinalRaw = 0.70 * TrustScore + 0.30 * QualityScore`
- Guardrail:
  - if `TrustScore < 50`, then
  - `FinalOpportunityScore = min(FinalRaw, 49.0)`

## 7.6 Badge mapping

- Trusted Pick: `Final >= 85` and `Trust >= 75`
- Promising: `Final >= 70`
- Verify First: `Final >= 50`
- Avoid: otherwise

---

## 8) Trust Component Scoring (Exact Rules and Weights)

Trust component maxima:

- Identity/Apply Integrity: max 30
- Communication Safety: max 25
- Monetary Safety: max 20
- Company Evidence: max 15

## 8.1 Identity/Apply Integrity (0-30)

Base and adjustments:

- +8 apply URL exists
- +4 HTTPS apply URL; else -3
- +12 known ATS domain
- +8 known trusted board domain
- +5 major job-board domain
- +6 apply domain matches listing domain
- -4 apply domain differs from listing domain (unless known ATS/board)
- -12 URL shortener domain
- -10 missing apply URL
- -4 redirect-like params (`redirect=`, `url=`)

## 8.2 Communication Safety (starts at 25; clipped 0-25)

- -12 messaging-app recruitment
- -10 text-only interview language
- -8 interview-over-messaging language
- -6 generic low-detail/high-pay language
- -4 urgency pressure language
- -6 no-interview / instant-hire language
- -5 free webmail domains in text

## 8.3 Monetary Safety (starts at 20; clipped 0-20)

- -20 pay-to-work / upfront fee
- -18 fake-check pattern
- -18 task-scam pattern
- -8 early bank/ID request
- -8 crypto/gift-card payment
- -10 money-mule/payment forwarding
- -8 recruit-others / pyramid indicators
- -10 equipment reimbursement-check pattern

## 8.4 Company Evidence (0-15)

- +5 company info present
- +3 location present
- +4 description length >= 80 words
- +3 requirements length >= 30 words

---

## 9) Hard Caps (Decisive Trust Controls)

If any of these fire, trust is capped:

1. Upfront fee/pay-to-get-paid:
   - `TrustCap = min(TrustCap, 10)`
2. Fake-check pattern:
   - `TrustCap = min(TrustCap, 20)`
3. Task-scam pattern:
   - `TrustCap = min(TrustCap, 20)`
4. Messaging-app + interview-via-text/chat semantic combo:
   - `TrustCap = min(TrustCap, 25)`
5. Equipment-check + fake-check combo:
   - `TrustCap = min(TrustCap, 20)`

---

## 10) Quality Component Scoring (Exact Rules and Weights)

Quality component maxima:

- Transparency: max 30
- Role Specificity: max 25
- Remote Clarity: max 20
- Apply Experience: max 15
- Freshness: max 10

## 10.1 Transparency (0-30)

- +10 salary signal from structured field/regex
- else +4 if compensation clarity terms present
- +5 employment type present
- +4 location present
- +6 company present
- +2 company context terms
- +5 benefits available (explicit field or inferred text)
- +2 if benefits span >=3 categories

## 10.2 Role Specificity (0-25)

Description depth:

- +10 if description >=150 words
- +6 if description >=80
- +4 if non-English non-translated but >=50
- else risk flag for short description

Requirements/qualification depth:

- +8 requirements >=60 words
- +5 requirements >=30 words
- +8 if description >=320 and embedded qualification signal
- +5 if description >=220 and embedded qualification signal
- +4 if non-English non-translated and req >=18
- +4 if desc>=300 with responsibility/experience/specificity/qualification cues
- else risk flag for sparse requirements

Structure and specificity bonuses:

- +3 bullet structure in requirements
- specificity signal count bonuses:
  - +2 (>=2 signals)
  - +2 (>=3 signals)
  - +1 (>=4 signals)
- +2 if responsibilities and qualification-section cues both present
- +2 fallback for non-English non-translated with enough length but no specificity signal

## 10.3 Remote Clarity (0-20)

- +10 remote hint (`remote`, `telecommute`, `work from home`, or telecommuting=1)
- +6 timezone/region terms
- +3 remote policy terms
- +4 if location present with remote hint

## 10.4 Apply Experience (0-15)

- +5 apply URL present
- +3 HTTPS (else -2)
- +5 known ATS
- +3 known trusted board
- +2 major job board
- -4 shortener
- +2 hiring process terms

## 10.5 Freshness (0-10)

- unknown age: +3
- <=7 days: +10
- <=14 days: +8
- <=30 days: +6
- <=60 days: +3
- >60 days: +1 + stale risk flag

---

## 11) Exhaustive Rule Inventory (Regex Families)

All pattern families below are implemented directly in `src/job_fraud_detector/rules.py`.

### 11.1 Fraud/safety-focused families

- `MESSAGING_APP_TERMS` (6)
- `TEXT_INTERVIEW_TERMS` (8)
- `INTERVIEW_OVER_MESSAGING_TERMS` (3)
- `PAY_TO_GET_PAID_TERMS` (11)
- `FAKE_CHECK_TERMS` (13)
- `TASK_SCAM_TERMS` (8)
- `BANK_INFO_TERMS` (9)
- `CRYPTO_OR_GIFTCARD_TERMS` (5)
- `MONEY_MULE_TERMS` (5)
- `RECRUIT_OTHERS_TERMS` (5)
- `GENERIC_JOB_TERMS` (7)
- `URGENCY_PRESSURE_TERMS` (6)
- `NO_INTERVIEW_TERMS` (4)
- `EQUIPMENT_CHECK_TERMS` (9)

### 11.2 Quality/specificity-focused families

- `TIMEZONE_OR_REGION_TERMS` (7)
- `ROLE_SPECIFICITY_TERMS` (53)
- `YEARS_EXPERIENCE_TERMS` (5)
- `CREDENTIAL_OR_REQUIREMENT_TERMS` (4)
- `RESPONSIBILITY_STRUCTURE_TERMS` (5)
- `SCHEDULE_OR_AVAILABILITY_TERMS` (6)
- `QUALIFICATION_SECTION_TERMS` (6)
- `HIRING_PROCESS_TERMS` (5)
- `REMOTE_POLICY_CLARITY_TERMS` (9)
- `COMPENSATION_CLARITY_TERMS` (7)
- `COMPANY_CONTEXT_TERMS` (6)

### 11.3 Benefit category detection patterns

- health
- time_off
- equity
- retirement
- family
- learning
- wellness

### 11.4 Domain and email registries

- Known ATS domains: 19
- Known trusted board domains: 4
- Major job-board domains: 6
- URL shortener domains: 9
- Free email domains: 10

---

## 12) Explainability Layer

## 12.1 Local model explanation (LIME)

Inference supports:

- `with_explanation=True`
- returns top token contributions for fraud class.

Implementation detail:

- LIME perturbs text while keeping numeric fields fixed (`telecommuting`, `has_company_logo`, `has_questions`) for consistent local explanations.

## 12.2 Rule-level explanation

Each scored row includes:

- `positive_evidence`
- `risk_flags`
- `hard_caps`
- `matched_signals` by family
- trust/quality component breakdowns

UI exposes fired/not-fired debug checks and score-computation panels.

---

## 13) Live Source Link Quality Reality

Observed behavior in real feeds:

- Many APIs/RSS provide a listing URL but not guaranteed direct ATS apply link.
- Current implementation tries to recover better apply targets by scanning text URL candidates and ranking them heuristically.
- This is intentionally non-scraping and compatible with lightweight deployment.

---

## 14) Testing and Reliability Evidence

Tests implemented:

1. `tests/test_features.py`
   - Required-column contract
   - Null handling + numeric coercion
2. `tests/test_modeling.py`
   - Candidate registry IDs
   - Probability output support
   - Voting pipeline fit/predict smoke
3. `tests/test_training_inference.py`
   - End-to-end training smoke
   - Artifact presence checks
   - Metrics schema checks
   - Inference contract checks
   - LIME non-empty output check
   - Voting model explicit load check
   - Behavioral threshold-direction checks

---

## 15) Deployment and Runtime Notes

From current app config behavior:

- Default share: enabled (`GRADIO_SHARE` defaults to true in app launch logic).
- LAN binding: `server_name="0.0.0.0"` for same-network access.
- Tabs:
  - Home
  - Job Board
  - About

---

## 16) Slide-by-Slide Deck Blueprint

Use this sequence for a strong judge-facing story.

### Slide 1 - Title

- RemoteTrust
- Scale Without Borders Hackathon 2026
- Team VectorSpace

### Slide 2 - Problem

- Remote jobs are abundant but noisy.
- Users need legitimacy + quality, not one binary label.

### Slide 3 - Product Demo Snapshot

- Show Job Board card.
- Highlight Trust, Quality, Final, and explanation panel.

### Slide 4 - System Architecture

- Source adapters -> language normalization -> ML + rules -> explainable UI.

### Slide 5 - Data and Feature Contract

- EMSCAD baseline + live feeds.
- Combined text + numeric flags.

### Slide 6 - Model Strategy

- Traditional ML benchmark.
- Why sparse text models + voting ensemble.
- Recall-first threshold policy.

### Slide 7 - Model Results

- Show per-model comparison table and selected model.
- Mention artifact-level metrics.

### Slide 8 - Trust and Quality Math

- Trust formula
- Final formula
- Guardrail rule
- Fraud Risk % derivation

### Slide 9 - Trust Rules

- Four trust components + point logic.
- Hard caps for severe scam patterns.

### Slide 10 - Quality Rules

- Five quality components.
- Why this improves user usefulness over pure fraud detection.

### Slide 11 - Explainability

- LIME local explanation.
- Rule fired/not-fired debug.
- Human-readable reasons.

### Slide 12 - Live Feeds and Real-World Constraints

- 4 sources.
- Apply-link limitations in API/RSS.
- Current non-scraping heuristic approach.

### Slide 13 - Product UX Decisions

- Action badges: Trusted Pick / Promising / Verify First / Avoid.
- Collapsible score explanations and full listing view.

### Slide 14 - Reliability and Testing

- Unit + integration tests.
- Artifacts generated per training run.

### Slide 15 - Impact and Roadmap

- Immediate value to remote job seekers.
- Next: stronger direct-apply resolution, richer quality signals, calibrated retraining cadence.

---

## 17) Suggested Speaker Notes (Short)

- “We separated trust from quality to avoid rewarding polished scams.”
- “We optimized thresholding for high recall first, then precision.”
- “We did not rely on black-box scoring only; every score is auditable.”
- “Hard caps are deliberate safety controls inspired by scam guidance.”
- “The UI is designed for decision support, not just classification output.”

---

## 18) References

- Scikit-learn sparse text benchmark: https://sklearn.org/1.7/auto_examples/text/plot_document_classification_20newsgroups.html
- ORFDetector (ensemble motivation): https://researchportal.hw.ac.uk/en/publications/orfdetector-ensemble-learning-based-online-recruitment-fraud-dete/
- VotingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
- Calibration guide: https://sklearn.org/stable/modules/calibration.html
- CalibratedClassifierCV: https://sklearn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- ComplementNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
- EMSCAD paper: https://www.mdpi.com/1999-5903/9/1/6
- FTC Consumer Alert (job scams via WhatsApp): https://consumer.ftc.gov/consumer-alerts/2024/08/dont-send-your-social-security-number-employer-whatsapp-its-scam
- FTC Data Spotlight (paying to get paid task scams): https://www.ftc.gov/news-events/data-visualizations/data-spotlight/2024/12/paying-get-paid-gamified-job-scams-drive-record-losses
- Government of Canada scam awareness: https://www.canada.ca/en/revenue-agency/campaigns/fraud-scams.html

---

## 19) Appendix A - Pattern Lists (Verbatim Style Names)

For full regex inventories, see `src/job_fraud_detector/rules.py` constants:

- `MESSAGING_APP_TERMS`
- `TEXT_INTERVIEW_TERMS`
- `INTERVIEW_OVER_MESSAGING_TERMS`
- `PAY_TO_GET_PAID_TERMS`
- `FAKE_CHECK_TERMS`
- `TASK_SCAM_TERMS`
- `BANK_INFO_TERMS`
- `CRYPTO_OR_GIFTCARD_TERMS`
- `MONEY_MULE_TERMS`
- `RECRUIT_OTHERS_TERMS`
- `GENERIC_JOB_TERMS`
- `URGENCY_PRESSURE_TERMS`
- `NO_INTERVIEW_TERMS`
- `EQUIPMENT_CHECK_TERMS`
- `TIMEZONE_OR_REGION_TERMS`
- `ROLE_SPECIFICITY_TERMS`
- `YEARS_EXPERIENCE_TERMS`
- `CREDENTIAL_OR_REQUIREMENT_TERMS`
- `RESPONSIBILITY_STRUCTURE_TERMS`
- `SCHEDULE_OR_AVAILABILITY_TERMS`
- `QUALIFICATION_SECTION_TERMS`
- `HIRING_PROCESS_TERMS`
- `REMOTE_POLICY_CLARITY_TERMS`
- `COMPENSATION_CLARITY_TERMS`
- `COMPANY_CONTEXT_TERMS`
- `BENEFIT_CATEGORY_PATTERNS`

---

## 20) Appendix B - Exhaustive Trust/Quality Rule Lists (Exact)

This section is intentionally verbose so you can map each slide claim to implemented code.

## 20.1 Domain and registry rules

### Known ATS domains (exact set)

- `boards.greenhouse.io`
- `job-boards.greenhouse.io`
- `greenhouse.io`
- `jobs.lever.co`
- `hire.lever.co`
- `jobs.ashbyhq.com`
- `ashbyhq.com`
- `myworkdayjobs.com`
- `myworkdaysite.com`
- `taleo.net`
- `icims.com`
- `smartrecruiters.com`
- `jobvite.com`
- `successfactors.com`
- `oraclecloud.com`
- `workable.com`
- `bamboohr.com`
- `teamtailor.com`
- `recruitee.com`

### Known trusted job-source domains

- `weworkremotely.com`
- `jobicy.com`
- `remotive.com`
- `usajobs.gov`

### Major job-board domains

- `linkedin.com`
- `indeed.com`
- `glassdoor.com`
- `ziprecruiter.com`
- `monster.com`
- `simplyhired.com`

### URL shortener domains

- `bit.ly`
- `tinyurl.com`
- `t.co`
- `goo.gl`
- `ow.ly`
- `shorturl.at`
- `rebrand.ly`
- `is.gd`
- `rb.gy`

### Free email domains used as risk cues

- `gmail.com`
- `yahoo.com`
- `outlook.com`
- `hotmail.com`
- `aol.com`
- `icloud.com`
- `proton.me`
- `protonmail.com`
- `gmx.com`
- `mail.com`

## 20.2 Regex rule families (exact patterns)

### MESSAGING_APP_TERMS

- `\bwhatsapp\b`
- `\btelegram\b`
- `\bsignal\b`
- `\bdiscord\b`
- `\bmessenger\b`
- `\breply\s+(yes|interested)\b`

### TEXT_INTERVIEW_TERMS

- `\btext\s*[- ]?only\s+interview\b`
- `\btext\s*[- ]?based\s+interview\b`
- `\bsms\s+interview\b`
- `\btext\s+message\s+interview\b`
- `\binterview\s+over\s+text\b`
- `\binterview\s+via\s+chat\b`
- `\bchat\s+interview\s+only\b`
- `\b(?:text|chat)\s*[- ]?only\s+(?:interview|screening)\b`

### INTERVIEW_OVER_MESSAGING_TERMS

- `\binterview(?:\s+process)?\s+(?:is\s+)?(?:conducted|held|done|completed)?\s*(?:over|via|through|on|in)\s+(?:telegram|whatsapp|signal|discord|messenger)\b`
- `\b(?:telegram|whatsapp|signal|discord|messenger)\s+(?:interview|chat\s+interview|text\s+interview)\b`
- `\b(?:telegram|whatsapp|signal|discord|messenger)\s+(?:text|chat)\s*[- ]?only\b`

### PAY_TO_GET_PAID_TERMS

- `\bpay\s+to\s+get\s+paid\b`
- `\bpay\s+to\s+start\b`
- `\bapplication\s+fee\b`
- `\bupfront\s+fee\b`
- `\bregistration\s+fee\b`
- `\bonboarding\s+fee\b`
- `\btraining\s+fee\b`
- `\bsecurity\s+deposit\b`
- `\bactivation\s+fee\b`
- `\bunlock\s+(payment|fee)\b`
- `\bupgrade\s+fee\b`

### FAKE_CHECK_TERMS

- `\bfake\s+check\b`
- `\bcounterfeit\s+cheque?\b`
- `\bdeposit\s+(a\s+)?check\b`
- `\bdeposit\s+it\b`
- `\bmobile\s+deposit\b`
- `\bcash\s+(a\s+)?check\b`
- `\bsend\s+(some\s+of\s+the\s+money\s+)?back\b`
- `\bsend\s+(part|portion|some)\s+(to|of)\s+(a\s+)?vendor\b`
- `\bcheck\s+will\s+be\s+mailed\b`
- `\bmailed?\s+(you\s+)?(a\s+)?check\b`
- `\boverpayment\b`
- `\bmystery\s+shopper\b`
- `\bsecret\s+shopper\b`

### TASK_SCAM_TERMS

- `\btask\s+scam\b`
- `\bproduct\s+boost(ing)?\b`
- `\bapp\s+optimization\b`
- `\bonline\s+tasks?\b`
- `\bcommission\s+per\s+click\b`
- `\bweb\s+surveys?\b`
- `\brating\s+tasks?\b`
- `\breview\s+tasks?\b`

### BANK_INFO_TERMS

- `\bbank\s+account\b`
- `\brouting\s+number\b`
- `\baccount\s+number\b`
- `\bsocial\s+security\s+number\b`
- `\bssn\b`
- `\bsin\b`
- `\bpassport\b`
- `\bdriver'?s\s+license\b`
- `\bcredit\s+card\b`

### CRYPTO_OR_GIFTCARD_TERMS

- `\bcrypto(currency)?\b`
- `\bbitcoin\b`
- `\bethereum\b`
- `\busdt\b`
- `\bgift\s+cards?\b`

### MONEY_MULE_TERMS

- `\breceive\s+payments?\b`
- `\bforward\s+money\b`
- `\bpayment\s+processor\b`
- `\bfinancial\s+agent\b`
- `\bwire\s+transfer\b`

### RECRUIT_OTHERS_TERMS

- `\brecruit\s+others\b`
- `\bdownline\b`
- `\bpyramid\s+selling\b`
- `\bmlm\b`
- `\bmulti\s*[- ]?level\s+marketing\b`

### GENERIC_JOB_TERMS

- `\bno\s+experience\s+needed\b`
- `\bimmediate\s+start\b`
- `\bwork\s+from\s+your\s+phone\b`
- `\beasy\s+money\b`
- `\bguaranteed\s+income\b`
- `\bearn\s+\$?\d+\s*(per\s+day|daily|weekly)\b`
- `\bdaily\s+pay\b`

### URGENCY_PRESSURE_TERMS

- `\bact\s+now\b`
- `\bstart\s+today\b`
- `\blimited\s+slots?\b`
- `\bimmediate\s+hiring\b`
- `\bonly\s+\d+\s+spots?\b`
- `\burgent(ly)?\b`

### NO_INTERVIEW_TERMS

- `\bno\s+interview\b`
- `\binstant\s+hire\b`
- `\bauto[- ]?approved\b`
- `\bguaranteed\s+hire\b`

### EQUIPMENT_CHECK_TERMS

- `\bequipment\s+reimbursement\b`
- `\bbuy\s+(your|the)\s+equipment\b`
- `\bhome\s+office\s+kit\b`
- `\bcheck\s+for\s+equipment\b`
- `\bequipment\s+check\b`
- `\bcheck\s+for\s+home\s+office\b`
- `\bbuy\s+from\s+(an?\s+)?approved\s+vendor\b`
- `\bsend\s+money\s+to\s+(an?\s+)?approved\s+vendor\b`
- `\bvendor\s+for\s+equipment\b`

### TIMEZONE_OR_REGION_TERMS

- `\btimezone\b`
- `\btime\s+zone\b`
- `\bnorth\s+america\b`
- `\bwithin\s+(the\s+)?(us|u\.s\.|usa|canada|eu|europe)\b`
- `\bmust\s+reside\s+in\b`
- `\bus[- ]based\b`
- `\bcanada[- ]based\b`

### ROLE_SPECIFICITY_TERMS

- `\bpython\b`
- `\bsql\b`
- `\bexcel\b`
- `\btableau\b`
- `\bpower\s*bi\b`
- `\baws\b`
- `\bgcp\b`
- `\bjavascript\b`
- `\breact\b`
- `\bjava\b`
- `\bc\+\+\b`
- `\bdbt\b`
- `\bairflow\b`
- `\bsnowflake\b`
- `\blooker\b`
- `\bjira\b`
- `\bconfluence\b`
- `\bfigma\b`
- `\badobe\b`
- `\bphotoshop\b`
- `\billustrator\b`
- `\bpremiere\b`
- `\bindesign\b`
- `\bcanva\b`
- `\bwordpress\b`
- `\bseo\b`
- `\bsem\b`
- `\bgoogle\s+analytics\b`
- `\bsalesforce\b`
- `\bhubspot\b`
- `\bzendesk\b`
- `\bintercom\b`
- `\bfreshdesk\b`
- `\bcrm\b`
- `\berp\b`
- `\bquickbooks\b`
- `\bnetsuite\b`
- `\bsap\b`
- `\bshopify\b`
- `\bamazon\s+seller\s+central\b`
- `\bemr\b`
- `\behr\b`
- `\bepic\b`
- `\bhipaa\b`
- `\brn\b`
- `\blpn\b`
- `\bcna\b`
- `\bcdl\b`
- `\bworkday\b`
- `\bgreenhouse\b`
- `\blever\b`
- `\bats\b`

### YEARS_EXPERIENCE_TERMS

- `\b\d+\+?\s+years?\s+of\s+experience\b`
- `\b\d+\+?\s+years?\s+experience\b`
- `\b\d+\+?\s+years?\s+of\s+[a-zA-Z0-9,\-\/\s]{1,60}\s+experience\b`
- `\bminimum\s+\d+\+?\s+years?\b`
- `\b\d+\+?\s+yrs?\b`

### CREDENTIAL_OR_REQUIREMENT_TERMS

- `\b(certification|certified|license|licensed|credential)\b`
- `\b(bachelor'?s|master'?s|ph\.?d|diploma|degree)\b`
- `\bbackground\s+check\b`
- `\bwork\s+authorization\b`

### RESPONSIBILITY_STRUCTURE_TERMS

- `\bresponsibilit(y|ies)\b`
- `\bkey\s+duties\b`
- `\bwhat\s+you(?:'ll|\s+will)\s+do\b`
- `\bday[- ]to[- ]day\b`
- `\byou\s+will\b`

### SCHEDULE_OR_AVAILABILITY_TERMS

- `\bshift\b`
- `\bweekend(s)?\b`
- `\bon[- ]call\b`
- `\bbusiness\s+hours\b`
- `\bavailability\b`
- `\boverlap\s+hours\b`

### QUALIFICATION_SECTION_TERMS

- `\brequirements?\b`
- `\bqualifications?\b`
- `\bmust\s+have\b`
- `\bnice\s+to\s+have\b`
- `\bwho\s+you\s+are\b`
- `\bwhat\s+you(?:'ll|\s+will)\s+bring\b`

### HIRING_PROCESS_TERMS

- `\bhiring\s+process\b`
- `\binterview\s+process\b`
- `\bapplication\s+process\b`
- `\bnext\s+steps\b`
- `\bhow\s+to\s+apply\b`

### REMOTE_POLICY_CLARITY_TERMS

- `\bworldwide\b`
- `\bglobal\b`
- `\banywhere\b`
- `\bremote[- ]?first\b`
- `\bdistributed\b`
- `\bcore\s+hours\b`
- `\boverlap\s+hours\b`
- `\b(?:async|asynchronous)\b`
- `\btime\s*zone\b`

### COMPENSATION_CLARITY_TERMS

- `\bsalary\s+range\b`
- `\bpay\s+range\b`
- `\bcompensation\s+range\b`
- `\bper\s+(hour|year|annum)\b`
- `\bhourly\b`
- `\bannually\b`
- `\btotal\s+compensation\b`

### COMPANY_CONTEXT_TERMS

- `\babout\s+us\b`
- `\babout\s+the\s+company\b`
- `\bour\s+mission\b`
- `\bour\s+values\b`
- `\bour\s+culture\b`
- `\bwho\s+we\s+are\b`

### BENEFIT_CATEGORY_PATTERNS

- `health`: `\b(?:health|medical|dental|vision|insurance)\b`
- `time_off`: `\b(?:pto|paid\s+time\s+off|vacation|holiday)s?\b`
- `equity`: `\b(?:equity|stock|rsu|options)\b`
- `retirement`: `\b(?:401\(k\)|pension|rrsp)\b`
- `family`: `\b(?:parental|maternity|paternity)\b`
- `learning`: `\b(?:learning\s+budget|tuition|professional\s+development|training)\b`
- `wellness`: `\b(?:wellness|mental\s+health|eap)\b`

---

## 21) Appendix C - CLI, Inference, and Artifact Contracts

## 21.1 Training CLI (implemented)

Script entrypoint:

- `python -m job_fraud_detector.train`

Arguments:

- `--data-path` (required)
- `--artifacts-dir` (required)
- `--seed` (default 42)
- `--target-recall` (default 0.90)
- `--test-size` (default 0.15)
- `--val-size` (default 0.15)
- `--c-grid` (optional CSV override for LogReg C values)
- `--candidate-models` (optional CSV ids from: `logreg_tfidf,sgd_log_tfidf,linsvc_cal_tfidf,cnb_bow,voting_soft`)
- `--disable-voting` (flag)
- `--plot-confusion-matrix / --no-plot-confusion-matrix` (default true)
- `--max-rows` (optional smoke-run cap)
- `--lime-num-samples` (default 1500; for saved example explanations)

## 21.2 Inference API contract (implemented)

Class:

- `FraudDetector.from_artifacts(artifacts_dir, model_name=None, prefer_voting=False)`

Prediction function:

- `predict_posting(posting_dict, with_explanation=False, num_features=10, num_samples=3000)`

Returns:

- `fraud_probability` (float)
- `prediction` (0/1 using selected threshold)
- `threshold` (float)
- optional `lime_explanation` list of `{token, weight}`

## 21.3 Produced artifacts

- `artifacts/.../model.joblib` (selected champion)
- `artifacts/.../candidate_models/*.joblib` (all trained candidates)
- `artifacts/.../config.json`
- `artifacts/.../metrics.json`
- `artifacts/.../model_comparison.csv`
- `artifacts/.../confusion_matrix_test.png`
- `artifacts/.../example_explanations.json`

---

## 22) Appendix D - Reproducibility Checklist for Demo and Slides

1. Train with fixed seed and save artifacts:
   - `python -m job_fraud_detector.train --data-path /path/to/fake_job_postings.csv --artifacts-dir artifacts/emscad_light_model`
2. Run tests:
   - `pytest -q`
3. Launch app:
   - `python app.py`
4. Capture screenshots for:
   - Home summary
   - Job Board card with trust/quality/final and formula panel
   - Rule Debug panel
   - Full listing metadata view
5. Export filtered CSV from UI and include one row in appendix slide.
6. Use this report as presenter script and slide-outline source.
