# Heuristic Rulebook (Trust + Quality)

This app now uses a transparent rules layer alongside the ML fraud probability.

## Composite Scoring

- `Trust Score = 0.45 * RuleEngineScore + 0.55 * MLLegitimacyScore`
- `Final Opportunity Score = 0.70 * Trust + 0.30 * Quality`
- If `Trust < 50`, final score is capped below 50 to avoid ranking polished scams as good opportunities.

## Trust Heuristics (Fraud/Safety Signals)

### 1) Identity & Apply Integrity (0-30)

- Positive signals:
  - Apply URL present
  - HTTPS apply URL
  - Apply URL domain matches known ATS domain (for example: `greenhouse`, `lever`, `ashby`, `myworkdayjobs`, `taleo`, `icims`, `smartrecruiters`, `jobvite`, `successfactors`)
  - Apply URL domain matches major job boards (`linkedin`, `indeed`, `glassdoor`) with a smaller boost than ATS
  - Apply domain matches posting domain
  - Apply domain from known trusted board source
- Negative signals:
  - URL shortener domains (`bit.ly`, `tinyurl`, etc.)
  - Redirect-like apply parameters
  - Missing apply URL

### 2) Communication Safety (0-25)

- Negative signals:
  - Messaging-app recruiting language (`WhatsApp`, `Telegram`, `Signal`, `Discord`, `Messenger`)
  - Text-only interview language (`SMS interview`, `interview over text`, etc.)
  - No-interview / instant-hire language
  - High-pressure urgency cues (`act now`, `start today`, `limited slots`)
  - Free-webmail contact domains (`gmail`, `yahoo`, etc.) in posting text
  - Generic high-pay/low-detail recruiting language (`no experience needed`, `immediate start`, etc.)

### 3) Monetary Safety (0-20)

- Negative signals:
  - Pay-to-work or upfront-fee language (`application fee`, `onboarding fee`, `pay to start`)
  - Fake-check patterns (`deposit check`, `send money back`, `overpayment`, `mystery shopper`)
  - Task-scam patterns (`product boosting`, `online tasks`, `commission per click`)
  - Equipment reimbursement/purchase check patterns
  - Early banking/identity request language (`bank account`, `routing number`, `SSN/SIN`)
  - Crypto or gift-card payment language
  - Money-mule/payment-forwarding language
  - Recruit-others / pyramid-style language

### 4) Company Evidence (0-15)

- Positive signals:
  - Company info present
  - Location present
  - Substantive description and requirements text length

## Hard Caps (Decisive Risk Controls)

- Upfront fee / pay-to-get-paid language: trust cap = `10`
- Fake-check scam language: trust cap = `20`
- Task-scam language: trust cap = `20`
- Messaging-app + interview-via-text/chat patterns together: trust cap = `25`
- Equipment reimbursement + fake-check patterns together: trust cap = `20`

## Quality Heuristics (High/Low Opportunity Signals)

### 1) Transparency (0-30)

- Salary signal present (from explicit salary field or salary text pattern)
- Employment type present
- Location present
- Company info present
- Benefits present

### 2) Role Specificity (0-25)

- Description length and requirements depth
- Structured requirements indicators (bullet-like formatting)
- Domain-agnostic specificity signals:
  - role tools/platforms across functions (engineering, design, support, sales, healthcare, operations, HR)
  - explicit years-of-experience patterns
  - credentials/licenses/degrees requirements
  - responsibility wording and schedule/availability clarity

### 3) Remote Clarity (0-20)

- Explicit remote/telecommute language
- Timezone/region constraints clearly stated
- Remote + location context clarity

### 4) Apply Experience (0-15)

- Apply URL present
- HTTPS
- Known ATS/public board domain
- Penalty for apply domains that differ from posting domains (unless known ATS/board)
- Penalty for shortener links

### 5) Freshness (0-10)

- Uses posting date recency buckets (`<=7`, `<=14`, `<=30`, `<=60`, older)

## Badges

- `Trusted Pick`: final 85-100 and trust >= 75
- `Promising`: final 70-84
- `Verify First`: final 50-69
- `Avoid`: final < 50

## ATS Domain Trust Signal: Why it helps

Greenhouse/Lever/Ashby are ATS providers, not generic job boards. If a posting's apply URL points to these domains, it usually indicates the posting is tied to a real recruiting workflow rather than an arbitrary form or messaging contact. This is treated as a strong positive trust signal, but not a guarantee.
