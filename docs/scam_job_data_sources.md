# Scam/Fake/Low-Quality Job Data Sources (Inventory Only)

This note lists public APIs/datasets that can support fraud-signal engineering or monitoring workflows.
These sources are **not integrated into model training** in this project.

## Job Feed APIs (Generally Not Fraud-Labeled)

- Remotive API: [GitHub docs](https://github.com/remotive-io/remote-jobs-api)
- Jobicy feeds/API references: [Jobicy RSS/API page](https://jobicy.com/jobs-rss-feed)
- We Work Remotely RSS: [WWR RSS feed](https://weworkremotely.com/remote-jobs.rss)
- USAJOBS API: [USAJOBS API reference](https://developer.usajobs.gov/api-reference/)

## Scam-Signal APIs (Domain/URL Risk, Not Per-Posting Job Labels)

- Google Safe Browsing Lookup API: [Docs](https://developers.google.com/safe-browsing/v4/lookup-api)
- VirusTotal URL APIs: [URL endpoint docs](https://docs.virustotal.com/reference/url)
- PhishTank developer feeds/API: [Developer info](https://phishtank.org/developer_info.php)
- ScamAdviser API: [API docs](https://api.scamadviser.cloud/docs/)

## Complaint and Enforcement Data Caveat

- FTC Consumer Sentinel is law-enforcement restricted: [FTC Sentinel](https://www.ftc.gov/enforcement/consumer-sentinel-network)
- FTC also publishes public data resources/dashboards: [FTC data visualizations + API index](https://www.ftc.gov/enforcement/data-visualizations)

## Practical Limitation

There is currently no widely adopted public API that provides high-quality, first-party,
at-scale **per-posting "fake job" labels** suitable for direct supervised training.
