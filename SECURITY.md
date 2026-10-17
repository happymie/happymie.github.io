# Security Policy

## Reporting a vulnerability

Please use [GitHub's private vulnerability reporting](https://github.com/happymie/happymie.github.io/security/advisories/new). Do not publish exploit details or sensitive data in a public issue.

This repository is a static GitHub Pages site. It does not run a private application server, so server-side rate limiting, request filtering, custom response headers, and bot challenges cannot be configured in this repository alone. For edge-level controls, place the domain behind a service that supports WAF rules, rate limits, and response-header policies.

## Scope and limitations

- Public files can always be downloaded by a client that can access the site. `robots.txt` is a voluntary crawler policy, not access control.
- Do not commit credentials, API tokens, private datasets, patient data, or unpublished research data.
- Keep GitHub Actions permissions minimal. Review third-party actions and pin them to full commit SHAs before using them in production.
- Review the repository's Actions results after each update. A successful static check does not prove resistance to denial-of-service attacks or scraping.
