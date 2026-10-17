# Security Policy

## Reporting a vulnerability

If GitHub's private reporting form is available, [report a vulnerability privately](https://github.com/happymie/happymie.github.io/security/advisories/new). If the form is unavailable, open an issue requesting a private contact channel **without** including exploit details, credentials, or sensitive data. Availability of the private form depends on the repository's GitHub settings.

Include the affected page or file, the observed behavior, and minimal reproduction steps in the private report. Test with synthetic data and avoid disrupting other visitors. The current contact policy is also published in [security.txt](https://happymie.github.io/.well-known/security.txt).

## Public site and crawler policy

This is a static GitHub Pages site with a public source repository. A client that can access a public file can download it, and the source repository provides another way to obtain the published material. `robots.txt` asks compliant automated clients to observe the publisher's crawl policy; it is not authentication, a rate limit, or a guarantee against copying. Regular visitors retain access to the learning materials and explicit downloads.

GitHub operates the hosting infrastructure and its platform-level traffic limits. This repository alone cannot install a server-side WAF, IP blocking, per-client request limits, or an enforced bot challenge. Browser checks, disabled right-click menus, and JavaScript-only CAPTCHAs do not add those controls.

## Maintenance and release safety

Daily maintenance uses the existing ChatGPT scheduled task and the connected GitHub integration. Accepted changes are checked against the expected current repository revision before advancing `main`; publication follows the repository's configured GitHub Pages deployment. Check the task result, accepted commit, deployment result, and live content before treating an update as published. The repository does not add a second daily schedule for the same work.

- Treat retrieved titles, descriptions, links, and repository metadata as untrusted input. Validate their schema and URL protocols, and escape text when generating HTML. Do not import executable page markup from a discovery feed.
- The source metadata checker only requests allowlisted public API hosts over HTTPS. It rejects cross-host redirects, credential-bearing URLs, nonstandard ports, and oversized responses. GitHub credentials are sent only to GitHub's API.
- Validate a candidate source record before replacing its last verified pin or date. Failed refreshes keep the prior verified values and record an error. Registry files are replaced atomically after complete serialization.
- Keep code revisions and applicable license notices explicit. A metadata refresh does not install or execute upstream training scripts and does not download model weights. Adding a runnable adapter requires separate implementation and verification.
- Keep Actions permissions minimal and pin actions to verified full commit SHAs. Use only the permissions required by each job, and do not expose credentials to untrusted pull-request code.
- Keep credentials, private datasets, patient information, and unpublished research out of the published artifact and repository. Static checks and secret scanners are useful gates, not proof that every sensitive value has been detected.

## Browser policy and optional edge controls

Keep scripts and essential assets local where practical. CSP must be checked against actual page interactions. A CSP delivered through an HTML `meta` element cannot enforce `frame-ancestors`, `sandbox`, `report-uri`, or a report-only policy. `X-Frame-Options` also requires an HTTP response header. Do not count ignored meta directives as working protection.

If enforceable traffic controls become necessary, use a domain and delivery service under the publisher's control, with response-header policies, request filtering, and measured rate limits. Confirm normal reading and downloads still work, and check whether the original site or public repository can bypass those controls. This is an optional deployment change; no edge service is enabled by the files in this repository.

## References

- [GitHub Pages hosting](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) and [platform limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
- [GitHub Actions secure use](https://docs.github.com/en/actions/reference/security/secure-use)
- [W3C: CSP delivery through a meta element](https://w3c.github.io/webappsec-csp/#meta-element)
- [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html#section-3)
