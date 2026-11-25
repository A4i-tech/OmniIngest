# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately in one of these ways:

- Use GitHub's private vulnerability reporting: the **Report a vulnerability** button on the repository's Security tab.
- Or email dev.a4i@iiitb.ac.in with the details and steps to reproduce.

We aim to acknowledge a report within 5 working days and to share a plan or fix within 30 days. We will credit you in the release notes unless you prefer otherwise.

## Supported versions

Security fixes go to the latest released version. OmniIngest is pre-1.0 in its public form; older versions are not patched.

## Scope notes

OmniIngest runs pipelines you define in YAML. It can read local files and remote resources and call cloud services with your credentials. Treat pipeline files and input documents from untrusted sources accordingly, and keep API keys out of pipeline files and logs.
