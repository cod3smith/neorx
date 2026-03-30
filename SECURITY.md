# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue.
2. Email **security@neoforge.dev** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive acknowledgement within **48 hours**.
4. We will work with you to understand and address the issue before
   any public disclosure.

## Scope

This policy covers:

- The NeoRx Python package and all sub-modules
- The Docker Compose deployment (API server, Redis, PostgreSQL)
- The FastAPI REST endpoints

### Out of Scope

- Third-party dependencies (report to their maintainers directly)
- External biomedical APIs (Monarch Initiative, Open Targets, etc.)

## Security Considerations

### API Keys & Credentials

NeoRx reads all credentials from environment variables.
No secrets are hardcoded. The `.env.example` file contains only
placeholder values.

**Users must change default passwords** (especially `POSTGRES_PASSWORD`)
before any non-local deployment.

### Network Access

The pipeline makes outbound HTTP requests to 7 biomedical databases.
All connections use HTTPS. No data is sent to any analytics or
telemetry service.

### Data Privacy

NeoRx processes disease names and molecular structures.
No patient data, PII, or PHI is collected or transmitted.
