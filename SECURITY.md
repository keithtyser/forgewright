# Security

Forgewright can run shell commands, write files, and use account credentials. Run it under an account with only the access the task needs. Approval prompts are a user control, not a sandbox.

Report vulnerabilities through this repository's private security reporting feature when available. If it is unavailable, open an issue asking for a private contact without including credentials or exploit details.

The development lock is checked by `pip-audit`; the frontend lock is checked by `npm audit`. These checks identify published advisories, not every possible flaw.

Credential files use restricted permissions and atomic replacement. They remain plaintext. Logs redact common secret fields and known environment credentials; inspect evidence before sharing it. Set provider spending limits outside Forgewright. Do not rely on agent prompts to enforce account or GPU limits.

The direct Codex OAuth adapter remains experimental. Its lock coordinates Forgewright processes, not the official Codex CLI. Avoid concurrent token refreshes from both applications. A future supported integration should delegate authentication to Codex app-server.
