"""Remove known credentials from durable logs."""
from __future__ import annotations

import os
import re


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"(?:api[_-]?key|password|secret|access_token|refresh_token|authorization)", str(k), re.I)
                else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if len(secret) >= 8 and re.search(r"KEY|TOKEN|PASSWORD|SECRET", key):
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
        return re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", value, flags=re.I)
    return value
