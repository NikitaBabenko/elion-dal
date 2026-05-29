"""Проверка API-токена для gRPC-ручек (чистые функции — легко тестировать).

Токен передаётся клиентом в metadata: `authorization: Bearer <token>` или
`x-api-token: <token>`. HealthCheck намеренно открыт (для проб платформы).
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable

HEALTH_METHOD = "HealthCheck"


def extract_token(metadata: Iterable[tuple[str, str]] | None) -> str:
    """Достать токен из gRPC-metadata (authorization: Bearer ... | x-api-token)."""
    md = {k.lower(): v for k, v in (metadata or [])}
    auth = md.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return md.get("x-api-token", "")


def token_ok(metadata: Iterable[tuple[str, str]] | None, expected: str) -> bool:
    """True, если токен не требуется (expected пуст) или совпадает (constant-time)."""
    if not expected:
        return True
    return secrets.compare_digest(extract_token(metadata), expected)
