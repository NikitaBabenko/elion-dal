"""Единая настройка структурного логирования (вместо print)."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
