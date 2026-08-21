"""Persist lifecycle metadata used by safe knowledge-base backfills.

Revision ID: 0005_backfill_metadata
Revises: 0004_document_metadata
Create Date: 2026-08-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_backfill_metadata"
down_revision: str | None = "0004_document_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("metadata_fingerprint", sa.String(64), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("documents", "metadata_fingerprint")
