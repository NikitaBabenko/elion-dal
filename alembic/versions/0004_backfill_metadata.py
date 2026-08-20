"""Persist lifecycle metadata used by safe knowledge-base backfills.

Revision ID: 0004_backfill_metadata
Revises: 0003_app_settings
Create Date: 2026-08-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_backfill_metadata"
down_revision: str | None = "0003_app_settings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("academic_year", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("is_active", sa.Boolean(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("metadata_fingerprint", sa.String(64), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("documents", "metadata_fingerprint")
    op.drop_column("documents", "is_active")
    op.drop_column("documents", "academic_year")
