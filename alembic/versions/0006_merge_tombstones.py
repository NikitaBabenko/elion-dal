"""Persist non-destructive cross-source merge tombstones.

Revision ID: 0006_merge_tombstones
Revises: 0005_backfill_metadata
Create Date: 2026-08-21
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_merge_tombstones"
down_revision: str | None = "0005_backfill_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("tombstone_reason", sa.String(128), nullable=False, server_default=""),
    )
    op.add_column(
        "documents",
        sa.Column("merged_into_doc_id", sa.String(256), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("documents", "merged_into_doc_id")
    op.drop_column("documents", "tombstone_reason")
