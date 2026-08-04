"""document metadata: canonical identity, academic year and active state

Revision ID: 0004_document_metadata
Revises: 0003_app_settings
Create Date: 2026-08-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_document_metadata"
down_revision: str | None = "0003_app_settings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _document_columns() -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns("documents")}


def _document_indexes() -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes("documents")
        if index.get("name")
    }


def upgrade() -> None:
    columns = _document_columns()
    if "canonical_doc_id" not in columns:
        op.add_column(
            "documents",
            sa.Column("canonical_doc_id", sa.String(256), nullable=True),
        )
    if "academic_year" not in columns:
        op.add_column(
            "documents",
            sa.Column("academic_year", sa.Integer(), nullable=False, server_default="0"),
        )
    if "is_active" not in columns:
        op.add_column(
            "documents",
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        )

    if "ix_documents_canonical_doc_id" not in _document_indexes():
        op.create_index(
            "ix_documents_canonical_doc_id",
            "documents",
            ["canonical_doc_id"],
        )


def downgrade() -> None:
    if "ix_documents_canonical_doc_id" in _document_indexes():
        op.drop_index("ix_documents_canonical_doc_id", table_name="documents")

    columns = _document_columns()
    if "is_active" in columns:
        op.drop_column("documents", "is_active")
    if "academic_year" in columns:
        op.drop_column("documents", "academic_year")
    if "canonical_doc_id" in columns:
        op.drop_column("documents", "canonical_doc_id")
