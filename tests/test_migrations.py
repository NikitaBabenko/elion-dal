"""Alembic smoke tests for repair-style document metadata migration."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from elion_dal.config import get_settings


@pytest.mark.parametrize("precreate_canonical", [False, True])
def test_document_metadata_migration_upgrade_and_downgrade(
    tmp_path, monkeypatch, precreate_canonical
):
    db_path = tmp_path / f"migration-{precreate_canonical}.db"
    dsn = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("PG_DSN", dsn)
    get_settings.cache_clear()
    config = Config("alembic.ini")

    try:
        command.upgrade(config, "0003_app_settings")
        engine = sa.create_engine(dsn)
        if precreate_canonical:
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN canonical_doc_id VARCHAR(256)"
                )
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO sources (source_id) VALUES ('s1')")
            connection.exec_driver_sql(
                "INSERT INTO documents (doc_id, source_id) VALUES ('existing', 's1')"
            )

        command.upgrade(config, "head")
        inspector = sa.inspect(engine)
        columns = {column["name"]: column for column in inspector.get_columns("documents")}
        indexes = {index["name"] for index in inspector.get_indexes("documents")}
        assert "canonical_doc_id" in columns
        assert "academic_year" in columns
        assert columns["academic_year"]["nullable"] is False
        assert "is_active" in columns
        assert columns["is_active"]["nullable"] is False
        assert "ix_documents_canonical_doc_id" in indexes
        with engine.connect() as connection:
            existing = connection.execute(
                sa.text("SELECT academic_year, is_active FROM documents WHERE doc_id = 'existing'")
            ).one()
        assert existing.academic_year == 0
        assert bool(existing.is_active) is True

        command.downgrade(config, "0003_app_settings")
        downgraded = {column["name"] for column in sa.inspect(engine).get_columns("documents")}
        assert "canonical_doc_id" not in downgraded
        assert "academic_year" not in downgraded
        assert "is_active" not in downgraded
    finally:
        if "engine" in locals():
            engine.dispose()
        get_settings.cache_clear()
