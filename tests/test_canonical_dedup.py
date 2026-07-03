"""Тесты дедупликации по canonical_doc_id.

Проверяем, что:
1. canonical_doc_id сохраняется при загрузке документа
2. При повторной загрузке с новым content_hash старые чанки удаляются, новые создаются
3. При повторной загрузке с тем же content_hash документ пропускается (skip)
4. После тестов все данные удаляются
"""

import os

os.environ["PG_DSN"] = "sqlite:///./elion_dev.db"
os.environ["QDRANT_URL"] = "./qdrant_local"

from elion_dal.service.bootstrap import build_index_service
from elion_dal.service.sync import UpsertCounts
from elion_dal.store.pg_repo import DocInput, PgRepo, SectionInput


def ensure_db():
    """Create database tables if they don't exist."""
    repo = PgRepo(os.environ["PG_DSN"])
    repo.create_all()
    return repo


def create_test_doc(doc_id: str, canonical_id: str, content_hash: str, text: str):
    """Создать тестовый документ."""
    return DocInput(
        doc_id=doc_id,
        source_id="test_canonical",
        url="https://example.com",
        title="Test document",
        lang="ru",
        published_ts=0,
        content_hash=content_hash,
        index_in_rag=True,
        academic_year=2026,
        is_active=True,
        canonical_doc_id=canonical_id,
        sections=[
            SectionInput(
                section_id="0",
                heading_path=[],
                url="https://example.com",
                text=text,
            )
        ],
    )


def test_canonical_doc_id_preserved():
    """Тест 1: Проверить, что canonical_doc_id сохраняется."""
    print("\n=== Тест 1: canonical_doc_id сохраняется ===")
    ensure_db()
    index = build_index_service(ensure=False)
    doc_id = "test-doc-v1"
    canonical_id = "test-doc-canonical"

    try:
        doc = create_test_doc(
            doc_id=doc_id,
            canonical_id=canonical_id,
            content_hash="hash1",
            text="Test document for canonical_doc_id test.",
        )
        counts = UpsertCounts()
        index.process_document(doc, counts)
        assert counts.indexed == 1, "Документ должен быть проиндексирован"

        # Проверяем, что canonical_doc_id сохранился
        pg = PgRepo(os.environ["PG_DSN"])
        saved_doc_id = pg.get_doc_id_by_canonical(canonical_id)
        assert saved_doc_id == doc_id, f"Ожидался doc_id '{doc_id}', получен '{saved_doc_id}'"

        print(f" canonical_doc_id '{canonical_id}' сохранён с doc_id '{doc_id}'")

    except AssertionError as e:
        print(f" Тест провален: {e}")
        return False
    finally:
        # Очистка
        try:
            index.delete_doc(doc_id)
            print(f"   Тестовый документ {doc_id} удалён")
        except Exception as e:
            print(f"   Не удалось удалить {doc_id}: {e}")


def test_dedup_by_canonical_id():
    """Тест 2: Проверить дедупликацию по canonical_doc_id."""
    print("\n=== Тест 2: Дедупликация по canonical_doc_id ===")
    ensure_db()
    index = build_index_service(ensure=False)
    canonical_id = "test-doc-canonical-dedup"
    doc_id_v1 = "test-doc-v1"
    doc_id_v2 = "test-doc-v2"

    try:
        # Первая загрузка
        doc1 = create_test_doc(
            doc_id=doc_id_v1,
            canonical_id=canonical_id,
            content_hash="hash1",
            text="First version of document.",
        )
        counts1 = UpsertCounts()
        index.process_document(doc1, counts1)
        assert counts1.indexed == 1, "Первая версия должна быть проиндексирована"

        # Проверяем, что документ создан
        pg = PgRepo(os.environ["PG_DSN"])
        saved_doc_id = pg.get_doc_id_by_canonical(canonical_id)
        assert saved_doc_id == doc_id_v1, "Должен быть doc_id_v1"

        print(f"   Первая загрузка: doc_id='{doc_id_v1}', hash='hash1'")

        # Вторая загрузка с новым content_hash и новым doc_id
        doc2 = create_test_doc(
            doc_id=doc_id_v2,
            canonical_id=canonical_id,
            content_hash="hash2",
            text="Second version of document with updated content.",
        )
        counts2 = UpsertCounts()
        index.process_document(doc2, counts2)
        assert counts2.indexed == 1, "Вторая версия должна быть проиндексирована"

        # Проверяем, что теперь doc_id_v2
        saved_doc_id = pg.get_doc_id_by_canonical(canonical_id)
        assert saved_doc_id == doc_id_v2, f"Должен быть doc_id_v2, получен '{saved_doc_id}'"

        # Проверяем, что старый документ удалён (chunks нет)
        # Проверяем через get_document_detail
        detail = index.get_document_detail(doc_id_v1)
        assert detail is None or detail.parents == [], "Старый документ должен быть удалён"

        print(f"   Вторая загрузка: doc_id='{doc_id_v2}', hash='hash2' (update)")

        # Проверяем, что новые чанки созданы
        detail_v2 = index.get_document_detail(doc_id_v2)
        assert detail_v2 is not None, "Новый документ должен существовать"
        print("Новый документ создан, старый удалён")

    except AssertionError as e:
        print(f" Тест провален: {e}")
        return False
    finally:
        # Очистка
        try:
            index.delete_doc(doc_id_v2)
            print(f"   Тестовый документ {doc_id_v2} удалён")
        except Exception as e:
            print(f"   Не удалось удалить {doc_id_v2}: {e}")


def test_skip_on_unchanged_hash():
    """Тест 3: Проверить skip при неизменном content_hash."""
    print("\n=== Тест 3: Skip при неизменном content_hash ===")
    ensure_db()
    index = build_index_service(ensure=False)
    canonical_id = "test-doc-canonical-skip"
    doc_id_v1 = "test-doc-v1"
    doc_id_v2 = "test-doc-v2"

    try:
        # Первая загрузка
        doc1 = create_test_doc(
            doc_id=doc_id_v1,
            canonical_id=canonical_id,
            content_hash="hash1",
            text="First version.",
        )
        counts1 = UpsertCounts()
        index.process_document(doc1, counts1)
        assert counts1.indexed == 1, "Первая версия должна быть проиндексирована"

        print(f"   Первая загрузка: doc_id='{doc_id_v1}', hash='hash1' -> indexed")

        # Вторая загрузка с тем же content_hash, но новым doc_id
        doc2 = create_test_doc(
            doc_id=doc_id_v2,
            canonical_id=canonical_id,
            content_hash="hash1",  # тот же хеш!
            text="First version." * 2,  # текст другой, но хеш тот же (специально)
        )
        counts2 = UpsertCounts()
        index.process_document(doc2, counts2)

        print(
            f"   Вторая загрузка: doc_id='{doc_id_v2}', hash='hash1'"
            f" -> skipped={counts2.skipped}"
        )

        # Должен быть пропущен (skip)
        assert counts2.skipped == 1, "Документ с тем же хешем должен быть пропущен"

        # Проверяем, что doc_id остался старый
        pg = PgRepo(os.environ["PG_DSN"])
        saved_doc_id = pg.get_doc_id_by_canonical(canonical_id)
        assert saved_doc_id == doc_id_v1, f"Должен быть doc_id_v1, получен '{saved_doc_id}'"

        print(f"    Документ пропущен, doc_id остался '{doc_id_v1}'")
        
    except AssertionError as e:
        print(f" Тест провален: {e}")
        return False
    finally:
        # Очистка
        try:
            # Удаляем оба возможных doc_id
            for doc_id in [doc_id_v1, doc_id_v2]:
                try:
                    index.delete_doc(doc_id)
                    print(f"   Тестовый документ {doc_id} удалён")
                except Exception:
                    pass
        except Exception as e:
            print(f"   Не удалось удалить тестовые документы: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТЫ ДЕДУПЛИКАЦИИ ПО canonical_doc_id")
    print("=" * 60)

    results = []
    results.append(("canonical_doc_id сохраняется", test_canonical_doc_id_preserved()))
    results.append(("дедупликация по canonical_id", test_dedup_by_canonical_id()))
    results.append(("skip при неизменном хеше", test_skip_on_unchanged_hash()))

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("=" * 60)
    for name, passed in results:
        status = " PASSED" if passed else " FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n Все тесты пройдены!")
    else:
        print("\n Некоторые тесты не пройдены.")