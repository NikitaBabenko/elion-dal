"""Точка входа сервера VectorStore — REST API (FastAPI) + healthz.

Запуск:  python -m elion_dal.service.server

Сейчас публичный контракт — REST (HTTPS-прокси платформы пропускает HTTP, но не
gRPC). Код gRPC сохранён (proto + servicer + клиент-стаб) — может быть переключён
обратно, когда платформа научится проксировать gRPC (см. ADR-006).
"""

from __future__ import annotations

import logging
import time

import uvicorn

# --- gRPC (временно отключён, см. ADR-006) ---
# from concurrent.futures import ThreadPoolExecutor
# import grpc
# from ..grpc_gen import vectorstore_pb2 as pb
# from ..grpc_gen import vectorstore_pb2_grpc as pb_grpc
# from .servicer import VectorStoreServicer
from ..config import Settings, get_settings
from ..logging_setup import setup_logging
from .bootstrap import build_index_service
from .rest_api import create_api

logger = logging.getLogger(__name__)


def _wait_for_backends(index, settings: Settings) -> None:
    """Backoff-ретрай создания коллекции и доступности Qdrant/Postgres на старте."""
    last_err: Exception | None = None
    for attempt in range(1, settings.startup_retries + 1):
        try:
            index.qdrant.ensure_collection()
            if index.pg.ping() and index.qdrant.ping():
                return
            raise RuntimeError("Qdrant/Postgres ещё недоступны")
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                "Старт: бэкенды недоступны (попытка %d/%d): %s",
                attempt,
                settings.startup_retries,
                e,
            )
            time.sleep(settings.startup_retry_delay_s)
    raise RuntimeError(f"Не удалось подключиться к бэкендам за отведённые попытки: {last_err}")


def serve() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "backend=%s model=%s", settings.embedding_backend, settings.embedding_model or "(default)"
    )
    logger.info("Загрузка эмбеддинг-модели и инициализация хранилищ...")
    index = build_index_service(settings, ensure=False)  # модель грузим один раз
    logger.info(
        "Модель загружена: dim=%d quantized=%s", index.provider.dim, index.provider.quantized
    )
    _wait_for_backends(index, settings)

    if settings.auto_migrate:
        # Создаём схему из моделей (идемпотентно) — деплой работает без отдельного шага alembic.
        index.pg.create_all()
        index.settings_store.load()  # перечитать app_settings после создания таблицы
        logger.info("Схема БД готова (auto_migrate=create_all)")

    token_on = bool(index.settings_store.get("api_token") or settings.api_token)
    logger.info("API-токен (Bearer): %s", "включён" if token_on else "ВЫКЛ (ручки открыты)")

    # --- gRPC (закомментировано — см. ADR-006). Когда платформа научится
    #     проксировать HTTP/2/gRPC через публичный домен — снимем комментарий.
    # mb = settings.grpc_max_message_mb * 1024 * 1024
    # options = [
    #     ("grpc.max_send_message_length", mb),
    #     ("grpc.max_receive_message_length", mb),
    # ]
    # grpc_server = grpc.server(
    #     ThreadPoolExecutor(max_workers=settings.grpc_max_workers), options=options
    # )
    # pb_grpc.add_VectorStoreServicer_to_server(VectorStoreServicer(index, settings), grpc_server)
    # grpc_server.add_insecure_port(f"{settings.grpc_host}:{settings.grpc_port}")
    # grpc_server.start()
    # logger.info("gRPC слушает на %s:%d", settings.grpc_host, settings.grpc_port)

    app = create_api(index, settings)
    logger.info("REST API на http://%s:%d", settings.admin_host, settings.admin_port)
    uvicorn.run(
        app,
        host=settings.admin_host,
        port=settings.admin_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    serve()
