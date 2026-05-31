"""Точка входа gRPC-сервера VectorStore.

Запуск:  python -m elion_dal.service.server

На проде поднимается только gRPC + крошечный stdlib HTTP /healthz (для health-проб
платформы). Админка отдельным процессом-клиентом — см. `elion_dal.admin.web`.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import grpc

from ..config import Settings, get_settings
from ..grpc_gen import vectorstore_pb2 as pb
from ..grpc_gen import vectorstore_pb2_grpc as pb_grpc
from ..logging_setup import setup_logging
from .bootstrap import build_index_service
from .servicer import VectorStoreServicer

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


class _HealthHandler(BaseHTTPRequestHandler):
    """Минимальный /healthz — для health-проб платформы (HTTP-роутер ждёт 200)."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_args, **_kwargs) -> None:  # тише дефолтного access-лога
        return


def _start_healthz(host: str, port: int) -> HTTPServer:
    srv = HTTPServer((host, port), _HealthHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="healthz")
    t.start()
    logger.info("HTTP /healthz слушает на %s:%d", host, port)
    return srv


def _max_workers(settings: Settings) -> int:
    # Embedded-Qdrant (:memory:/on-disk) не потокобезопасен — гоняем в 1 поток.
    if not settings.qdrant_url.startswith(("http://", "https://")):
        logger.warning(
            "Embedded-Qdrant (%s): max_workers=1 (нет потокобезопасности).", settings.qdrant_url
        )
        return 1
    return settings.grpc_max_workers


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
    logger.info("gRPC API-токен: %s", "включён" if token_on else "ВЫКЛ (ручки открыты)")

    mb = settings.grpc_max_message_mb * 1024 * 1024
    options = [
        ("grpc.max_send_message_length", mb),
        ("grpc.max_receive_message_length", mb),
    ]
    server = grpc.server(ThreadPoolExecutor(max_workers=_max_workers(settings)), options=options)
    pb_grpc.add_VectorStoreServicer_to_server(VectorStoreServicer(index, settings), server)

    # gRPC reflection (для grpcurl), если доступен модуль
    try:
        from grpc_reflection.v1alpha import reflection

        service_names = (
            pb.DESCRIPTOR.services_by_name["VectorStore"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
    except Exception:  # noqa: BLE001
        pass

    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("gRPC слушает на %s", addr)

    # Side-port HTTP /healthz (admin_host/admin_port — на проде это 8080,
    # туда смотрит платформа). Сама админка теперь — отдельный локальный процесс.
    _start_healthz(settings.admin_host, settings.admin_port)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
