"""Точка входа gRPC-сервера VectorStore.

Запуск:  python -m elion_dal.service.server
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import grpc

from ..config import get_settings
from ..grpc_gen import vectorstore_pb2 as pb
from ..grpc_gen import vectorstore_pb2_grpc as pb_grpc
from .bootstrap import build_index_service
from .servicer import VectorStoreServicer


def serve() -> None:
    settings = get_settings()
    print(f"[elion-dal] backend={settings.embedding_backend} model={settings.embedding_model}")
    print("[elion-dal] загрузка эмбеддинг-модели и инициализация хранилищ...")
    index = build_index_service(settings)

    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_VectorStoreServicer_to_server(VectorStoreServicer(index, settings), server)

    # gRPC reflection (для grpcurl), если доступен модуль
    try:
        from grpc_reflection.v1alpha import reflection

        service_names = (
            pb.DESCRIPTOR.services_by_name["VectorStore"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
    except Exception:
        pass

    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"[elion-dal] gRPC слушает на {addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
