FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сначала grpcio-tools, чтобы сгенерировать код до установки пакета.
RUN pip install --no-cache-dir grpcio-tools

COPY pyproject.toml ./
COPY proto ./proto
COPY scripts ./scripts
COPY src ./src

# Генерируем gRPC-код в src/elion_dal/grpc_gen, затем ставим пакет с этим кодом.
RUN python scripts/gen_proto.py && pip install --no-cache-dir .

EXPOSE 50051

CMD ["python", "-m", "elion_dal.service.server"]
