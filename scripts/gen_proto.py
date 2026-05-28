"""Генерация gRPC-кода из proto/vectorstore.proto в src/elion_dal/grpc_gen.

Запуск:  python scripts/gen_proto.py

Чинит абсолютный импорт в *_pb2_grpc.py на относительный, чтобы пакет
импортировался как elion_dal.grpc_gen.*
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src" / "elion_dal" / "grpc_gen"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_DIR / "vectorstore.proto"),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Абсолютный импорт -> относительный
    grpc_file = OUT_DIR / "vectorstore_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    text = re.sub(
        r"^import vectorstore_pb2 as",
        "from . import vectorstore_pb2 as",
        text,
        flags=re.MULTILINE,
    )
    grpc_file.write_text(text, encoding="utf-8")
    print(f"OK -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
