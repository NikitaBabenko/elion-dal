"""Exercise the actual JSONL loader without a running server."""

import io
import json

from scripts import load_local


def test_loader_preserves_nested_lifecycle(tmp_path, monkeypatch):
    cases = [(156, 2025, True), (149, 2025, True), (146, 2025, True),
             (152, 2026, True), (119, None, False)]
    records = [{
        "doc_id": f"articles/{article}",
        "text": {"normalized": "Biology final date"},
        "lifecycle": {"academic_year": year, "is_active": active},
    } for article, year, active in cases]
    path = tmp_path / "ready_for_indexing.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    sent = []

    def urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{"indexed": 1}')

    monkeypatch.setattr(load_local.sys, "argv", ["load_local.py", str(path)])
    monkeypatch.setattr(load_local.urllib.request, "urlopen", urlopen)
    assert load_local.main() == 0
    assert len(sent) == len(cases)
    for payload, (article, year, active) in zip(sent, cases, strict=True):
        assert payload["doc_id"] == f"articles/{article}"
        assert payload.get("academic_year") == year
        assert payload["is_active"] is active
