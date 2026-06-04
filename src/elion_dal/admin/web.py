"""Локальная веб-админка (FastAPI) — отдельный процесс, HTTP-клиент к REST API.

Запускается на машине администратора:
    API_BASE_URL=https://elion-dal.vibenest.net API_TOKEN=... python -m elion_dal.admin.web

UI без изменений: дашборд, поиск с dense_score, удаление, загрузка PDF/DOCX
(парсится локально, отправляется через POST /api/v1/documents).

(gRPC-клиент сохранён в admin/grpc_client.py для возможного возврата — см. ADR-006.)
"""

from __future__ import annotations

# Длинные строки — это HTML/JS-шаблоны; File()/Form() в дефолтах — идиома FastAPI.
# ruff: noqa: E501, B008
import hashlib
import html
import secrets
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..ingestion.loaders import load_document
from ..service.sync import UpsertCounts
from ..store.pg_repo import DocInput, SectionInput, sha256
from ..store.settings_store import FIELDS
from .http_client import HttpAdminClient

_HEAD = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<title>Элион — DAL Admin</title>
<style>
 body{font-family:system-ui,Arial;margin:24px;max-width:1000px;color:#222}
 h1{font-size:20px} h2{font-size:16px;margin-top:28px}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}
 .cards{display:flex;gap:16px;margin:12px 0}
 .card{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:120px}
 .card b{font-size:22px;display:block}
 input,button{font-size:14px;padding:6px 8px} button{cursor:pointer}
 .hit{border:1px solid #eee;border-radius:8px;padding:10px;margin:8px 0}
 .muted{color:#888;font-size:12px}
 textarea{font-family:inherit;font-size:14px;padding:6px 8px;box-sizing:border-box}
 .chunk{border:1px solid #eee;border-left:3px solid #4a90d9;border-radius:6px;padding:8px 10px;margin:6px 0;font-size:13px;white-space:pre-wrap;word-break:break-word}
 .tok{display:inline-block;background:#eef3fb;color:#345;border-radius:10px;padding:1px 8px;font-size:11px;margin-right:6px}
 .ov{background:#fff3bf;border-radius:2px}
 .summary{margin:8px 0;font-size:14px}
 .sec{border:1px solid #eee;border-radius:8px;padding:8px 10px;margin:8px 0}
</style></head><body>
<h1>Элион — DAL Admin</h1>"""

_SCRIPT = """
<script>
async function doSearch(e){
  e.preventDefault();
  const q=document.getElementById('q').value, k=document.getElementById('k').value;
  const r=await fetch('api/search',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:`query=${encodeURIComponent(q)}&top_k=${k}`});
  const data=await r.json(); const box=document.getElementById('results'); box.innerHTML='';
  if(!data.length){box.innerHTML='<p class=muted>Ничего не найдено (no-hit).</p>';return;}
  for(const h of data){
    const el=document.createElement('div'); el.className='hit';
    el.innerHTML=`<div><b>${h.title||h.parent_id}</b> <span class=muted>score=${h.score.toFixed(4)} dense=${h.dense_score.toFixed(4)} · ${h.source_id}</span></div>`+
      (h.heading_path?`<div class=muted>${h.heading_path.join(' › ')}</div>`:'')+
      `<div class=muted>нашли по: ${h.matched_child.slice(0,160)}</div>`+
      `<div>${h.text.slice(0,400)}</div><div class=muted><a href='${h.url}'>${h.url}</a></div>`;
    box.appendChild(el);
  }
}

function esc(s){const d=document.createElement('div');d.textContent=(s==null?'':String(s));return d.innerHTML;}

// наибольший общий «суффикс a ∩ префикс b» — визуализация перекрытия соседних чанков
function overlapLen(a,b){
  const max=Math.min(a.length,b.length);
  for(let n=max;n>0;n--){ if(a.slice(a.length-n)===b.slice(0,n)) return n; }
  return 0;
}

function renderChunks(box, chunks){
  box.innerHTML='';
  for(let i=0;i<chunks.length;i++){
    const c=chunks[i], t=c.text||'';
    const idx=(c.index!=null?c.index:c.chunk_index);
    let head=(i>0)?overlapLen(chunks[i-1].text||'', t):0;
    let tail=(i<chunks.length-1)?overlapLen(t, chunks[i+1].text||''):0;
    if(head+tail>t.length){head=0;tail=0;}  // защита от наложения подсветок на коротком чанке
    const body=(head?`<span class=ov>${esc(t.slice(0,head))}</span>`:'')+
      esc(t.slice(head, t.length-tail))+
      (tail?`<span class=ov>${esc(t.slice(t.length-tail))}</span>`:'');
    const el=document.createElement('div'); el.className='chunk';
    el.innerHTML=`<span class=tok>#${idx} · ${c.token_count} ток.</span>`+body;
    box.appendChild(el);
  }
}

async function doPreview(e){
  e.preventDefault();
  const body=new URLSearchParams();
  body.set('text', document.getElementById('pv_text').value);
  const tk=document.getElementById('pv_tokens').value; if(tk) body.set('chunk_tokens',tk);
  const ov=document.getElementById('pv_overlap').value; if(ov) body.set('chunk_overlap',ov);
  const mn=document.getElementById('pv_min').value; if(mn) body.set('min_tokens',mn);
  const md=document.getElementById('pv_mode').value; if(md) body.set('separator_mode',md);
  const r=await fetch('api/chunk-preview',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body});
  const data=await r.json(); const s=data.summary||{};
  const box=document.getElementById('pv_results');
  box.innerHTML=`<div class=summary>чанков: <b>${s.count}</b> · токенов: ${s.total_tokens} · среднее: ${s.avg_tokens} · отсеяно: ${s.dropped} `+
    `<span class=muted>(tokens=${s.chunk_tokens} overlap=${s.chunk_overlap} min=${s.min_tokens} mode=${s.separator_mode})</span></div>`;
  const wrap=document.createElement('div'); box.appendChild(wrap);
  renderChunks(wrap, data.chunks||[]);
}

async function loadDocs(e){
  if(e) e.preventDefault();
  const src=document.getElementById('doc_source').value;
  const r=await fetch('api/documents'+(src?`?source_id=${encodeURIComponent(src)}`:''));
  const data=await r.json();
  document.getElementById('doc_detail').innerHTML='';
  const box=document.getElementById('doc_list');
  if(!data.length){box.innerHTML='<p class=muted>Документов нет.</p>';return;}
  let html='<table><tr><th>title</th><th>source</th><th>род.</th><th>чанки</th><th>RAG</th><th></th></tr>';
  for(const d of data){
    const rag=d.index_in_rag?(d.indexed?'✓':'pending'):'—';
    html+=`<tr><td>${esc(d.title)}</td><td>${esc(d.source_id)}</td><td>${d.parent_count}</td><td>${d.chunk_count}</td>`+
      `<td>${rag}</td><td><button onclick="showChunks('${esc(d.doc_id)}')">чанки</button></td></tr>`;
  }
  box.innerHTML=html+'</table>';
}

async function showChunks(docId){
  const r=await fetch(`api/documents/${encodeURIComponent(docId)}/detail`);
  const d=await r.json();
  const box=document.getElementById('doc_detail');
  const parents=d.parents||[];
  let html=`<h3>${esc(d.title)} <span class=muted>${esc(d.doc_id)} · ${d.indexed?'indexed':'pending'}</span></h3>`;
  for(let i=0;i<parents.length;i++){
    const p=parents[i];
    html+=`<div class=sec><b>секция ${esc(p.section_id)}</b> <span class=muted>${(p.heading_path||[]).join(' › ')} · ${p.token_count} ток. · детей: ${p.chunks.length}</span><div id=sec_${i}></div></div>`;
  }
  box.innerHTML=html;
  for(let i=0;i<parents.length;i++){ renderChunks(document.getElementById('sec_'+i), parents[i].chunks); }
}
</script></body></html>"""


def _fmt_ts(ts: int) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _doc_id(filename: str) -> str:
    return "kb-" + hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _settings_form(views) -> str:
    rows = ""
    for v in views:
        badge = " <span class=muted>(после рестарта)</span>" if v.tier == "restart" else ""
        ovr = " <span class=muted>· override</span>" if v.is_override else ""
        if v.type == "bool":
            checked = "checked" if v.value else ""
            field = f"<input type=checkbox name='{v.key}' {checked}>"
        else:
            val = "" if v.value is None else html.escape(str(v.value))
            typ = "number" if v.type in ("int", "float") else "text"
            step = " step=any" if v.type == "float" else ""
            field = f"<input type={typ}{step} name='{v.key}' value='{val}'>"
        rows += f"<tr><td>{html.escape(v.label)}{badge}{ovr}</td><td>{field}</td></tr>"
    if not rows:
        return ""
    return (
        "<h2>Настройки</h2><form method=post action='settings'>"
        "<table>" + rows + "</table>"
        "<button>Сохранить</button> "
        "<span class=muted>live применяются сразу; restart — после перезапуска сервиса</span>"
        "</form>"
    )


def _basic_auth_dependency(settings):
    """HTTP Basic из env (ADMIN_USER/ADMIN_PASSWORD). /healthz — открыт (для проб)."""
    security = HTTPBasic(auto_error=False)

    def check(request: Request, creds: HTTPBasicCredentials | None = Depends(security)) -> None:
        if request.url.path == "/healthz":
            return  # health-проба платформы без auth
        ok = (
            creds is not None
            and secrets.compare_digest(creds.username, settings.admin_user)
            and secrets.compare_digest(creds.password, settings.admin_password)
        )
        if not ok:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

    return check


def create_app(client, settings=None) -> FastAPI:
    # Basic-auth включается, только если задан пароль (иначе dev-режим без auth).
    deps = []
    if settings is not None and settings.admin_password:
        deps = [Depends(_basic_auth_dependency(settings))]
    app = FastAPI(title="Элион — DAL Admin", dependencies=deps)

    @app.get("/healthz")
    def healthz() -> dict:
        # Без auth (исключение в зависимости) — для health-проб платформы.
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        st = client.get_stats()
        src_options = "<option value=''>(все источники)</option>" + "".join(
            f"<option value='{html.escape(s.source_id)}'>{html.escape(s.source_id)}</option>"
            for s in st.sources
        )
        rows = ""
        for s in st.sources:
            rows += (
                f"<tr><td>{html.escape(s.source_id)}</td><td>{html.escape(s.name)}</td>"
                f"<td>{s.document_count}</td><td>{s.parent_count}</td><td>{s.chunk_count}</td>"
                f"<td>{_fmt_ts(s.last_indexed_ts)}</td>"
                f"<td><form method=post action='sources/{html.escape(s.source_id)}/delete' "
                f"onsubmit=\"return confirm('Удалить источник {html.escape(s.source_id)}?')\">"
                f"<button>Удалить</button></form></td></tr>"
            )
        body = f"""
        <div class=cards>
          <div class=card>документы<b>{st.total_documents}</b></div>
          <div class=card>родители<b>{st.total_parents}</b></div>
          <div class=card>чанки<b>{st.total_chunks}</b></div>
        </div>
        <h2>Источники</h2>
        <table><tr><th>source_id</th><th>имя</th><th>док.</th><th>род.</th><th>чанки</th>
          <th>синхронизация</th><th></th></tr>{rows or "<tr><td colspan=7 class=muted>пусто</td></tr>"}</table>
        <h2>Загрузить документ</h2>
        <form method=post action='upload' enctype='multipart/form-data'>
          <input type=file name=file required>
          <input type=text name=source_id value='knowledge_base' title='source_id'>
          <button>Загрузить и проиндексировать</button>
        </form>
        <h2>Поиск</h2>
        <form onsubmit='doSearch(event)'>
          <input id=q size=60 placeholder='запрос...' required>
          <input id=k type=number value=5 min=1 max=20 style='width:60px'>
          <button>Искать</button>
        </form>
        <div id=results></div>
        <h2>Превью нарезки (dry-run)</h2>
        <p class=muted>Вставьте текст и посмотрите, как он нарежется. Пустые поля = текущие настройки. Реальные данные не меняются.</p>
        <form onsubmit='doPreview(event)'>
          <textarea id=pv_text rows=6 style='width:100%' placeholder='Вставьте текст...' required></textarea>
          <div style='margin-top:6px'>
            токены <input id=pv_tokens type=number min=1 style='width:70px'>
            overlap <input id=pv_overlap type=number min=0 style='width:70px'>
            мин <input id=pv_min type=number min=0 style='width:70px'>
            режим <select id=pv_mode>
              <option value=''>(текущий)</option>
              <option value='structured'>structured</option>
              <option value='token'>token</option>
            </select>
            <button>Показать нарезку</button>
          </div>
        </form>
        <div id=pv_results></div>
        <h2>Документы и чанки</h2>
        <form onsubmit='loadDocs(event)'>
          источник <select id=doc_source>{src_options}</select>
          <button>Загрузить список</button>
        </form>
        <div id=doc_list></div>
        <div id=doc_detail></div>
        {_settings_form(client.settings_view())}
        """
        return _HEAD + body + _SCRIPT

    @app.post("/settings")
    async def update_settings(request: Request) -> RedirectResponse:
        form = await request.form()
        items: dict[str, str] = {}
        for f in FIELDS:
            if f.type == "bool":
                # снятый чекбокс не приходит в форме -> false
                items[f.key] = "true" if form.get(f.key) is not None else "false"
            else:
                val = form.get(f.key)
                if val is not None and str(val) != "":
                    items[f.key] = str(val)
        client.update_settings(items)
        return RedirectResponse(str(request.url_for("dashboard")), status_code=303)

    @app.get("/api/stats")
    def api_stats() -> dict:
        st = client.get_stats()
        return {
            "total_documents": st.total_documents,
            "total_parents": st.total_parents,
            "total_chunks": st.total_chunks,
            "sources": [asdict(s) for s in st.sources],
        }

    @app.post("/api/search")
    def api_search(query: str = Form(...), top_k: int = Form(5)) -> list[dict]:
        hits = client.search(query=query, top_k=top_k, source_ids=[], min_published_ts=0)
        return [
            {
                "parent_id": h.parent_id,
                "doc_id": h.doc_id,
                "source_id": h.source_id,
                "title": h.title,
                "url": h.url,
                "heading_path": h.heading_path,
                "text": h.text,
                "matched_child": h.matched_child,
                "score": h.score,
                "dense_score": h.dense_score,
            }
            for h in hits
        ]

    # Возвращаем результат клиента как есть (БЕЗ аннотации/response_model): в in-process
    # режиме client=IndexService отдаёт dataclass'ы, в standalone — dict'ы; jsonable_encoder
    # FastAPI корректно разворачивает оба (response_model тут только мешал бы коэрцией).
    @app.get("/api/documents")
    def api_documents(source_id: str = ""):
        return client.list_documents(source_id)

    @app.get("/api/documents/{doc_id}/detail")
    def api_document_detail(doc_id: str):
        return client.get_document_detail(doc_id)

    @app.post("/api/chunk-preview")
    def api_chunk_preview(
        text: str = Form(...),
        chunk_tokens: int | None = Form(None),
        chunk_overlap: int | None = Form(None),
        min_tokens: int | None = Form(None),
        separator_mode: str | None = Form(None),
    ):
        return client.preview_chunking(
            text=text,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            min_tokens=min_tokens,
            separator_mode=separator_mode,
        )

    def _dashboard_url(request: Request) -> str:
        # request.url_for учитывает mount-префикс (/admin/), не ломаясь при разном размещении.
        return str(request.url_for("dashboard"))

    @app.post("/sources/{source_id}/delete")
    def delete_source(source_id: str, request: Request) -> RedirectResponse:
        client.delete_source(source_id)
        return RedirectResponse(_dashboard_url(request), status_code=303)

    @app.post("/docs/{doc_id}/delete")
    def delete_doc(doc_id: str, request: Request) -> RedirectResponse:
        client.delete_doc(doc_id)
        return RedirectResponse(_dashboard_url(request), status_code=303)

    @app.post("/upload")
    def upload(request: Request, file: UploadFile = File(...), source_id: str = Form("knowledge_base")):
        data = file.file.read()
        suffix = Path(file.filename or "f").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            text = load_document(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        name = file.filename or "upload"
        url = f"file://{name}"
        doc = DocInput(
            doc_id=_doc_id(name),
            source_id=source_id,
            url=url,
            title=Path(name).stem,
            lang="ru",
            published_ts=0,
            content_hash=sha256(text),
            index_in_rag=True,
            sections=[SectionInput(section_id="0", heading_path=[], url=url, text=text)],
        )
        client.process_document(doc, UpsertCounts())
        return RedirectResponse(_dashboard_url(request), status_code=303)

    return app


def main() -> None:
    """Локальный запуск админки. Читает env через Settings (см. .env.example).

    Пример:
        API_BASE_URL=https://elion-dal.vibenest.net API_TOKEN=... \
            ADMIN_PASSWORD=secret python -m elion_dal.admin.web
    """
    import logging

    import uvicorn

    from ..config import get_settings
    from ..logging_setup import setup_logging

    settings = get_settings()
    setup_logging(settings.log_level)
    log = logging.getLogger("elion_dal.admin")
    log.info("REST API: %s", settings.api_base_url)
    client = HttpAdminClient(
        base_url=settings.api_base_url,
        token=settings.api_token,
    )
    auth = "basic-auth" if settings.admin_password else "БЕЗ auth (локально)"
    log.info("Admin UI на http://%s:%d (%s)", settings.admin_host, settings.admin_port, auth)
    uvicorn.run(
        create_app(client, settings),
        host=settings.admin_host,
        port=settings.admin_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
