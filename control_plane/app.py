from __future__ import annotations

import json
from typing import Any, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .proxy import proxy_http
from .system_info import get_system_info
from .worker_manager import WorkerManager, WorkerSpec


class WorkerStartRequest(BaseModel):
    model_path: str = Field(..., description="Path to model weights without .pth extension")
    tokenizer_path: str = Field(default="reference/rwkv_vocab_v20230424.txt")
    gpu_id: Optional[int] = Field(default=None, ge=0)
    host: str = Field(default="127.0.0.1")
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    max_batch_size: int = Field(default=64, ge=1, le=4096)
    max_prefill_batch: int = Field(default=16, ge=1, le=4096)
    max_queue_size: int = Field(default=1000, ge=1, le=1000000)
    prefill_chunk_size: int = Field(default=512, ge=1, le=65536)
    use_cuda_graph: bool = Field(default=False)


def create_app(data_dir: str) -> FastAPI:
    app = FastAPI(title="Albatross Control Plane", version="0.1.0")

    manager = WorkerManager(data_dir=data_dir)
    session: Optional[aiohttp.ClientSession] = None

    @app.on_event("startup")
    async def _startup():
        nonlocal session
        session = aiohttp.ClientSession()

    @app.on_event("shutdown")
    async def _shutdown():
        if session is not None:
            await session.close()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/api/system")
    async def system() -> dict[str, Any]:
        return get_system_info()

    @app.get("/api/workers")
    async def list_workers() -> list[dict]:
        return manager.list_workers()

    @app.post("/api/workers/start")
    async def start_worker(req: WorkerStartRequest) -> dict:
        spec = WorkerSpec(
            model_path=req.model_path,
            tokenizer_path=req.tokenizer_path,
            host=req.host,
            port=req.port,
            gpu_id=req.gpu_id,
            max_batch_size=req.max_batch_size,
            max_prefill_batch=req.max_prefill_batch,
            max_queue_size=req.max_queue_size,
            prefill_chunk_size=req.prefill_chunk_size,
            use_cuda_graph=req.use_cuda_graph,
        )
        worker = manager.start_worker(spec)
        info = manager.describe_worker(worker.worker_id)
        assert info is not None
        return info

    @app.post("/api/workers/{worker_id}/stop")
    async def stop_worker(worker_id: str) -> dict:
        ok = manager.stop_worker(worker_id)
        if not ok:
            raise HTTPException(status_code=404, detail="worker not found")
        return {"status": "ok"}

    @app.get("/api/workers/{worker_id}/logs")
    async def worker_logs(worker_id: str, tail: int = 200) -> dict:
        lines = manager.get_logs(worker_id, tail=tail)
        if lines is None:
            raise HTTPException(status_code=404, detail="worker not found")
        return {"worker_id": worker_id, "lines": lines}

    @app.get("/api/workers/{worker_id}")
    async def worker_info(worker_id: str) -> dict:
        info = manager.describe_worker(worker_id)
        if info is None:
            raise HTTPException(status_code=404, detail="worker not found")
        return info

    @app.api_route("/api/workers/{worker_id}/proxy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(worker_id: str, path: str, request: Request):
        worker = manager.get_worker(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="worker not found")
        if session is None:
            raise HTTPException(status_code=503, detail="proxy not ready")
        upstream = f"http://{worker.spec.host}:{worker.spec.port}"
        return await proxy_http(client=session, upstream_base=upstream, path=path, request=request)

    return app


_INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Albatross Control Plane</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }
      h1 { margin: 0 0 16px; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-width: 320px; }
      label { display:block; font-size: 12px; color: #444; margin-top: 8px; }
      input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; font-family: inherit; }
      button { padding: 8px 10px; border: 1px solid #333; background: #111; color: #fff; border-radius: 6px; cursor: pointer; }
      button.secondary { background: #fff; color: #111; }
      pre { background: #0b1020; color: #d6e2ff; padding: 10px; border-radius: 8px; overflow: auto; max-height: 320px; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace; }
      .muted { color: #666; font-size: 12px; }
      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px; font-size: 13px; }
    </style>
  </head>
  <body>
    <h1>Albatross Control Plane</h1>
    <div class="row">
      <div class="card" style="flex: 1;">
        <h3>Start Worker</h3>
        <div class="muted">One worker = one model instance (process) bound to one GPU (optional).</div>
        <label>model_path (without .pth)</label>
        <input id="model_path" placeholder="models/RWKV-7-World-0.4B-v2.8-20241022-ctx4096" />
        <label>gpu_id (optional)</label>
        <input id="gpu_id" placeholder="0" />
        <label>max_batch_size</label>
        <input id="max_batch_size" value="64" />
        <label>port (optional)</label>
        <input id="port" placeholder="auto" />
        <div style="margin-top:10px; display:flex; gap:8px;">
          <button onclick="startWorker()">Start</button>
          <button class="secondary" onclick="refresh()">Refresh</button>
        </div>
      </div>

      <div class="card" style="flex: 2;">
        <h3>Workers</h3>
        <table>
          <thead>
            <tr><th>ID</th><th>Status</th><th>GPU</th><th>Endpoint</th><th>Model</th><th>Actions</th></tr>
          </thead>
          <tbody id="workers"></tbody>
        </table>
      </div>
    </div>

    <div class="row" style="margin-top: 16px;">
      <div class="card" style="flex: 1;">
        <h3>Logs</h3>
        <div class="muted">Select a worker to view last 200 lines.</div>
        <label>worker_id</label>
        <select id="log_worker" onchange="loadLogs()"></select>
        <pre id="log_box"></pre>
      </div>

      <div class="card" style="flex: 1;">
        <h3>Playground</h3>
        <div class="muted">Uses worker OpenAI-compatible endpoint via control-plane proxy.</div>
        <label>worker_id</label>
        <select id="pg_worker"></select>
        <label>prompt</label>
        <textarea id="prompt" rows="6">Hello, how are you?</textarea>
        <label>max_tokens</label>
        <input id="pg_max_tokens" value="128" />
        <div style="margin-top:10px; display:flex; gap:8px;">
          <button onclick="runCompletion()">Run</button>
        </div>
        <label>output</label>
        <pre id="output"></pre>
      </div>
    </div>

    <script>
      async function refresh() {
        const resp = await fetch('/api/workers');
        const workers = await resp.json();

        const tbody = document.getElementById('workers');
        tbody.innerHTML = '';

        for (const w of workers) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><code>${w.worker_id}</code></td>
            <td>${w.status}</td>
            <td>${w.spec.gpu_id ?? ''}</td>
            <td><code>${w.endpoint}</code></td>
            <td title="${w.spec.model_path}">${(w.spec.model_path || '').split('/').slice(-1)[0]}</td>
            <td>
              <button class="secondary" onclick="selectWorker('${w.worker_id}')">Select</button>
              <button onclick="stopWorker('${w.worker_id}')">Stop</button>
            </td>
          `;
          tbody.appendChild(tr);
        }

        const logSel = document.getElementById('log_worker');
        const pgSel = document.getElementById('pg_worker');
        logSel.innerHTML = '';
        pgSel.innerHTML = '';
        for (const w of workers) {
          const opt1 = document.createElement('option');
          opt1.value = w.worker_id;
          opt1.textContent = w.worker_id;
          logSel.appendChild(opt1);
          const opt2 = document.createElement('option');
          opt2.value = w.worker_id;
          opt2.textContent = w.worker_id;
          pgSel.appendChild(opt2);
        }
      }

      async function startWorker() {
        const model_path = document.getElementById('model_path').value.trim();
        const gpu_raw = document.getElementById('gpu_id').value.trim();
        const port_raw = document.getElementById('port').value.trim();
        const max_batch_size = parseInt(document.getElementById('max_batch_size').value.trim(), 10);
        const payload = {
          model_path,
          gpu_id: gpu_raw === '' ? null : parseInt(gpu_raw, 10),
          port: port_raw === '' ? null : parseInt(port_raw, 10),
          max_batch_size: Number.isFinite(max_batch_size) ? max_batch_size : 64,
        };
        const resp = await fetch('/api/workers/start', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        if (!resp.ok) {
          const txt = await resp.text();
          alert('start failed: ' + txt);
          return;
        }
        await refresh();
      }

      async function stopWorker(worker_id) {
        const resp = await fetch(`/api/workers/${worker_id}/stop`, { method: 'POST' });
        if (!resp.ok) {
          alert('stop failed');
        }
        await refresh();
      }

      function selectWorker(worker_id) {
        document.getElementById('log_worker').value = worker_id;
        document.getElementById('pg_worker').value = worker_id;
        loadLogs();
      }

      async function loadLogs() {
        const worker_id = document.getElementById('log_worker').value;
        if (!worker_id) return;
        const resp = await fetch(`/api/workers/${worker_id}/logs?tail=200`);
        const data = await resp.json();
        document.getElementById('log_box').textContent = (data.lines || []).join('\\n');
      }

      async function runCompletion() {
        const worker_id = document.getElementById('pg_worker').value;
        const prompt = document.getElementById('prompt').value;
        const max_tokens = parseInt(document.getElementById('pg_max_tokens').value.trim(), 10);
        const out = document.getElementById('output');
        out.textContent = '';
        const resp = await fetch(`/api/workers/${worker_id}/proxy/v1/completions`, {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ prompt, max_tokens: Number.isFinite(max_tokens) ? max_tokens : 128, stream: true })
        });
        if (!resp.ok) {
          out.textContent = await resp.text();
          return;
        }
        const reader = resp.body.getReader();
        const dec = new TextDecoder('utf-8');
        let buf = '';
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buf += dec.decode(value, {stream:true});
          const parts = buf.split('\\n\\n');
          buf = parts.pop();
          for (const p of parts) {
            const line = p.trim();
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (payload === '[DONE]') return;
            try {
              const obj = JSON.parse(payload);
              const text = obj?.choices?.[0]?.text ?? '';
              out.textContent += text;
            } catch (e) {}
          }
        }
      }

      refresh().then(loadLogs);
      setInterval(loadLogs, 2000);
    </script>
  </body>
</html>
"""
