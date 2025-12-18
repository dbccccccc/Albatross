from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


@dataclass
class WorkerSpec:
    model_path: str
    tokenizer_path: str = "reference/rwkv_vocab_v20230424.txt"
    host: str = "127.0.0.1"
    port: Optional[int] = None
    gpu_id: Optional[int] = None

    # Performance knobs (passed through to server.main)
    max_batch_size: int = 64
    max_prefill_batch: int = 16
    max_queue_size: int = 1000
    prefill_chunk_size: int = 512
    use_cuda_graph: bool = False


@dataclass
class WorkerProcess:
    worker_id: str
    spec: WorkerSpec
    created_at: float
    process: subprocess.Popen[str]
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    exited_at: Optional[float] = None
    exit_code: Optional[int] = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None


class WorkerManager:
    def __init__(self, data_dir: str):
        self._lock = threading.RLock()
        self._workers: dict[str, WorkerProcess] = {}
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def list_workers(self) -> list[dict]:
        with self._lock:
            result: list[dict] = []
            for w in self._workers.values():
                result.append(self._worker_to_dict(w))
            return result

    def describe_worker(self, worker_id: str) -> Optional[dict]:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return None
            return self._worker_to_dict(worker)

    def get_worker(self, worker_id: str) -> Optional[WorkerProcess]:
        with self._lock:
            return self._workers.get(worker_id)

    def start_worker(self, spec: WorkerSpec) -> WorkerProcess:
        if spec.port is None:
            spec.port = _pick_free_port(spec.host)

        worker_id = f"wkr-{uuid.uuid4().hex[:10]}"

        env = os.environ.copy()
        if spec.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)

        cmd: list[str] = [
            os.environ.get("PYTHON") or sys.executable,
            "-m",
            "server.main",
            "--model-path",
            spec.model_path,
            "--tokenizer-path",
            spec.tokenizer_path,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--max-batch-size",
            str(spec.max_batch_size),
            "--max-prefill-batch",
            str(spec.max_prefill_batch),
            "--max-queue-size",
            str(spec.max_queue_size),
            "--prefill-chunk-size",
            str(spec.prefill_chunk_size),
        ]
        if spec.use_cuda_graph:
            cmd.append("--use-cuda-graph")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        worker = WorkerProcess(
            worker_id=worker_id,
            spec=spec,
            created_at=time.time(),
            process=proc,
        )

        self._start_log_reader(worker)

        with self._lock:
            self._workers[worker_id] = worker
        return worker

    def stop_worker(self, worker_id: str, timeout_s: float = 8.0) -> bool:
        worker = self.get_worker(worker_id)
        if worker is None:
            return False

        if not worker.is_running:
            return True

        try:
            if os.name == "nt":
                worker.process.terminate()
            else:
                os.kill(worker.pid, signal.SIGTERM)
        except Exception:
            pass

        try:
            worker.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                worker.process.kill()
            except Exception:
                pass

        self._finalize_exit(worker)
        return True

    def restart_worker(self, worker_id: str) -> Optional[WorkerProcess]:
        old = self.get_worker(worker_id)
        if old is None:
            return None
        spec = old.spec
        self.stop_worker(worker_id)
        return self.start_worker(spec)

    def get_logs(self, worker_id: str, tail: int = 200) -> Optional[list[str]]:
        worker = self.get_worker(worker_id)
        if worker is None:
            return None
        with self._lock:
            if tail <= 0:
                return []
            return list(worker.logs)[-tail:]

    def _start_log_reader(self, worker: WorkerProcess) -> None:
        def _reader():
            pipe = worker.process.stdout
            if pipe is None:
                return
            try:
                for line in iter(pipe.readline, ""):
                    if line == "" and worker.process.poll() is not None:
                        break
                    with self._lock:
                        worker.logs.append(line.rstrip("\n"))
            finally:
                self._finalize_exit(worker)

        t = threading.Thread(target=_reader, name=f"log-reader-{worker.worker_id}", daemon=True)
        t.start()

    def _finalize_exit(self, worker: WorkerProcess) -> None:
        if worker.exit_code is not None:
            return
        code = worker.process.poll()
        if code is None:
            return
        worker.exit_code = int(code)
        worker.exited_at = time.time()

    def _worker_to_dict(self, worker: WorkerProcess) -> dict:
        return {
            "worker_id": worker.worker_id,
            "pid": worker.pid,
            "status": "running" if worker.is_running else "exited",
            "created_at": worker.created_at,
            "exited_at": worker.exited_at,
            "exit_code": worker.exit_code,
            "endpoint": f"http://{worker.spec.host}:{worker.spec.port}",
            "spec": {
                "model_path": worker.spec.model_path,
                "tokenizer_path": worker.spec.tokenizer_path,
                "host": worker.spec.host,
                "port": worker.spec.port,
                "gpu_id": worker.spec.gpu_id,
                "max_batch_size": worker.spec.max_batch_size,
                "max_prefill_batch": worker.spec.max_prefill_batch,
                "max_queue_size": worker.spec.max_queue_size,
                "prefill_chunk_size": worker.spec.prefill_chunk_size,
                "use_cuda_graph": worker.spec.use_cuda_graph,
            },
        }
