import os
import re
import json
import time
import signal
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates

# ----------------------------
# Config
# ----------------------------
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/data/downloads")
TASKS_DIR = os.getenv("TASKS_DIR", "/data/tasks")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app = FastAPI(title="VideoDL WebUI", version="0.1.0")


# ----------------------------
# Helpers
# ----------------------------
def _safe_filename(name: str) -> str:
    # prevent path traversal; keep simple safe chars
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._\- ()\[\]]+", "_", name)
    return name or "file"


def _task_path(task_id: str) -> str:
    return os.path.join(TASKS_DIR, task_id)


def _task_meta_file(task_id: str) -> str:
    return os.path.join(_task_path(task_id), "meta.json")


def _task_log_file(task_id: str) -> str:
    return os.path.join(_task_path(task_id), "run.log")


def _now_ts() -> int:
    return int(time.time())


@dataclass
class TaskMeta:
    id: str
    url: str
    extra_args: str
    created_at: int
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    status: str = "queued"  # queued | running | success | failed | stopped

    def to_dict(self):
        return asdict(self)


def load_task(task_id: str) -> TaskMeta:
    meta_path = _task_meta_file(task_id)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Task not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskMeta(**data)


def save_task(meta: TaskMeta) -> None:
    os.makedirs(_task_path(meta.id), exist_ok=True)
    with open(_task_meta_file(meta.id), "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)


def list_tasks() -> List[TaskMeta]:
    out: List[TaskMeta] = []
    for name in sorted(os.listdir(TASKS_DIR), reverse=True):
        p = os.path.join(TASKS_DIR, name)
        if not os.path.isdir(p):
            continue
        meta_path = os.path.join(p, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            out.append(load_task(name))
        except Exception:
            continue
    return out


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _spawn_download(task_id: str, url: str, extra_args: str):
    """
    Spawn: videodl -i URL [extra args...]
    We keep it simple: extra_args is a raw string appended to command (shell=False).
    """
    meta = load_task(task_id)
    meta.status = "running"
    meta.started_at = _now_ts()
    save_task(meta)

    log_path = _task_log_file(task_id)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)

    # Basic command: videodl -i "URL"
    # User extra args (optional) are split by shell-like rules (but minimal).
    # If you need complex quoting, extend this with shlex.split.
    import shlex
    extra = shlex.split(extra_args) if extra_args else []

    cmd = ["videodl", "-i", url] + extra

    # Run in its own process group so we can stop it later.
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=DOWNLOAD_DIR,  # many tools output relative files; keep everything under download dir
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        text=True,
    )

    meta.pid = proc.pid
    save_task(meta)

    # Wait in a detached helper process (still inside server process).
    # For small NAS usage, this is OK. For heavy usage, move to a queue worker.
    def _waiter():
        rc = proc.wait()
        meta2 = load_task(task_id)
        meta2.returncode = rc
        meta2.finished_at = _now_ts()
        if meta2.status == "stopped":
            # keep stopped
            pass
        else:
            meta2.status = "success" if rc == 0 else "failed"
        save_task(meta2)
        try:
            log_f.close()
        except Exception:
            pass

    import threading
    t = threading.Thread(target=_waiter, daemon=True)
    t.start()


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    tasks = list_tasks()
    files = []
    try:
        for name in sorted(os.listdir(DOWNLOAD_DIR)):
            fp = os.path.join(DOWNLOAD_DIR, name)
            if os.path.isfile(fp):
                files.append({
                    "name": name,
                    "size": os.path.getsize(fp),
                    "mtime": int(os.path.getmtime(fp)),
                })
    except FileNotFoundError:
        pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "download_dir": DOWNLOAD_DIR,
            "tasks": tasks,
            "files": files,
        },
    )


@app.post("/tasks")
def create_task(
    url: str = Form(...),
    extra_args: str = Form(""),
):
    task_id = f"t{_now_ts()}{os.getpid()}"
    meta = TaskMeta(
        id=task_id,
        url=url.strip(),
        extra_args=extra_args.strip(),
        created_at=_now_ts(),
        status="queued",
    )
    save_task(meta)

    # spawn immediately (simple single-server usage)
    _spawn_download(task_id, meta.url, meta.extra_args)

    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(request: Request, task_id: str):
    meta = load_task(task_id)

    # If marked running but pid is dead, refresh status by returncode if possible
    if meta.status == "running" and meta.pid and not _is_pid_alive(meta.pid):
        # process ended but waiter might not have saved yet; leave as-is
        pass

    log_text = ""
    log_path = _task_log_file(task_id)
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                # show tail
                log_text = f.read()[-20000:]
        except Exception:
            log_text = "(failed to read log)"

    return templates.TemplateResponse(
        "task.html",
        {"request": request, "task": meta, "log_text": log_text},
    )


@app.get("/tasks/{task_id}/log", response_class=PlainTextResponse)
def task_log(task_id: str):
    log_path = _task_log_file(task_id)
    if not os.path.exists(log_path):
        return PlainTextResponse("", status_code=200)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse(f.read())


@app.post("/tasks/{task_id}/stop")
def stop_task(task_id: str):
    meta = load_task(task_id)
    if meta.status not in ("running", "queued"):
        return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)

    if meta.pid:
        try:
            # kill process group if possible
            if hasattr(os, "killpg"):
                os.killpg(meta.pid, signal.SIGTERM)
            else:
                os.kill(meta.pid, signal.SIGTERM)
        except Exception:
            pass

    meta.status = "stopped"
    meta.finished_at = _now_ts()
    save_task(meta)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.get("/files")
def list_files_api():
    items = []
    for name in sorted(os.listdir(DOWNLOAD_DIR)):
        fp = os.path.join(DOWNLOAD_DIR, name)
        if os.path.isfile(fp):
            items.append({"name": name, "size": os.path.getsize(fp), "mtime": int(os.path.getmtime(fp))})
    return {"download_dir": DOWNLOAD_DIR, "files": items}


@app.get("/files/{filename}")
def download_file(filename: str):
    safe = _safe_filename(filename)
    fp = os.path.join(DOWNLOAD_DIR, safe)
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp, filename=safe)


@app.post("/files/{filename}/delete")
def delete_file(filename: str):
    safe = _safe_filename(filename)
    fp = os.path.join(DOWNLOAD_DIR, safe)
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(fp)
    return RedirectResponse(url="/", status_code=303)


@app.get("/health")
def health():
    # basic health check
    return {"ok": True, "download_dir": DOWNLOAD_DIR}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
