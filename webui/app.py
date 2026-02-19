import os
import re
import json
import time
import signal
import shutil
import subprocess
import threading
import shlex
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates

# ----------------------------
# Config
# ----------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "61771"))

# Docker/NAS 默认固定：/data/downloads
# Windows 本地默认：./downloads（避免 /data/... cwd 问题）
if os.getenv("DOWNLOAD_DIR"):
    DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR")
else:
    DOWNLOAD_DIR = "./downloads" if os.name == "nt" else "/data/downloads"

if os.getenv("TASKS_DIR"):
    TASKS_DIR = os.getenv("TASKS_DIR")
else:
    TASKS_DIR = "./tasks" if os.name == "nt" else "/data/tasks"

if os.name == "nt":
    DOWNLOAD_DIR = os.path.abspath(DOWNLOAD_DIR)
    TASKS_DIR = os.path.abspath(TASKS_DIR)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

SETTINGS_FILE = os.path.join(TASKS_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "max_concurrent": 2,          # 并发上限
    "log_retention_days": 7,      # 日志保留天数（清理 run.log）
    "task_timeout": 7200,         # 单任务超时（秒）
    "proxy_url": "",              # 全局代理（http/socks5）
}

FILES_REFRESH_SECONDS = 600      # 10分钟刷新文件缓存
CLEANUP_CHECK_SECONDS = 3600     # 1小时检查一次清理
SCHEDULER_TICK_SECONDS = 2       # 队列调度轮询间隔

PAGE_SIZE = 10                   # index 分页每页条数

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app = FastAPI(title="VideoDL WebUI", version="0.7.1")

# 最终成品常见扩展（你可按需增减）
ALLOWED_DOWNLOAD_EXTS = {
    ".mp4", ".mkv", ".flv", ".webm", ".mov", ".avi",
    ".mp3", ".m4a", ".aac", ".wav", ".flac",
    ".srt", ".ass", ".ssa", ".vtt",
}

# 明确排除的过程/临时扩展
DENY_EXTS = {".txt", ".log", ".json", ".m3u8", ".ts", ".part", ".tmp"}

# 分片文件名（关键：segment_00000001.mp4）
SEGMENT_NAME_RE = re.compile(r"^segment_\d+\.mp4$", re.IGNORECASE)


# ----------------------------
# Global state
# ----------------------------
_settings_lock = threading.Lock()
_process_lock = threading.Lock()
_running_lock = threading.Lock()

process_map: Dict[str, subprocess.Popen] = {}  # task_id -> Popen
running_tasks: set[str] = set()

_files_cache_lock = threading.Lock()
_files_cache: Dict[str, Any] = {"ts": 0, "files": []}


# ----------------------------
# Jinja filters
# ----------------------------
def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ts)))
    except Exception:
        return str(ts)


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return ""
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        v /= 1024.0
        if v < 1024.0:
            return f"{v:.2f} {u}"
    return f"{v:.2f} PiB"


templates.env.filters["fmt_ts"] = _fmt_ts
templates.env.filters["fmt_bytes"] = _fmt_bytes


# ----------------------------
# Helpers
# ----------------------------
def _now_ts() -> int:
    return int(time.time())


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._\-]+", "_", name)


def _task_path(task_id: str) -> str:
    return os.path.join(TASKS_DIR, task_id)


def _task_meta_file(task_id: str) -> str:
    return os.path.join(_task_path(task_id), "meta.json")


def _task_log_file(task_id: str) -> str:
    return os.path.join(_task_path(task_id), "run.log")


def _write_log(task_id: str, text: str):
    os.makedirs(_task_path(task_id), exist_ok=True)
    with open(_task_log_file(task_id), "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def _safe_join_under(base_dir: str, rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    full = os.path.abspath(os.path.join(base_dir, rel_path))
    base_abs = os.path.abspath(base_dir)
    if not (full == base_abs or full.startswith(base_abs + os.sep)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return full


def _is_under(base_dir: str, p: str) -> bool:
    base_abs = os.path.abspath(base_dir)
    full = os.path.abspath(p)
    return full == base_abs or full.startswith(base_abs + os.sep)


def _validate_proxy_url(proxy_url: str) -> str:
    proxy_url = (proxy_url or "").strip()
    if not proxy_url:
        return ""
    if any(ch in proxy_url for ch in ["\n", "\r", "\t", " "]):
        raise HTTPException(status_code=400, detail="Proxy URL contains whitespace")
    if len(proxy_url) > 300:
        raise HTTPException(status_code=400, detail="Proxy URL too long")
    if not re.match(r"^(http|https|socks5|socks5h)://", proxy_url, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Proxy URL must start with http(s):// or socks5(h)://")
    return proxy_url


def load_settings() -> Dict[str, Any]:
    with _settings_lock:
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_SETTINGS, f, ensure_ascii=False, indent=2)
            return dict(DEFAULT_SETTINGS)

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        merged = dict(DEFAULT_SETTINGS)
        merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})

        merged["max_concurrent"] = max(1, int(merged.get("max_concurrent", DEFAULT_SETTINGS["max_concurrent"])))
        merged["log_retention_days"] = max(1, int(merged.get("log_retention_days", DEFAULT_SETTINGS["log_retention_days"])))
        merged["task_timeout"] = max(60, int(merged.get("task_timeout", DEFAULT_SETTINGS["task_timeout"])))
        merged["proxy_url"] = str(merged.get("proxy_url", "") or "").strip()

        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        return merged


def save_settings(new_values: Dict[str, Any]) -> Dict[str, Any]:
    cur = load_settings()
    for k, v in new_values.items():
        if k in DEFAULT_SETTINGS:
            cur[k] = v

    cur["max_concurrent"] = max(1, int(cur.get("max_concurrent", DEFAULT_SETTINGS["max_concurrent"])))
    cur["log_retention_days"] = max(1, int(cur.get("log_retention_days", DEFAULT_SETTINGS["log_retention_days"])))
    cur["task_timeout"] = max(60, int(cur.get("task_timeout", DEFAULT_SETTINGS["task_timeout"])))
    cur["proxy_url"] = _validate_proxy_url(str(cur.get("proxy_url", "") or "").strip()) if str(cur.get("proxy_url", "") or "").strip() else ""

    with _settings_lock:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    return cur


def _get_videodl_version() -> str:
    # 尽量用 pip 安装包版本（最稳定）
    try:
        from importlib.metadata import version as _pkg_version  # py>=3.8
        return _pkg_version("videodl")
    except Exception:
        # fallback：尝试读取模块变量
        try:
            import videodl  # type: ignore
            v = getattr(videodl, "__version__", "") or getattr(videodl, "version", "")
            return str(v) if v else "unknown"
        except Exception:
            return "unknown"


def _paginate(items: List[Any], page: int, page_size: int) -> Dict[str, Any]:
    total = len(items)
    page_size = max(1, int(page_size))
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(int(page), total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "items": items[start:end],
    }


@dataclass
class TaskMeta:
    id: str
    url: str
    extra_args: str
    created_at: int
    use_proxy: bool = False
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    status: str = "queued"  # queued | running | success | failed | stopped
    download_dir: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def save_task(meta: TaskMeta) -> None:
    os.makedirs(_task_path(meta.id), exist_ok=True)
    with open(_task_meta_file(meta.id), "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)


def load_task(task_id: str) -> TaskMeta:
    meta_path = _task_meta_file(task_id)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Task not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskMeta(**data)


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


def refresh_files_cache(force: bool = False) -> Dict[str, Any]:
    with _files_cache_lock:
        now = _now_ts()
        if (not force) and _files_cache["ts"] and (now - _files_cache["ts"] < FILES_REFRESH_SECONDS):
            return dict(_files_cache)

    items: List[Dict[str, Any]] = []
    base = os.path.abspath(DOWNLOAD_DIR)

    for root, _, files in os.walk(base):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()

            # 只展示我们关心的成品扩展
            if ext not in ALLOWED_DOWNLOAD_EXTS:
                continue

            # 排除过程/临时扩展（双保险）
            if ext in DENY_EXTS:
                continue

            # 关键：过滤分片 mp4
            if ext == ".mp4" and SEGMENT_NAME_RE.match(fn):
                continue

            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except Exception:
                continue

            # 过滤 0 字节文件
            if st.st_size <= 0:
                continue

            rel = os.path.relpath(fp, base).replace("\\", "/")
            items.append({
                "path": rel,
                "name": fn,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
            })

    items.sort(key=lambda x: (x["mtime"], x["path"]), reverse=True)

    with _files_cache_lock:
        _files_cache["ts"] = _now_ts()
        _files_cache["files"] = items
        return dict(_files_cache)


def _get_running_count() -> int:
    with _running_lock:
        return len(running_tasks)


# ----------------------------
# Safety: extra args filtering
# ----------------------------
def _parse_extra_args_safe(extra_args: str) -> List[str]:
    r"""
    We do NOT use shell=True, so classic shell injection isn't possible.
    But users could still pass path-escape style args (e.g., output to /etc, C:\, ../..).
    We apply a conservative token filter:
    - Reject tokens containing '..' (unless it's a URL with ://)
    - Reject absolute path tokens (/xxx or \xxx)
    - Reject Windows drive path tokens like C:\...
    - Reject tokens with newlines
    """
    if not extra_args:
        return []
    if any(ch in extra_args for ch in ["\n", "\r", "\x00"]):
        raise ValueError("extra_args contains invalid control characters")

    tokens = shlex.split(extra_args, posix=(os.name != "nt"))
    cleaned: List[str] = []
    for t in tokens:
        if any(ch in t for ch in ["\n", "\r", "\x00"]):
            raise ValueError("extra_args token contains invalid control characters")

        is_flag = t.startswith("-")
        is_url = "://" in t

        if (not is_flag) and (not is_url):
            if ".." in t:
                raise ValueError("extra_args contains path traversal token '..'")
            if t.startswith("/") or t.startswith("\\"):
                raise ValueError("extra_args contains absolute path token")
            if re.match(r"^[A-Za-z]:\\", t):
                raise ValueError("extra_args contains Windows drive path token")

        cleaned.append(t)
    return cleaned


def _cleanup_temp_for_stopped_task(meta: TaskMeta) -> None:
    """
    用户手动停止任务时清理临时目录：
    删除 <task_download_dir>/videodl_outputs 整棵树（包含中间分片、hash 目录等）。
    """
    if not meta.download_dir:
        return
    dd = meta.download_dir
    if not _is_under(DOWNLOAD_DIR, dd):
        # 安全兜底：只允许删除 DOWNLOAD_DIR 下内容
        return
    temp_root = os.path.join(dd, "videodl_outputs")
    if os.path.isdir(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)


# ----------------------------
# Runner & Scheduler
# ----------------------------
def _run_task(task_id: str):
    settings = load_settings()
    timeout = int(settings["task_timeout"])
    proxy_url = str(settings.get("proxy_url", "") or "").strip()

    meta = load_task(task_id)
    meta.status = "running"
    meta.started_at = _now_ts()
    if not meta.download_dir:
        meta.download_dir = os.path.join(DOWNLOAD_DIR, meta.id)
    os.makedirs(meta.download_dir, exist_ok=True)
    save_task(meta)

    try:
        extra = _parse_extra_args_safe(meta.extra_args)

        # proxy option (per task)
        if meta.use_proxy:
            if not proxy_url:
                raise RuntimeError("Task requires proxy, but global proxy_url is empty")
            if "-p" not in extra and "--proxy" not in extra:
                extra = ["-p", proxy_url] + extra

        videodl_bin = os.getenv("VIDEODL_BIN", "videodl")
        cmd = [videodl_bin, "-i", meta.url] + extra

        _write_log(task_id, f"[webui] DOWNLOAD_DIR={DOWNLOAD_DIR}")
        _write_log(task_id, f"[webui] TASK_DOWNLOAD_DIR={meta.download_dir}")
        _write_log(task_id, f"[webui] use_proxy={meta.use_proxy}")
        if meta.use_proxy:
            _write_log(task_id, f"[webui] proxy_url={proxy_url}")
        _write_log(task_id, f"[webui] cmd={' '.join(cmd)}")
        _write_log(task_id, f"[webui] TASK_TIMEOUT={timeout}s")

        log_fp = open(_task_log_file(task_id), "a", encoding="utf-8", buffering=1)

        popen_kwargs = dict(
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=meta.download_dir,   # per-task dir avoids collisions
            text=True,
        )
        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid

        proc = subprocess.Popen(cmd, **popen_kwargs)

        with _process_lock:
            process_map[task_id] = proc

        meta.pid = proc.pid
        save_task(meta)

        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _write_log(task_id, f"[webui] TIMEOUT after {timeout}s, terminating...")
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except Exception as e:
                _write_log(task_id, f"[webui] terminate exception: {repr(e)}")
            rc = -2

        meta2 = load_task(task_id)
        meta2.returncode = int(rc)
        meta2.finished_at = _now_ts()
        if meta2.status != "stopped":
            meta2.status = "success" if rc == 0 else "failed"
        save_task(meta2)

        _write_log(task_id, f"[webui] process exited rc={rc}")

        try:
            log_fp.close()
        except Exception:
            pass

    except Exception as e:
        _write_log(task_id, f"[webui] ERROR: {repr(e)}")
        meta2 = load_task(task_id)
        meta2.status = "failed"
        meta2.finished_at = _now_ts()
        meta2.returncode = -1
        save_task(meta2)

    finally:
        with _process_lock:
            process_map.pop(task_id, None)
        with _running_lock:
            running_tasks.discard(task_id)


def _scheduler_loop():
    while True:
        try:
            settings = load_settings()
            max_conc = int(settings["max_concurrent"])

            while _get_running_count() < max_conc:
                tasks = list_tasks()
                queued = [t for t in tasks if t.status == "queued"]
                if not queued:
                    break

                queued.sort(key=lambda x: x.created_at)  # FIFO
                t = queued[0]

                with _running_lock:
                    if len(running_tasks) >= max_conc:
                        break
                    running_tasks.add(t.id)

                th = threading.Thread(target=_run_task, args=(t.id,), daemon=True)
                th.start()

            time.sleep(SCHEDULER_TICK_SECONDS)

        except Exception:
            time.sleep(SCHEDULER_TICK_SECONDS)


def _files_refresher_loop():
    while True:
        try:
            refresh_files_cache(force=True)
        except Exception:
            pass
        time.sleep(FILES_REFRESH_SECONDS)


def _cleanup_logs_loop():
    while True:
        try:
            settings = load_settings()
            days = int(settings["log_retention_days"])
            cutoff = _now_ts() - days * 86400

            tasks = list_tasks()
            for t in tasks:
                ts = t.finished_at or t.created_at
                if ts and ts < cutoff:
                    lp = _task_log_file(t.id)
                    if os.path.exists(lp):
                        try:
                            os.remove(lp)
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(CLEANUP_CHECK_SECONDS)


@app.on_event("startup")
def _startup():
    load_settings()
    refresh_files_cache(force=True)
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    threading.Thread(target=_files_refresher_loop, daemon=True).start()
    threading.Thread(target=_cleanup_logs_loop, daemon=True).start()


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    settings = load_settings()
    tasks_all = list_tasks()
    cache = refresh_files_cache(force=False)

    # pagination
    try:
        task_page = int(request.query_params.get("task_page", "1"))
    except Exception:
        task_page = 1
    try:
        file_page = int(request.query_params.get("file_page", "1"))
    except Exception:
        file_page = 1

    tasks_p = _paginate(tasks_all, task_page, PAGE_SIZE)
    files_p = _paginate(cache["files"], file_page, PAGE_SIZE)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "download_dir": DOWNLOAD_DIR,
            "tasks_dir": TASKS_DIR,

            "tasks": tasks_p["items"],
            "tasks_page": tasks_p["page"],
            "tasks_total_pages": tasks_p["total_pages"],
            "tasks_total": tasks_p["total"],

            "files": files_p["items"],
            "files_page": files_p["page"],
            "files_total_pages": files_p["total_pages"],
            "files_total": files_p["total"],

            "files_cache_ts": cache["ts"],
            "settings": settings,
            "running_count": _get_running_count(),

            # versions
            "webui_version": app.version,
            "videodl_version": _get_videodl_version(),
        },
    )


@app.post("/tasks")
def create_task(
    url: str = Form(...),
    extra_args: str = Form(""),
    use_proxy: Optional[str] = Form(None),  # checkbox -> "on"
):
    task_id = _safe_id(f"t{_now_ts()}{os.getpid()}")
    meta = TaskMeta(
        id=task_id,
        url=url.strip(),
        extra_args=(extra_args or "").strip(),
        created_at=_now_ts(),
        status="queued",
        use_proxy=(use_proxy == "on"),
        download_dir=os.path.join(DOWNLOAD_DIR, task_id),
    )
    save_task(meta)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(request: Request, task_id: str):
    meta = load_task(task_id)
    log_text = ""
    lp = _task_log_file(task_id)
    if os.path.exists(lp):
        try:
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()[-20000:]
        except Exception:
            log_text = "(failed to read log)"
    return templates.TemplateResponse("task.html", {"request": request, "task": meta, "log_text": log_text})


@app.get("/tasks/{task_id}/log", response_class=PlainTextResponse)
def task_log(task_id: str):
    lp = _task_log_file(task_id)
    if not os.path.exists(lp):
        return PlainTextResponse("", status_code=200, headers={"Cache-Control": "no-store"})
    with open(lp, "r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse(f.read(), headers={"Cache-Control": "no-store"})


@app.get("/tasks/{task_id}/meta")
def task_meta(task_id: str):
    meta = load_task(task_id)
    return meta.to_dict()


@app.post("/tasks/{task_id}/stop")
def stop_task(task_id: str):
    meta = load_task(task_id)
    if meta.status not in ("running", "queued"):
        return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)

    # queued -> 直接标 stopped + 清临时（一般为空）
    if meta.status == "queued":
        meta.status = "stopped"
        meta.finished_at = _now_ts()
        save_task(meta)
        with _running_lock:
            running_tasks.discard(task_id)
        # 清理临时
        _cleanup_temp_for_stopped_task(meta)
        refresh_files_cache(force=True)
        return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)

    # running -> 先停进程
    with _process_lock:
        proc = process_map.get(task_id)
    if proc:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass

    meta.status = "stopped"
    meta.finished_at = _now_ts()
    save_task(meta)

    # ✅ 关键：手动停止后自动清理临时目录（videodl_outputs/**）
    _cleanup_temp_for_stopped_task(meta)
    refresh_files_cache(force=True)

    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/{task_id}/delete")
def delete_task(task_id: str, delete_downloads: str = Form("0")):
    """
    delete_downloads=1 时：删除整个任务下载目录（如 DOWNLOAD_DIR/<task_id>/）包含目录本身。
    """
    try:
        meta = load_task(task_id)
    except Exception:
        meta = None

    if meta and meta.status == "running":
        with _process_lock:
            proc = process_map.get(task_id)
        if proc:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass

    with _running_lock:
        running_tasks.discard(task_id)

    # 删除任务记录目录（tasks/<task_id>/）
    tp = _task_path(task_id)
    if os.path.isdir(tp):
        shutil.rmtree(tp, ignore_errors=True)

    # 删除任务下载目录（downloads/<task_id>/）
    if str(delete_downloads).strip() == "1":
        dd = (meta.download_dir if (meta and meta.download_dir) else os.path.join(DOWNLOAD_DIR, task_id))
        if _is_under(DOWNLOAD_DIR, dd) and os.path.isdir(dd):
            shutil.rmtree(dd, ignore_errors=True)
        refresh_files_cache(force=True)

    return RedirectResponse(url="/", status_code=303)


@app.post("/settings")
def update_settings(
    max_concurrent: int = Form(...),
    log_retention_days: int = Form(...),
    task_timeout: int = Form(...),
    proxy_url: str = Form(""),
):
    proxy_url = (proxy_url or "").strip()
    if proxy_url:
        proxy_url = _validate_proxy_url(proxy_url)
    save_settings({
        "max_concurrent": int(max_concurrent),
        "log_retention_days": int(log_retention_days),
        "task_timeout": int(task_timeout),
        "proxy_url": proxy_url,
    })
    return RedirectResponse(url="/", status_code=303)


@app.get("/files")
def list_files_api():
    cache = refresh_files_cache(force=False)
    return {"download_dir": DOWNLOAD_DIR, "ts": cache["ts"], "files": cache["files"]}


@app.post("/files/refresh")
def refresh_files():
    cache = refresh_files_cache(force=True)
    return {"ok": True, "ts": cache["ts"], "count": len(cache["files"])}


@app.get("/files/{filepath:path}")
def download_file(filepath: str):
    fp = _safe_join_under(DOWNLOAD_DIR, filepath)
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp, filename=os.path.basename(fp))


@app.post("/files/{filepath:path}/delete")
def delete_file(filepath: str):
    fp = _safe_join_under(DOWNLOAD_DIR, filepath)
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(fp)
    refresh_files_cache(force=True)
    return RedirectResponse(url="/", status_code=303)


@app.get("/health")
def health():
    settings = load_settings()
    return {
        "ok": True,
        "download_dir": DOWNLOAD_DIR,
        "tasks_dir": TASKS_DIR,
        "settings": settings,
        "running": _get_running_count(),
        "webui_version": app.version,
        "videodl_version": _get_videodl_version(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
