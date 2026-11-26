"""Asynchronous job registry, SSE progress pub/sub broker, and Modal Volume/Dict persistence.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudzip.config import CloudZipSettings
from cloudzip.extractor import ArchiveError, ArchivePasswordError
from cloudzip.models import (
    ConversionOptions,
    ConversionStage,
    JobCreatedResponse,
    JobResultSummary,
    JobStatusResponse,
    ProgressEvent,
)
from cloudzip.nvcomp_engine import NvcompCompressionEngine
from cloudzip.pipeline import run_conversion_pipeline

TERMINAL_STAGES = frozenset({ConversionStage.COMPLETED, ConversionStage.FAILED})

_OFFLOAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="cloudzip-modal-io"
)


def _current_loop() -> asyncio.AbstractEventLoop | None:
    """Return the loop running in *this* thread, if any."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def _run_modal_async(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Await a Modal handle operation without ever blocking the event loop.

    Modal's synchronized handles expose an ``.aio`` twin for every method; when it
    is available we await that so no ``AsyncUsageWarning`` is emitted. Handles that
    do not provide ``.aio`` (fakes in tests, older Modal releases) are offloaded to
    a worker thread so the loop still never blocks.
    """
    aio_fn = getattr(fn, "aio", None)
    if aio_fn is not None:
        return await aio_fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


def _run_modal_sync(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Call a blocking Modal handle method from sync code without triggering the warning.

    Modal's ``AsyncUsageWarning`` fires when a blocking interface is used while an
    event loop is running *in the calling thread*. If that is the case we hop onto a
    dedicated worker thread (which has no loop) so the call stays legal.
    """
    if _current_loop() is None:
        return fn(*args, **kwargs)
    return _OFFLOAD_POOL.submit(fn, *args, **kwargs).result()


class _ModalStateStore:
    """Thin sync/async facade over the optional Modal Dict & Volume handles.

    Worker threads use the blocking ``*`` methods; coroutines on the event loop use
    the ``*_async`` methods, which prefer Modal's ``.aio`` interfaces.
    """

    def __init__(
        self,
        *,
        dict_handle: Any | None = None,
        volume_handle: Any | None = None,
    ) -> None:
        self._dict = dict_handle
        self._volume = volume_handle

    # ------------------------------------------------------------------ Dict
    def put(self, key: str, value: dict[str, Any]) -> None:
        if self._dict is None:
            return
        try:
            _run_modal_sync(self._dict.put, key, value)
        except Exception:
            pass

    async def put_async(self, key: str, value: dict[str, Any]) -> None:
        if self._dict is None:
            return
        try:
            await _run_modal_async(self._dict.put, key, value)
        except Exception:
            pass

    def get(self, key: str) -> dict[str, Any] | None:
        if self._dict is None:
            return None
        try:
            return _run_modal_sync(self._dict.get, key)
        except Exception:
            return None

    async def get_async(self, key: str) -> dict[str, Any] | None:
        if self._dict is None:
            return None
        try:
            return await _run_modal_async(self._dict.get, key)
        except Exception:
            return None

    def discard(self, key: str) -> bool:
        """Remove `key` from the Dict; return True when an entry was removed."""
        if self._dict is None:
            return False
        try:
            if not _run_modal_sync(self._dict.contains, key):
                return False
            _run_modal_sync(self._dict.pop, key)
            return True
        except Exception:
            return False

    async def discard_async(self, key: str) -> bool:
        if self._dict is None:
            return False
        try:
            if not await _run_modal_async(self._dict.contains, key):
                return False
            await _run_modal_async(self._dict.pop, key)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- Volume
    def reload_volume(self) -> None:
        if self._volume is None:
            return
        try:
            _run_modal_sync(self._volume.reload)
        except Exception:
            pass

    async def reload_volume_async(self) -> None:
        if self._volume is None:
            return
        try:
            await _run_modal_async(self._volume.reload)
        except Exception:
            pass

    def commit_volume(self) -> None:
        if self._volume is None:
            return
        try:
            _run_modal_sync(self._volume.commit)
        except Exception:
            pass

    async def commit_volume_async(self) -> None:
        if self._volume is None:
            return
        try:
            await _run_modal_async(self._volume.commit)
        except Exception:
            pass


@dataclass
class _JobRuntimeState:
    job_id: str
    created_at: str
    updated_at: str
    created_monotonic: float
    status: ConversionStage
    percentage: float
    message: str
    input_path: Path
    output_zip_path: Path
    original_filename: str
    events: list[ProgressEvent] = field(default_factory=list)
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[ProgressEvent]]] = field(
        default_factory=list
    )
    result: JobResultSummary | None = None
    error: str | None = None


class JobManager:
    """Manages conversion jobs, thread-pool GPU execution, and live percentage streaming."""

    def __init__(
        self,
        *,
        settings: CloudZipSettings,
        engine: NvcompCompressionEngine,
        modal_volume: Any | None = None,
        modal_dict: Any | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._modal_volume = modal_volume
        self._modal_dict = modal_dict
        self._modal_store = _ModalStateStore(dict_handle=modal_dict, volume_handle=modal_volume)
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobRuntimeState] = {}

    def job_dir(self, job_id: str) -> Path:
        safe_id = "".join(ch for ch in job_id if ch.isalnum() or ch == "-")
        return self._settings.storage_dir / "jobs" / safe_id

    def create_job_id(self) -> str:
        return uuid.uuid4().hex[:16]

    @staticmethod
    def _build_snapshot(state: _JobRuntimeState) -> dict[str, Any]:
        return {
            "job_id": state.job_id,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "status": state.status.value,
            "percentage": state.percentage,
            "message": state.message,
            "output_zip_path": str(state.output_zip_path),
            "original_filename": state.original_filename,
            "events": [ev.model_dump(mode="json") for ev in state.events],
            "result": state.result.model_dump(mode="json") if state.result else None,
            "error": state.error,
        }

    def _write_local_snapshot(self, snapshot: dict[str, Any]) -> None:
        try:
            jdir = self.job_dir(snapshot["job_id"])
            jdir.mkdir(parents=True, exist_ok=True)
            state_file = jdir / "state.json"
            tmp_file = jdir / "state.json.tmp"
            tmp_file.write_text(json.dumps(snapshot), encoding="utf-8")
            tmp_file.replace(state_file)
        except Exception:
            pass

    def _persist_snapshot(self, state: _JobRuntimeState) -> None:
        """Persist `state` from a worker thread (blocking Modal calls are safe here)."""
        snapshot = self._build_snapshot(state)
        self._write_local_snapshot(snapshot)
        self._modal_store.put(state.job_id, snapshot)

    async def _persist_snapshot_async(self, state: _JobRuntimeState) -> None:
        """Persist `state` from the event loop without blocking it."""
        snapshot = self._build_snapshot(state)
        await asyncio.to_thread(self._write_local_snapshot, snapshot)
        await self._modal_store.put_async(state.job_id, snapshot)

    def _read_local_snapshot(self, job_id: str) -> dict[str, Any] | None:
        state_file = self.job_dir(job_id) / "state.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _load_remote_snapshot(self, job_id: str) -> dict[str, Any] | None:
        """Blocking snapshot lookup — only call from worker threads."""
        remote = self._modal_store.get(job_id)
        if remote is not None:
            return dict(remote)
        self._modal_store.reload_volume()
        return self._read_local_snapshot(job_id)

    async def _load_remote_snapshot_async(self, job_id: str) -> dict[str, Any] | None:
        """Snapshot lookup that is safe to await on the event loop."""
        remote = await self._modal_store.get_async(job_id)
        if remote is not None:
            return dict(remote)
        await self._modal_store.reload_volume_async()
        return await asyncio.to_thread(self._read_local_snapshot, job_id)

    def _new_state(
        self,
        *,
        job_id: str,
        input_path: Path,
        original_filename: str,
    ) -> _JobRuntimeState:
        now_iso = datetime.now(timezone.utc).isoformat()
        jdir = self.job_dir(job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        out_zip = jdir / "output.zip"

        initial_event = ProgressEvent(
            job_id=job_id,
            stage=ConversionStage.QUEUED,
            percentage=0.0,
            message=f"Queued '{original_filename}' for GPU conversion.",
            backend=self._engine.active_backend,
            timestamp=now_iso,
        )
        return _JobRuntimeState(
            job_id=job_id,
            created_at=now_iso,
            updated_at=now_iso,
            created_monotonic=time.monotonic(),
            status=ConversionStage.QUEUED,
            percentage=0.0,
            message=initial_event.message,
            input_path=input_path,
            output_zip_path=out_zip,
            original_filename=original_filename,
            events=[initial_event],
        )

    def register_job(
        self,
        *,
        job_id: str,
        input_path: Path,
        original_filename: str,
    ) -> _JobRuntimeState:
        self.cleanup_expired_jobs()
        state = self._new_state(
            job_id=job_id,
            input_path=input_path,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job_id] = state
        self._persist_snapshot(state)
        return state

    async def register_job_async(
        self,
        *,
        job_id: str,
        input_path: Path,
        original_filename: str,
    ) -> _JobRuntimeState:
        """Async twin of `register_job` that never blocks the event loop on Modal I/O."""
        await self.cleanup_expired_jobs_async()
        state = self._new_state(
            job_id=job_id,
            input_path=input_path,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job_id] = state
        await self._persist_snapshot_async(state)
        return state

    def emit_event(self, job_id: str, event: ProgressEvent) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.status = event.stage
            state.percentage = event.percentage
            state.message = event.message
            state.updated_at = event.timestamp
            if event.result is not None:
                state.result = event.result
            if event.error is not None:
                state.error = event.error
            state.events.append(event)
            subs = list(state.subscribers)
            snapshot = self._build_snapshot(state)

        self._write_local_snapshot(snapshot)
        self._modal_store.put(job_id, snapshot)

        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

    def execute_job_sync(
        self,
        *,
        job_id: str,
        options: ConversionOptions,
    ) -> JobResultSummary:
        """Run conversion synchronously on worker thread while publishing live ProgressEvents."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                raise KeyError(f"Unknown job '{job_id}'.")
            input_path = state.input_path
            output_zip_path = state.output_zip_path
            original_filename = state.original_filename

        download_url = f"/api/v1/jobs/{job_id}/download"
        try:
            summary = run_conversion_pipeline(
                job_id=job_id,
                source_archive_path=input_path,
                original_filename=original_filename,
                output_zip_path=output_zip_path,
                options=options,
                engine=self._engine,
                settings=self._settings,
                download_url=download_url,
                emit=lambda ev: self.emit_event(job_id, ev),
            )
            # Remove uploaded input archive once output ZIP is finalized to save disk space
            input_path.unlink(missing_ok=True)
            self._modal_store.commit_volume()
            return summary
        except (ArchivePasswordError, ArchiveError, ValueError) as exc:
            err_msg = str(exc)
            self.emit_event(
                job_id,
                ProgressEvent(
                    job_id=job_id,
                    stage=ConversionStage.FAILED,
                    percentage=self._current_pct(job_id),
                    message=f"Conversion failed: {err_msg}",
                    backend=self._engine.active_backend,
                    error=err_msg,
                ),
            )
            input_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            err_msg = f"Internal conversion error: {exc}"
            self.emit_event(
                job_id,
                ProgressEvent(
                    job_id=job_id,
                    stage=ConversionStage.FAILED,
                    percentage=self._current_pct(job_id),
                    message=err_msg,
                    backend=self._engine.active_backend,
                    error=err_msg,
                ),
            )
            input_path.unlink(missing_ok=True)
            raise

    def _current_pct(self, job_id: str) -> float:
        with self._lock:
            st = self._jobs.get(job_id)
            return st.percentage if st else 0.0

    def build_created_response(self, state: _JobRuntimeState) -> JobCreatedResponse:
        return JobCreatedResponse(
            job_id=state.job_id,
            status=state.status,
            status_url=f"/api/v1/jobs/{state.job_id}",
            progress_stream_url=f"/api/v1/jobs/{state.job_id}/progress",
            download_url=f"/api/v1/jobs/{state.job_id}/download",
            created_at=state.created_at,
        )

    def get_job_status(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state is not None:
            return self._status_from_state(state)

        remote = self._load_remote_snapshot(job_id)
        if remote is None:
            return None
        return self._status_from_snapshot(remote)

    async def get_job_status_async(self, job_id: str) -> JobStatusResponse | None:
        """Async twin of `get_job_status` that never blocks the event loop on Modal I/O."""
        with self._lock:
            state = self._jobs.get(job_id)
        if state is not None:
            return self._status_from_state(state)

        remote = await self._load_remote_snapshot_async(job_id)
        if remote is None:
            return None
        return self._status_from_snapshot(remote)

    def _status_from_state(self, state: _JobRuntimeState) -> JobStatusResponse:
        with self._lock:
            ready = state.status == ConversionStage.COMPLETED and state.output_zip_path.exists()
            return JobStatusResponse(
                job_id=state.job_id,
                status=state.status,
                percentage=state.percentage,
                message=state.message,
                created_at=state.created_at,
                updated_at=state.updated_at,
                download_ready=ready,
                download_url=f"/api/v1/jobs/{state.job_id}/download" if ready else None,
                result=state.result,
                error=state.error,
            )

    def _status_from_snapshot(self, remote: dict[str, Any]) -> JobStatusResponse:
        job_id = remote["job_id"]
        status_enum = ConversionStage(remote["status"])
        out_path = Path(remote["output_zip_path"])
        ready = status_enum == ConversionStage.COMPLETED and out_path.exists()
        res_obj = JobResultSummary.model_validate(remote["result"]) if remote.get("result") else None
        return JobStatusResponse(
            job_id=job_id,
            status=status_enum,
            percentage=float(remote["percentage"]),
            message=remote["message"],
            created_at=remote["created_at"],
            updated_at=remote["updated_at"],
            download_ready=ready,
            download_url=f"/api/v1/jobs/{job_id}/download" if ready else None,
            result=res_obj,
            error=remote.get("error"),
        )

    def get_download_target(self, job_id: str) -> tuple[Path, str, JobResultSummary | None] | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state is not None:
            target = self._download_target_from_state(state)
            if target is not None:
                return target

        remote = self._load_remote_snapshot(job_id)
        if remote and remote.get("status") == ConversionStage.COMPLETED.value:
            self._modal_store.reload_volume()
            return self._download_target_from_snapshot(job_id, remote)
        return None

    async def get_download_target_async(
        self, job_id: str
    ) -> tuple[Path, str, JobResultSummary | None] | None:
        """Async twin of `get_download_target` that never blocks the event loop."""
        with self._lock:
            state = self._jobs.get(job_id)
        if state is not None:
            target = self._download_target_from_state(state)
            if target is not None:
                return target

        remote = await self._load_remote_snapshot_async(job_id)
        if remote and remote.get("status") == ConversionStage.COMPLETED.value:
            await self._modal_store.reload_volume_async()
            return self._download_target_from_snapshot(job_id, remote)
        return None

    def _download_target_from_state(
        self, state: _JobRuntimeState
    ) -> tuple[Path, str, JobResultSummary | None] | None:
        with self._lock:
            if state.status != ConversionStage.COMPLETED:
                return None
            if not state.output_zip_path.exists():
                return None
            fname = state.result.output_filename if state.result else f"{state.job_id}.zip"
            return state.output_zip_path, fname, state.result

    @staticmethod
    def _download_target_from_snapshot(
        job_id: str, remote: dict[str, Any]
    ) -> tuple[Path, str, JobResultSummary | None] | None:
        out_path = Path(remote["output_zip_path"])
        if not out_path.exists():
            return None
        res_obj = (
            JobResultSummary.model_validate(remote["result"]) if remote.get("result") else None
        )
        fname = res_obj.output_filename if res_obj else f"{job_id}.zip"
        return out_path, fname, res_obj

    async def subscribe_events(self, job_id: str) -> AsyncIterator[ProgressEvent]:
        """Yield historical and live ProgressEvents for `job_id` until a terminal stage is reached."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        with self._lock:
            local_state = self._jobs.get(job_id)
            if local_state is not None:
                backlog = list(local_state.events)
                already_done = local_state.status in TERMINAL_STAGES
                if not already_done:
                    local_state.subscribers.append((loop, queue))
            else:
                backlog = []
                already_done = False

        if local_state is not None:
            try:
                for ev in backlog:
                    yield ev
                if already_done:
                    return
                while True:
                    ev = await queue.get()
                    yield ev
                    if ev.stage in TERMINAL_STAGES:
                        return
            finally:
                with self._lock:
                    if (loop, queue) in local_state.subscribers:
                        local_state.subscribers.remove((loop, queue))
            return

        # Cross-replica fallback via Modal Dict / Volume polling
        seen_count = 0
        while True:
            snap = await self._load_remote_snapshot_async(job_id)
            if snap is None:
                return
            raw_events = snap.get("events", [])
            while seen_count < len(raw_events):
                ev = ProgressEvent.model_validate(raw_events[seen_count])
                seen_count += 1
                yield ev
                if ev.stage in TERMINAL_STAGES:
                    return
            await asyncio.sleep(0.25)

    def _forget_local_job(self, job_id: str) -> bool:
        """Drop in-memory + on-disk job artifacts; return True if anything existed."""
        existed = False
        with self._lock:
            if job_id in self._jobs:
                self._jobs.pop(job_id, None)
                existed = True
        jdir = self.job_dir(job_id)
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)
            existed = True
        return existed

    def delete_job(self, job_id: str) -> bool:
        existed = self._forget_local_job(job_id)
        if self._modal_store.discard(job_id):
            existed = True
        if existed:
            self._modal_store.commit_volume()
        return existed

    async def delete_job_async(self, job_id: str) -> bool:
        """Async twin of `delete_job` that never blocks the event loop on Modal I/O."""
        existed = await asyncio.to_thread(self._forget_local_job, job_id)
        if await self._modal_store.discard_async(job_id):
            existed = True
        if existed:
            await self._modal_store.commit_volume_async()
        return existed

    def _expired_job_ids(self) -> list[str]:
        ttl = self._settings.job_ttl_seconds
        now = time.monotonic()
        with self._lock:
            return [
                jid
                for jid, st in self._jobs.items()
                if st.status in TERMINAL_STAGES and (now - st.created_monotonic) > ttl
            ]

    def cleanup_expired_jobs(self) -> int:
        expired_ids = self._expired_job_ids()
        for jid in expired_ids:
            self.delete_job(jid)
        return len(expired_ids)

    async def cleanup_expired_jobs_async(self) -> int:
        """Async twin of `cleanup_expired_jobs` that never blocks the event loop."""
        expired_ids = self._expired_job_ids()
        for jid in expired_ids:
            await self.delete_job_async(jid)
        return len(expired_ids)


def format_sse_event(event: ProgressEvent) -> str:
    """Serialize a ProgressEvent as a standards-compliant Server-Sent Event frame."""
    if event.stage == ConversionStage.COMPLETED:
        event_name = "complete"
    elif event.stage == ConversionStage.FAILED:
        event_name = "error"
    else:
        event_name = "progress"
    payload = event.model_dump_json()
    return f"event: {event_name}\ndata: {payload}\n\n"
