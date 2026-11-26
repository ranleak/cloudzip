"""Asynchronous job registry, SSE progress pub/sub broker, and Modal Volume/Dict persistence.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import asyncio
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
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobRuntimeState] = {}

    def job_dir(self, job_id: str) -> Path:
        safe_id = "".join(ch for ch in job_id if ch.isalnum() or ch == "-")
        return self._settings.storage_dir / "jobs" / safe_id

    def create_job_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _persist_snapshot(self, state: _JobRuntimeState) -> None:
        snapshot = {
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
        try:
            jdir = self.job_dir(state.job_id)
            jdir.mkdir(parents=True, exist_ok=True)
            state_file = jdir / "state.json"
            tmp_file = jdir / "state.json.tmp"
            tmp_file.write_text(json.dumps(snapshot), encoding="utf-8")
            tmp_file.replace(state_file)
        except Exception:
            pass

        if self._modal_dict is not None:
            try:
                self._modal_dict[state.job_id] = snapshot
            except Exception:
                pass

    def _load_remote_snapshot(self, job_id: str) -> dict[str, Any] | None:
        if self._modal_dict is not None:
            try:
                if job_id in self._modal_dict:
                    return dict(self._modal_dict[job_id])
            except Exception:
                pass

        if self._modal_volume is not None:
            try:
                self._modal_volume.reload()
            except Exception:
                pass

        state_file = self.job_dir(job_id) / "state.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def register_job(
        self,
        *,
        job_id: str,
        input_path: Path,
        original_filename: str,
    ) -> _JobRuntimeState:
        self.cleanup_expired_jobs()
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
        state = _JobRuntimeState(
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
        with self._lock:
            self._jobs[job_id] = state
            self._persist_snapshot(state)
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
            self._persist_snapshot(state)

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
            if self._modal_volume is not None:
                try:
                    self._modal_volume.commit()
                except Exception:
                    pass
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
                ready = (
                    state.status == ConversionStage.COMPLETED and state.output_zip_path.exists()
                )
                return JobStatusResponse(
                    job_id=state.job_id,
                    status=state.status,
                    percentage=state.percentage,
                    message=state.message,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    download_ready=ready,
                    download_url=f"/api/v1/jobs/{job_id}/download" if ready else None,
                    result=state.result,
                    error=state.error,
                )

        remote = self._load_remote_snapshot(job_id)
        if remote is None:
            return None
        status_enum = ConversionStage(remote["status"])
        out_path = Path(remote["output_zip_path"])
        ready = status_enum == ConversionStage.COMPLETED and out_path.exists()
        res_obj = JobResultSummary.model_validate(remote["result"]) if remote.get("result") else None
        return JobStatusResponse(
            job_id=remote["job_id"],
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
            if state is not None and state.status == ConversionStage.COMPLETED:
                if state.output_zip_path.exists():
                    fname = (
                        state.result.output_filename
                        if state.result
                        else f"{state.job_id}.zip"
                    )
                    return state.output_zip_path, fname, state.result

        remote = self._load_remote_snapshot(job_id)
        if remote and remote.get("status") == ConversionStage.COMPLETED.value:
            if self._modal_volume is not None:
                try:
                    self._modal_volume.reload()
                except Exception:
                    pass
            out_path = Path(remote["output_zip_path"])
            if out_path.exists():
                res_obj = (
                    JobResultSummary.model_validate(remote["result"])
                    if remote.get("result")
                    else None
                )
                fname = res_obj.output_filename if res_obj else f"{job_id}.zip"
                return out_path, fname, res_obj
        return None

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
            snap = self._load_remote_snapshot(job_id)
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

    def delete_job(self, job_id: str) -> bool:
        existed = False
        with self._lock:
            if job_id in self._jobs:
                self._jobs.pop(job_id, None)
                existed = True
        jdir = self.job_dir(job_id)
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)
            existed = True
        if self._modal_dict is not None:
            try:
                if job_id in self._modal_dict:
                    del self._modal_dict[job_id]
                    existed = True
            except Exception:
                pass
        if existed and self._modal_volume is not None:
            try:
                self._modal_volume.commit()
            except Exception:
                pass
        return existed

    def cleanup_expired_jobs(self) -> int:
        ttl = self._settings.job_ttl_seconds
        now = time.monotonic()
        expired_ids: list[str] = []
        with self._lock:
            for jid, st in self._jobs.items():
                if st.status in TERMINAL_STAGES and (now - st.created_monotonic) > ttl:
                    expired_ids.append(jid)
        for jid in expired_ids:
            self.delete_job(jid)
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
