"""Regression tests: Modal Dict/Volume handles must never be used blockingly on the event loop.

Modal emits `AsyncUsageWarning` whenever a blocking interface (e.g. `dict[key] = value`,
`volume.reload()`) is called while an asyncio loop is running. The fakes below reproduce
that behaviour exactly so the API/job layer is proven to use Modal's `.aio` interfaces.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import asyncio
import io
import tarfile
import threading
from typing import Any

import pytest
import zstandard as zstd
from httpx import ASGITransport, AsyncClient

from cloudzip.api import create_app
from cloudzip.config import CloudZipSettings
from cloudzip.jobs import JobManager
from cloudzip.models import ConversionStage, ProgressEvent
from cloudzip.nvcomp_engine import NvcompCompressionEngine
from tests.test_nvcomp_engine import _FakeNvcompModule


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _AioMethod:
    """Mimics Modal's `method.aio` twin."""

    def __init__(self, fn):
        self._fn = fn

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._fn, *args, **kwargs)


class _BlockingGuard:
    """Base fake that records blocking calls made from inside a running event loop."""

    def __init__(self) -> None:
        self.blocking_violations: list[str] = []
        self.aio_calls: list[str] = []

    def _record_blocking(self, name: str) -> None:
        if _running_loop():
            self.blocking_violations.append(name)

    def _wrap(self, name: str, fn):
        def sync(*args: Any, **kwargs: Any) -> Any:
            self._record_blocking(name)
            return fn(*args, **kwargs)

        def inner(*args: Any, **kwargs: Any) -> Any:
            self.aio_calls.append(name)
            return fn(*args, **kwargs)

        sync.aio = _AioMethod(inner)  # type: ignore[attr-defined]
        return sync


class _FakeModalDict(_BlockingGuard):
    def __init__(self) -> None:
        super().__init__()
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.put = self._wrap("put", self._put)
        self.get = self._wrap("get", self._get)
        self.contains = self._wrap("contains", self._contains)
        self.pop = self._wrap("pop", self._pop)

    def _put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def _get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def _contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def _pop(self, key: str) -> Any:
        with self._lock:
            return self._data.pop(key)

    # Blocking operators, exactly like Modal's synchronized `modal.Dict`.
    def __setitem__(self, key: str, value: Any) -> None:
        self._record_blocking("__setitem__")
        self._put(key, value)

    def __getitem__(self, key: str) -> Any:
        self._record_blocking("__getitem__")
        value = self._get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        self._record_blocking("__contains__")
        return self._contains(key)

    def __delitem__(self, key: str) -> None:
        self._record_blocking("__delitem__")
        self._pop(key)


class _FakeModalVolume(_BlockingGuard):
    def __init__(self) -> None:
        super().__init__()
        self.commit = self._wrap("commit", lambda: None)
        self.reload = self._wrap("reload", lambda: None)


def _settings(tmp_path) -> CloudZipSettings:
    settings = CloudZipSettings(storage_dir=tmp_path / "storage")
    settings.ensure_dirs()
    return settings


def _engine() -> NvcompCompressionEngine:
    return NvcompCompressionEngine(require_gpu=True, nvcomp_module=_FakeNvcompModule())


def _event(job_id: str, pct: float, stage: ConversionStage) -> ProgressEvent:
    return ProgressEvent(
        job_id=job_id,
        stage=stage,
        percentage=pct,
        message=f"{stage.value} {pct}",
        backend="nvcomp-cuda-gpu",
        timestamp="2026-01-01T00:00:00+00:00",
    )


async def test_job_manager_async_paths_never_block_event_loop(tmp_path) -> None:
    fake_dict = _FakeModalDict()
    fake_volume = _FakeModalVolume()
    manager = JobManager(
        settings=_settings(tmp_path),
        engine=_engine(),
        modal_volume=fake_volume,
        modal_dict=fake_dict,
    )
    job_id = "jobasync0001"

    state = await manager.register_job_async(
        job_id=job_id,
        input_path=tmp_path / "input.tar.zst",
        original_filename="input.tar.zst",
    )
    assert state.status == ConversionStage.QUEUED
    assert job_id in fake_dict._data

    manager.emit_event(job_id, _event(job_id, 42.0, ConversionStage.COMPRESSING))

    status = await manager.get_job_status_async(job_id)
    assert status is not None and status.percentage == 42.0

    # Unknown job -> remote snapshot lookup (Dict + Volume) must stay async-safe.
    assert await manager.get_job_status_async("missing") is None
    assert await manager.get_download_target_async("missing") is None

    assert await manager.cleanup_expired_jobs_async() == 0
    assert await manager.delete_job_async(job_id) is True
    assert job_id not in fake_dict._data

    assert fake_dict.blocking_violations == []
    assert fake_volume.blocking_violations == []
    assert "put" in fake_dict.aio_calls
    assert "reload" in fake_volume.aio_calls
    assert "commit" in fake_volume.aio_calls


async def test_job_manager_sync_paths_still_work_off_loop(tmp_path) -> None:
    fake_dict = _FakeModalDict()
    fake_volume = _FakeModalVolume()
    manager = JobManager(
        settings=_settings(tmp_path),
        engine=_engine(),
        modal_volume=fake_volume,
        modal_dict=fake_dict,
    )
    job_id = "jobsync0001"

    def worker() -> None:
        manager.register_job(
            job_id=job_id,
            input_path=tmp_path / "in.tar.zst",
            original_filename="in.tar.zst",
        )
        manager.emit_event(job_id, _event(job_id, 100.0, ConversionStage.COMPLETED))
        assert manager.get_job_status(job_id) is not None
        assert manager.delete_job(job_id) is True

    await asyncio.to_thread(worker)
    assert fake_dict.blocking_violations == []
    assert fake_volume.blocking_violations == []


def _build_tar_zst() -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        payload = b"# CloudZIP\n" * 40
        ti = tarfile.TarInfo(name="repo/README.md")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    return zstd.ZstdCompressor(level=3).compress(tar_buf.getvalue())


@pytest.fixture
def modal_app_with_fakes(tmp_path):
    fake_dict = _FakeModalDict()
    fake_volume = _FakeModalVolume()
    app = create_app(
        settings=_settings(tmp_path),
        engine=_engine(),
        modal_dict=fake_dict,
        modal_volume=fake_volume,
    )
    return app, fake_dict, fake_volume


async def test_api_endpoints_use_modal_aio_interfaces(modal_app_with_fakes) -> None:
    app, fake_dict, fake_volume = modal_app_with_fakes
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/jobs",
            files={"file": ("sample.tar.zst", _build_tar_zst(), "application/octet-stream")},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        status = await client.get(f"/api/v1/jobs/{job_id}")
        assert status.status_code == 200

        missing = await client.get("/api/v1/jobs/does-not-exist")
        assert missing.status_code == 404

        # Cross-replica SSE fallback polls the Dict/Volume from the event loop.
        stream = await client.get("/api/v1/jobs/does-not-exist/progress")
        assert stream.status_code == 404

        deleted = await client.delete(f"/api/v1/jobs/{job_id}")
        assert deleted.status_code == 200

    assert fake_dict.blocking_violations == []
    assert fake_volume.blocking_violations == []


# ---------------------------------------------------------------------------
# Real Modal handles: prove no `modal.AsyncUsageWarning` is ever emitted.
# ---------------------------------------------------------------------------


def _real_modal_handles():
    """Build hydrated `modal.Dict`/`modal.Volume` handles backed by stub RPC clients."""
    modal = pytest.importorskip("modal")
    from modal.dict import Dict as ModalDict
    from modal.volume import Volume as ModalVolume

    class _Stub:
        async def DictUpdate(self, req, **kwargs: Any) -> Any:
            return type("R", (), {"created": True})()

        async def DictGet(self, req, **kwargs: Any) -> Any:
            return type("R", (), {"found": False, "value": None})()

        async def DictContains(self, req, **kwargs: Any) -> Any:
            return type("R", (), {"found": False})()

        async def DictPop(self, req, **kwargs: Any) -> Any:
            return type("R", (), {"found": False})()

        async def VolumeReload(self, req, **kwargs: Any) -> Any:
            return type("R", (), {})()

        async def VolumeCommit(self, req, **kwargs: Any) -> Any:
            return type("R", (), {"skip_reload": True})()

    class _Client:
        _snapshotted = False
        _is_rehydrated = False
        _stub = _Stub()

    client = _Client()
    modal_dict = ModalDict._new_hydrated(
        "di-cloudzip-test", client, None, rep="modal.Dict.from_name('cloudzip-test')"
    )
    modal_volume = ModalVolume._new_hydrated(
        "vo-cloudzip-test", client, None, rep="modal.Volume.from_name('cloudzip-test')"
    )
    return modal, modal_dict, modal_volume


async def test_no_modal_async_usage_warning_on_event_loop(tmp_path) -> None:
    import warnings

    modal, modal_dict, modal_volume = _real_modal_handles()
    manager = JobManager(
        settings=_settings(tmp_path),
        engine=_engine(),
        modal_volume=modal_volume,
        modal_dict=modal_dict,
    )
    job_id = "realmodal0001"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await manager.register_job_async(
            job_id=job_id,
            input_path=tmp_path / "input.tar.zst",
            original_filename="input.tar.zst",
        )
        # Sync entry point invoked from coroutine code (as tests/other callers may do).
        manager.emit_event(job_id, _event(job_id, 55.0, ConversionStage.COMPRESSING))
        assert await manager.get_job_status_async(job_id) is not None
        assert await manager.get_job_status_async("missing-job") is None
        assert await manager.get_download_target_async("missing-job") is None
        assert await manager.delete_job_async(job_id) is True

    offenders = [w for w in caught if issubclass(w.category, modal.exception.AsyncUsageWarning)]
    assert offenders == [], [str(w.message) for w in offenders]
