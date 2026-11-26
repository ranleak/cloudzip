"""End-to-end FastAPI & SSE/NDJSON progress streaming tests for CloudZIP.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
import pyzipper
import zstandard as zstd
from httpx import ASGITransport, AsyncClient

import modal_app
from cloudzip.api import create_app
from cloudzip.config import CloudZipSettings
from tests.test_nvcomp_engine import _FakeNvcompModule
from cloudzip.nvcomp_engine import NvcompCompressionEngine


def _build_sample_tar_zst() -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        files = {
            "repo-main/README.md": b"# Sample Repo\nPowered by CloudZIP & NVIDIA nvCOMP!\n" * 60,
            "repo-main/src/engine.py": b"def accelerate():\n    return 'GPU'\n" * 120,
            "repo-main/assets/empty.txt": b"",
            "repo-main/.DS_Store": b"junk",
        }
        for path, content in files.items():
            ti = tarfile.TarInfo(name=path)
            ti.size = len(content)
            tf.addfile(ti, io.BytesIO(content))
    return zstd.ZstdCompressor(level=6).compress(tar_buf.getvalue())


@pytest.fixture
def gpu_app(tmp_path: Path):
    settings = CloudZipSettings(storage_dir=tmp_path / "storage")
    settings.ensure_dirs()
    fake_nvcomp = _FakeNvcompModule()
    engine = NvcompCompressionEngine(require_gpu=True, nvcomp_module=fake_nvcomp)
    return create_app(settings=settings, engine=engine)


async def test_health_formats_and_openapi_spec(gpu_app) -> None:
    transport = ASGITransport(app=gpu_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_health = await client.get("/api/v1/health")
        assert r_health.status_code == 200
        body = r_health.json()
        assert body["status"] == "healthy"
        assert "Alexander Scott" in body["copyright"]
        assert body["gpu"]["active_backend"] == "nvcomp-cuda-gpu"
        assert body["gpu"]["gpu_available"] is True

        r_formats = await client.get("/api/v1/formats")
        assert r_formats.status_code == 200
        f_body = r_formats.json()
        assert any(f["format_id"] == "7z" for f in f_body["input_formats"])
        assert f_body["output_compression_methods"][0]["method"] == "deflate"
        assert f_body["output_compression_methods"][0]["default"] is True

        spec = gpu_app.openapi()
        assert spec["info"]["x-copyright"] == "Copyright (c) 2026 Alexander Scott"
        assert "ConversionOptions" in spec["components"]["schemas"]
        assert "ProgressEvent" in spec["components"]["schemas"]


async def test_inspect_endpoint(gpu_app) -> None:
    archive_bytes = _build_sample_tar_zst()
    transport = ASGITransport(app=gpu_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/inspect",
            files={"file": ("project.tar.zst", archive_bytes, "application/zstd")},
        )
        assert resp.status_code == 200
        info = resp.json()
        assert info["detected_format"] == "tar.zst"
        assert info["file_count"] == 4
        assert info["single_top_level_root"] == "repo-main"


async def test_convert_stream_sse_and_download(gpu_app) -> None:
    archive_bytes = _build_sample_tar_zst()
    transport = ASGITransport(app=gpu_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        options = {
            "compression_method": "deflate",
            "compression_level": 5,
            "unwrap_single_root": True,
            "exclude_junk_files": True,
            "archive_comment": "Created by CloudZIP GPU",
        }
        events: list[tuple[str, dict]] = []
        async with client.stream(
            "POST",
            "/api/v1/convert/stream",
            files={"file": ("project.tar.zst", archive_bytes, "application/octet-stream")},
            data={"options": json.dumps(options)},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw_text = (await resp.aread()).decode("utf-8")

        for block in raw_text.strip().split("\n\n"):
            lines = [ln for ln in block.splitlines() if not ln.startswith(":")]
            if not lines:
                continue
            ev_type = "message"
            ev_data = "{}"
            for ln in lines:
                if ln.startswith("event:"):
                    ev_type = ln.split(":", 1)[1].strip()
                elif ln.startswith("data:"):
                    ev_data = ln.split(":", 1)[1].strip()
            events.append((ev_type, json.loads(ev_data)))

        percentages = [payload["percentage"] for _, payload in events]
        assert len(percentages) >= 4
        # Percentage stream MUST be strictly non-decreasing from 0% up to 100.0%!
        assert percentages == sorted(percentages)
        assert percentages[-1] == 100.0
        assert events[-1][0] == "complete"

        final_result = events[-1][1]["result"]
        assert final_result["output_filename"] == "project.zip"
        assert final_result["stats"]["backend"] == "nvcomp-cuda-gpu"
        assert final_result["stats"]["space_saved_percent"] > 50.0

        # Download the produced ZIP and extract on CPU stdlib zipfile!
        dl_resp = await client.get(final_result["download_url"])
        assert dl_resp.status_code == 200
        assert dl_resp.headers["x-cloudzip-backend"] == "nvcomp-cuda-gpu"

        with zipfile.ZipFile(io.BytesIO(dl_resp.content), "r") as zf:
            assert zf.testzip() is None
            assert zf.comment == b"Created by CloudZIP GPU"
            names = sorted(zf.namelist())
            assert names == ["README.md", "assets/empty.txt", "src/engine.py"]
            assert zf.read("assets/empty.txt") == b""
            assert b"Powered by CloudZIP" in zf.read("README.md")


async def test_async_job_lifecycle_and_direct_encrypted_convert(gpu_app) -> None:
    archive_bytes = _build_sample_tar_zst()
    transport = ASGITransport(app=gpu_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Async POST /api/v1/jobs -> GET /progress (ndjson) -> GET /download -> DELETE
        create_r = await client.post(
            "/api/v1/jobs",
            files={"file": ("dataset.tar.zst", archive_bytes, "application/zstd")},
            data={"flatten_paths": "true", "exclude_junk_files": "true"},
        )
        assert create_r.status_code == 202
        created = create_r.json()
        job_id = created["job_id"]

        ndjson_r = await client.get(f"{created['progress_stream_url']}?stream_format=ndjson")
        assert ndjson_r.status_code == 200
        nd_events = [json.loads(line) for line in ndjson_r.text.strip().splitlines()]
        assert nd_events[-1]["stage"] == "completed"
        assert nd_events[-1]["percentage"] == 100.0

        dl_r = await client.get(created["download_url"])
        assert dl_r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(dl_r.content), "r") as zf:
            assert sorted(zf.namelist()) == ["README.md", "empty.txt", "engine.py"]

        del_r = await client.delete(f"/api/v1/jobs/{job_id}")
        assert del_r.status_code == 200
        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404

        # 2. Synchronous POST /api/v1/convert with AES-256 output password
        direct_r = await client.post(
            "/api/v1/convert",
            files={"file": ("dataset.tar.zst", archive_bytes, "application/zstd")},
            data={
                "output_password": "GpuVaultPassword2026!",
                "encryption_method": "aes256",
                "unwrap_single_root": "true",
            },
        )
        assert direct_r.status_code == 200
        with pyzipper.AESZipFile(io.BytesIO(direct_r.content), "r") as azf:
            azf.setpassword(b"GpuVaultPassword2026!")
            assert b"def accelerate():" in azf.read("src/engine.py")


def test_modal_app_definition_is_valid() -> None:
    import modal

    assert modal_app.app.name == "cloudzip"
    assert isinstance(modal_app.web_api, modal.Function)
    assert modal_app.CloudZipGpuWorker is not None


async def test_nested_archive_extraction_and_error_sse_stream(gpu_app, tmp_path: Path) -> None:
    # Build an outer .tar.gz containing an inner .zip
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w", compression=zipfile.ZIP_DEFLATED) as izf:
        izf.writestr("inner_doc.txt", b"Unpacked from nested archive! " * 30)

    outer_buf = io.BytesIO()
    with tarfile.open(fileobj=outer_buf, mode="w:gz") as otf:
        raw_inner = inner_buf.getvalue()
        ti = tarfile.TarInfo(name="sub/inner.zip")
        ti.size = len(raw_inner)
        otf.addfile(ti, io.BytesIO(raw_inner))

    transport = ASGITransport(app=gpu_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.post(
            "/api/v1/convert",
            files={"file": ("outer.tar.gz", outer_buf.getvalue(), "application/gzip")},
            data={"extract_nested_archives": "true"},
        )
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.content), "r") as zf:
            assert "sub/inner_extracted/inner_doc.txt" in zf.namelist()

        # Verify SSE error event when encrypted archive lacks source_password
        enc_buf = io.BytesIO()
        with pyzipper.AESZipFile(enc_buf, "w", encryption=pyzipper.WZ_AES) as azf:
            azf.setpassword(b"NeedKey")
            azf.writestr("locked.txt", b"locked")

        async with client.stream(
            "POST",
            "/api/v1/convert/stream",
            files={"file": ("locked.zip", enc_buf.getvalue(), "application/zip")},
        ) as err_stream:
            sse_body = (await err_stream.aread()).decode("utf-8")
        assert "event: error" in sse_body
        assert "password" in sse_body.lower()
