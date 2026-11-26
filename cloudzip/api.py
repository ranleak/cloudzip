"""FastAPI application for CloudZIP — GPU-accelerated archive manipulation & ZIP conversion.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from cloudzip import __copyright__, __version__
from cloudzip.config import CloudZipSettings, _is_modal_runtime, get_settings
from cloudzip.extractor import (
    SUPPORTED_INPUT_FORMATS,
    ArchiveError,
    ArchivePasswordError,
    inspect_archive,
)
from cloudzip.jobs import JobManager, format_sse_event
from cloudzip.models import (
    ArchiveInspectionResponse,
    CompressionMethod,
    ConversionOptions,
    ConversionStage,
    EncryptionMethod,
    HealthResponse,
    JobCreatedResponse,
    JobStatusResponse,
    ProgressEvent,
    RenameRule,
    SortOrder,
    SupportedFormatsResponse,
    Zip64Mode,
)
from cloudzip.nvcomp_engine import NvcompCompressionEngine

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

EXPOSED_HEADERS = [
    "Content-Disposition",
    "Content-Length",
    "X-CloudZip-Job-Id",
    "X-CloudZip-Backend",
    "X-CloudZip-Input-Format",
    "X-CloudZip-Compression-Method",
    "X-CloudZip-Space-Saved-Percent",
    "X-CloudZip-SHA256",
]


async def _save_upload_with_limit(
    upload: UploadFile,
    dest_path: Path,
    *,
    max_bytes: int,
) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    chunk_size = 1024 * 1024
    try:
        with open(dest_path, "wb") as out_fp:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed size of {max_bytes:,} bytes.",
                    )
                out_fp.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if total == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded archive file is empty (0 bytes).")
    return total


def _parse_conversion_options(
    options_json: str | None,
    *,
    compression_method: CompressionMethod | None = None,
    compression_level: int | None = None,
    smart_store: bool | None = None,
    flatten_paths: bool | None = None,
    strip_components: int | None = None,
    unwrap_single_root: bool | None = None,
    root_prefix: str | None = None,
    exclude_junk_files: bool | None = None,
    exclude_hidden_files: bool | None = None,
    deduplicate_identical: bool | None = None,
    extract_nested_archives: bool | None = None,
    source_password: str | None = None,
    output_password: str | None = None,
    encryption_method: EncryptionMethod | None = None,
    normalize_timestamps: bool | None = None,
    zip64_mode: Zip64Mode | None = None,
    sort_entries: SortOrder | None = None,
    archive_comment: str | None = None,
    output_filename: str | None = None,
) -> ConversionOptions:
    payload: dict[str, Any] = {}
    if options_json and options_json.strip():
        try:
            decoded = json.loads(options_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON in 'options' form field: {exc}",
            ) from exc
        if not isinstance(decoded, dict):
            raise HTTPException(
                status_code=422,
                detail="'options' form field must decode to a JSON object.",
            )
        payload.update(decoded)

    overrides = {
        "compression_method": compression_method,
        "compression_level": compression_level,
        "smart_store": smart_store,
        "flatten_paths": flatten_paths,
        "strip_components": strip_components,
        "unwrap_single_root": unwrap_single_root,
        "root_prefix": root_prefix,
        "exclude_junk_files": exclude_junk_files,
        "exclude_hidden_files": exclude_hidden_files,
        "deduplicate_identical": deduplicate_identical,
        "extract_nested_archives": extract_nested_archives,
        "source_password": source_password,
        "output_password": output_password,
        "encryption_method": encryption_method,
        "normalize_timestamps": normalize_timestamps,
        "zip64_mode": zip64_mode,
        "sort_entries": sort_entries,
        "archive_comment": archive_comment,
        "output_filename": output_filename,
    }
    for key, val in overrides.items():
        if val is not None:
            payload[key] = val

    try:
        return ConversionOptions.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def create_app(
    *,
    settings: CloudZipSettings | None = None,
    engine: NvcompCompressionEngine | None = None,
    modal_volume: Any | None = None,
    modal_dict: Any | None = None,
) -> FastAPI:
    """Construct and configure the CloudZIP FastAPI application."""
    cfg = settings or get_settings()
    comp_engine = engine or NvcompCompressionEngine(require_gpu=cfg.require_gpu)
    job_manager = JobManager(
        settings=cfg,
        engine=comp_engine,
        modal_volume=modal_volume,
        modal_dict=modal_dict,
    )
    worker_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cloudzip-gpu")

    app = FastAPI(
        title="CloudZIP GPU Archive Manipulation API",
        version=__version__,
        summary=(
            "Modal.com cloud service for universal archive extraction, path/content manipulation, "
            "and NVIDIA nvCOMP GPU-accelerated ZIP compression with real-time percentage streaming."
        ),
        description=(
            "Upload an archive of virtually any format (`.7z`, `.rar`, `.tar.gz`, `.tar.bz2`, "
            "`.tar.xz`, `.tar.zst`, `.tar.lz4`, `.tar.br`, `.zip`, `.iso`, `.cab`, `.deb`, "
            "`.rpm`, `.cpio`, `.xar`, `.wim`, `.gz`, `.bz2`, `.xz`, `.zst`, `.lz4`, `.br`, `.snappy`) "
            "and convert it into a universally compatible `.zip` archive using **NVIDIA nvCOMP** "
            "GPU acceleration (`Deflate` RFC 1951 `algorithm_type=5` `BitstreamKind.RAW`, "
            "ZIP Method 8 — readable on every CPU-only client device worldwide).\n\n"
            f"**{__copyright__}. All rights reserved.**"
        ),
        contact={"name": "Alexander Scott"},
        license_info={"name": "MIT License", "identifier": "MIT"},
    )

    app.state.settings = cfg
    app.state.engine = comp_engine
    app.state.job_manager = job_manager
    app.state.worker_pool = worker_pool

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )

    @app.get(
        "/",
        response_model=HealthResponse,
        tags=["System"],
        summary="Service root & GPU telemetry",
    )
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check & NVIDIA nvCOMP capability probe",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service="CloudZIP",
            version=__version__,
            copyright=__copyright__,
            modal_runtime=_is_modal_runtime(),
            gpu=comp_engine.capabilities(),
        )

    @app.get(
        "/api/v1/formats",
        response_model=SupportedFormatsResponse,
        tags=["Metadata"],
        summary="List supported input archive formats and GPU output compression codecs",
    )
    async def list_formats() -> SupportedFormatsResponse:
        return SupportedFormatsResponse(
            copyright=__copyright__,
            input_formats=SUPPORTED_INPUT_FORMATS,
            output_compression_methods=[
                {
                    "method": "deflate",
                    "zip_method_id": 8,
                    "nvcomp_algorithm": "Deflate (algorithm_type=5, BitstreamKind.RAW)",
                    "cpu_client_compatibility": "Universal (100% — Windows, macOS, iOS, Android, Linux, browsers)",
                    "default": True,
                    "description": (
                        "Best compressed size among universally compatible ZIP codecs using "
                        "NVIDIA nvCOMP GPU Deflate level 5 + Smart Store Method 0 fallback."
                    ),
                },
                {
                    "method": "zstd",
                    "zip_method_id": 93,
                    "nvcomp_algorithm": "Zstd (BitstreamKind.RAW, parallel multi-frame)",
                    "cpu_client_compatibility": "Modern CPU tools (7-Zip, WinRAR, libarchive/bsdtar, PeaZip, Windows 11)",
                    "default": False,
                    "description": (
                        "High-ratio GPU Zstandard compression per PKWARE APPNOTE 6.3.8+ (Method 93)."
                    ),
                },
                {
                    "method": "store",
                    "zip_method_id": 0,
                    "nvcomp_algorithm": "Passthrough (uncompressed)",
                    "cpu_client_compatibility": "Universal (100%)",
                    "default": False,
                    "description": "Zero-compression instant container packaging (PKZIP Method 0).",
                },
            ],
        )

    @app.get(
        "/api/v1/options/defaults",
        response_model=ConversionOptions,
        tags=["Metadata"],
        summary="Retrieve default ConversionOptions model and parameter values",
    )
    async def get_default_options() -> ConversionOptions:
        return ConversionOptions()

    @app.post(
        "/api/v1/inspect",
        response_model=ArchiveInspectionResponse,
        tags=["Inspection"],
        summary="Inspect an uploaded archive of any format without converting",
    )
    async def inspect_uploaded_archive(
        file: UploadFile = File(..., description="Archive file of any supported format."),
        source_password: str | None = Form(
            default=None,
            description="Optional password if the uploaded archive is encrypted.",
        ),
    ) -> ArchiveInspectionResponse:
        orig_name = file.filename or "archive.bin"
        tmp_dir = Path(tempfile.mkdtemp(prefix="cloudzip_upload_inspect_"))
        staged = tmp_dir / Path(orig_name).name
        try:
            await _save_upload_with_limit(file, staged, max_bytes=cfg.max_upload_bytes)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                worker_pool,
                lambda: inspect_archive(
                    staged,
                    original_filename=orig_name,
                    password=source_password,
                    max_uncompressed_bytes=cfg.max_uncompressed_bytes,
                    max_entries=cfg.max_archive_entries,
                ),
            )
        except ArchivePasswordError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except ArchiveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @app.post(
        "/api/v1/jobs",
        response_model=JobCreatedResponse,
        status_code=202,
        tags=["Jobs & Progress Streaming"],
        summary="Upload an archive and start an asynchronous GPU ZIP conversion job",
        description=(
            "Accepts an archive file of any format plus advanced `ConversionOptions` (either as a "
            "JSON object in the `options` form field or as individual form fields). Immediately "
            "returns `202 Accepted` with `job_id`, `progress_stream_url` (SSE `0%–100%`), and `download_url`."
        ),
    )
    async def create_conversion_job(
        file: UploadFile = File(..., description="Source archive file of any format."),
        options: str | None = Form(
            default=None,
            description="JSON-encoded ConversionOptions object for advanced archive manipulation.",
        ),
        compression_method: CompressionMethod | None = Form(default=None),
        compression_level: int | None = Form(default=None, ge=1, le=5),
        smart_store: bool | None = Form(default=None),
        flatten_paths: bool | None = Form(default=None),
        strip_components: int | None = Form(default=None, ge=0, le=32),
        unwrap_single_root: bool | None = Form(default=None),
        root_prefix: str | None = Form(default=None),
        exclude_junk_files: bool | None = Form(default=None),
        exclude_hidden_files: bool | None = Form(default=None),
        deduplicate_identical: bool | None = Form(default=None),
        extract_nested_archives: bool | None = Form(default=None),
        source_password: str | None = Form(default=None),
        output_password: str | None = Form(default=None),
        encryption_method: EncryptionMethod | None = Form(default=None),
        normalize_timestamps: bool | None = Form(default=None),
        zip64_mode: Zip64Mode | None = Form(default=None),
        sort_entries: SortOrder | None = Form(default=None),
        archive_comment: str | None = Form(default=None),
        output_filename: str | None = Form(default=None),
    ) -> JobCreatedResponse:
        parsed_opts = _parse_conversion_options(
            options,
            compression_method=compression_method,
            compression_level=compression_level,
            smart_store=smart_store,
            flatten_paths=flatten_paths,
            strip_components=strip_components,
            unwrap_single_root=unwrap_single_root,
            root_prefix=root_prefix,
            exclude_junk_files=exclude_junk_files,
            exclude_hidden_files=exclude_hidden_files,
            deduplicate_identical=deduplicate_identical,
            extract_nested_archives=extract_nested_archives,
            source_password=source_password,
            output_password=output_password,
            encryption_method=encryption_method,
            normalize_timestamps=normalize_timestamps,
            zip64_mode=zip64_mode,
            sort_entries=sort_entries,
            archive_comment=archive_comment,
            output_filename=output_filename,
        )

        orig_name = file.filename or "archive.bin"
        job_id = job_manager.create_job_id()
        input_path = job_manager.job_dir(job_id) / f"input_{Path(orig_name).name}"
        await _save_upload_with_limit(file, input_path, max_bytes=cfg.max_upload_bytes)

        state = job_manager.register_job(
            job_id=job_id,
            input_path=input_path,
            original_filename=orig_name,
        )

        loop = asyncio.get_running_loop()

        def _bg_runner() -> None:
            try:
                job_manager.execute_job_sync(job_id=job_id, options=parsed_opts)
            except Exception:
                # Error event is already emitted & persisted inside execute_job_sync
                pass

        loop.run_in_executor(worker_pool, _bg_runner)
        return job_manager.build_created_response(state)

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["Jobs & Progress Streaming"],
        summary="Get current status, percentage, and telemetry for a conversion job",
    )
    async def get_job_status(job_id: str) -> JobStatusResponse:
        status = job_manager.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return status

    @app.get(
        "/api/v1/jobs/{job_id}/progress",
        tags=["Jobs & Progress Streaming"],
        summary="Stream real-time conversion percentage (0–100%) via SSE or NDJSON",
        responses={
            200: {
                "description": "Server-Sent Events (`text/event-stream`) or NDJSON stream of ProgressEvent objects.",
                "content": {
                    "text/event-stream": {
                        "example": (
                            'event: progress\ndata: {"job_id":"a1b2c3d4","stage":"compressing",'
                            '"percentage":68.4,"message":"Compressed batch 1/1 via nvcomp-cuda-gpu"}\n\n'
                        )
                    },
                    "application/x-ndjson": {},
                },
            }
        },
    )
    async def stream_job_progress(
        job_id: str,
        stream_format: Literal["sse", "ndjson"] = Query(
            default="sse",
            description="Wire format for progress events: 'sse' (Server-Sent Events) or 'ndjson'.",
        ),
    ) -> StreamingResponse:
        if job_manager.get_job_status(job_id) is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        async def _event_generator() -> AsyncIterator[str]:
            sub = job_manager.subscribe_events(job_id).__aiter__()
            while True:
                try:
                    ev = await asyncio.wait_for(sub.__anext__(), timeout=12.0)
                except asyncio.TimeoutError:
                    if stream_format == "sse":
                        yield ": keep-alive\n\n"
                    continue
                except StopAsyncIteration:
                    break

                if stream_format == "ndjson":
                    yield ev.model_dump_json() + "\n"
                else:
                    yield format_sse_event(ev)

                if ev.stage in {ConversionStage.COMPLETED, ConversionStage.FAILED}:
                    break

        media = "application/x-ndjson" if stream_format == "ndjson" else "text/event-stream"
        return StreamingResponse(_event_generator(), media_type=media, headers=SSE_HEADERS)

    @app.get(
        "/api/v1/jobs/{job_id}/download",
        tags=["Jobs & Progress Streaming"],
        summary="Download the converted GPU-compressed .zip archive",
        responses={
            200: {
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
                "description": "Converted .zip archive file.",
            }
        },
    )
    async def download_job_zip(job_id: str) -> FileResponse:
        status = job_manager.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        if status.status == ConversionStage.FAILED:
            raise HTTPException(
                status_code=409,
                detail=f"Job '{job_id}' failed: {status.error or status.message}",
            )
        if status.status != ConversionStage.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job '{job_id}' is still in '{status.status.value}' stage ({status.percentage}%).",
            )

        target = job_manager.get_download_target(job_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Output ZIP for job '{job_id}' is missing.")
        zip_path, filename, summary = target
        headers: dict[str, str] = {"X-CloudZip-Job-Id": job_id}
        if summary:
            headers.update(
                {
                    "X-CloudZip-Backend": summary.stats.backend,
                    "X-CloudZip-Input-Format": summary.stats.detected_input_format,
                    "X-CloudZip-Compression-Method": summary.stats.compression_method.value,
                    "X-CloudZip-Space-Saved-Percent": str(summary.stats.space_saved_percent),
                    "X-CloudZip-SHA256": summary.stats.sha256,
                }
            )
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=filename,
            headers=headers,
        )

    @app.delete(
        "/api/v1/jobs/{job_id}",
        tags=["Jobs & Progress Streaming"],
        summary="Delete a conversion job and remove its stored artifacts",
    )
    async def delete_job(job_id: str) -> dict[str, Any]:
        removed = job_manager.delete_job(job_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return {"job_id": job_id, "deleted": True}

    @app.post(
        "/api/v1/convert/stream",
        tags=["Direct Conversion"],
        summary="Upload an archive and stream live 0–100% percentage events in a single request",
        description=(
            "Single-call streaming endpoint: uploads any archive + `ConversionOptions` and "
            "immediately streams `ProgressEvent` percentage updates (`0.0%` -> `100.0%`) as "
            "Server-Sent Events (`text/event-stream`) or NDJSON (`?stream_format=ndjson`). "
            "The final `complete` event includes `result.download_url` (`/api/v1/jobs/{job_id}/download`) "
            "and full GPU compression statistics."
        ),
    )
    async def convert_with_progress_stream(
        file: UploadFile = File(..., description="Source archive file of any format."),
        options: str | None = Form(
            default=None,
            description="JSON-encoded ConversionOptions object.",
        ),
        stream_format: Literal["sse", "ndjson"] = Query(default="sse"),
        compression_method: CompressionMethod | None = Form(default=None),
        compression_level: int | None = Form(default=None, ge=1, le=5),
        smart_store: bool | None = Form(default=None),
        flatten_paths: bool | None = Form(default=None),
        strip_components: int | None = Form(default=None, ge=0, le=32),
        unwrap_single_root: bool | None = Form(default=None),
        root_prefix: str | None = Form(default=None),
        exclude_junk_files: bool | None = Form(default=None),
        exclude_hidden_files: bool | None = Form(default=None),
        deduplicate_identical: bool | None = Form(default=None),
        extract_nested_archives: bool | None = Form(default=None),
        source_password: str | None = Form(default=None),
        output_password: str | None = Form(default=None),
        encryption_method: EncryptionMethod | None = Form(default=None),
        normalize_timestamps: bool | None = Form(default=None),
        zip64_mode: Zip64Mode | None = Form(default=None),
        sort_entries: SortOrder | None = Form(default=None),
        archive_comment: str | None = Form(default=None),
        output_filename: str | None = Form(default=None),
    ) -> StreamingResponse:
        parsed_opts = _parse_conversion_options(
            options,
            compression_method=compression_method,
            compression_level=compression_level,
            smart_store=smart_store,
            flatten_paths=flatten_paths,
            strip_components=strip_components,
            unwrap_single_root=unwrap_single_root,
            root_prefix=root_prefix,
            exclude_junk_files=exclude_junk_files,
            exclude_hidden_files=exclude_hidden_files,
            deduplicate_identical=deduplicate_identical,
            extract_nested_archives=extract_nested_archives,
            source_password=source_password,
            output_password=output_password,
            encryption_method=encryption_method,
            normalize_timestamps=normalize_timestamps,
            zip64_mode=zip64_mode,
            sort_entries=sort_entries,
            archive_comment=archive_comment,
            output_filename=output_filename,
        )

        orig_name = file.filename or "archive.bin"
        job_id = job_manager.create_job_id()
        input_path = job_manager.job_dir(job_id) / f"input_{Path(orig_name).name}"
        await _save_upload_with_limit(file, input_path, max_bytes=cfg.max_upload_bytes)

        job_manager.register_job(
            job_id=job_id,
            input_path=input_path,
            original_filename=orig_name,
        )

        loop = asyncio.get_running_loop()

        def _bg_runner() -> None:
            try:
                job_manager.execute_job_sync(job_id=job_id, options=parsed_opts)
            except Exception:
                pass

        loop.run_in_executor(worker_pool, _bg_runner)

        async def _stream() -> AsyncIterator[str]:
            sub = job_manager.subscribe_events(job_id).__aiter__()
            while True:
                try:
                    ev = await asyncio.wait_for(sub.__anext__(), timeout=12.0)
                except asyncio.TimeoutError:
                    if stream_format == "sse":
                        yield ": keep-alive\n\n"
                    continue
                except StopAsyncIteration:
                    break
                if stream_format == "ndjson":
                    yield ev.model_dump_json() + "\n"
                else:
                    yield format_sse_event(ev)
                if ev.stage in {ConversionStage.COMPLETED, ConversionStage.FAILED}:
                    break

        media = "application/x-ndjson" if stream_format == "ndjson" else "text/event-stream"
        headers = {**SSE_HEADERS, "X-CloudZip-Job-Id": job_id}
        return StreamingResponse(_stream(), media_type=media, headers=headers)

    @app.post(
        "/api/v1/convert",
        tags=["Direct Conversion"],
        summary="Synchronous one-shot archive-to-ZIP conversion returning the .zip binary directly",
        responses={
            200: {
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
                "description": "Converted GPU-compressed .zip archive.",
            }
        },
    )
    async def convert_direct_download(
        file: UploadFile = File(..., description="Source archive file of any format."),
        options: str | None = Form(
            default=None,
            description="JSON-encoded ConversionOptions object.",
        ),
        compression_method: CompressionMethod | None = Form(default=None),
        compression_level: int | None = Form(default=None, ge=1, le=5),
        smart_store: bool | None = Form(default=None),
        flatten_paths: bool | None = Form(default=None),
        strip_components: int | None = Form(default=None, ge=0, le=32),
        unwrap_single_root: bool | None = Form(default=None),
        root_prefix: str | None = Form(default=None),
        exclude_junk_files: bool | None = Form(default=None),
        exclude_hidden_files: bool | None = Form(default=None),
        deduplicate_identical: bool | None = Form(default=None),
        extract_nested_archives: bool | None = Form(default=None),
        source_password: str | None = Form(default=None),
        output_password: str | None = Form(default=None),
        encryption_method: EncryptionMethod | None = Form(default=None),
        normalize_timestamps: bool | None = Form(default=None),
        zip64_mode: Zip64Mode | None = Form(default=None),
        sort_entries: SortOrder | None = Form(default=None),
        archive_comment: str | None = Form(default=None),
        output_filename: str | None = Form(default=None),
    ) -> FileResponse:
        parsed_opts = _parse_conversion_options(
            options,
            compression_method=compression_method,
            compression_level=compression_level,
            smart_store=smart_store,
            flatten_paths=flatten_paths,
            strip_components=strip_components,
            unwrap_single_root=unwrap_single_root,
            root_prefix=root_prefix,
            exclude_junk_files=exclude_junk_files,
            exclude_hidden_files=exclude_hidden_files,
            deduplicate_identical=deduplicate_identical,
            extract_nested_archives=extract_nested_archives,
            source_password=source_password,
            output_password=output_password,
            encryption_method=encryption_method,
            normalize_timestamps=normalize_timestamps,
            zip64_mode=zip64_mode,
            sort_entries=sort_entries,
            archive_comment=archive_comment,
            output_filename=output_filename,
        )

        orig_name = file.filename or "archive.bin"
        job_id = job_manager.create_job_id()
        input_path = job_manager.job_dir(job_id) / f"input_{Path(orig_name).name}"
        await _save_upload_with_limit(file, input_path, max_bytes=cfg.max_upload_bytes)
        job_manager.register_job(
            job_id=job_id,
            input_path=input_path,
            original_filename=orig_name,
        )

        loop = asyncio.get_running_loop()
        try:
            summary = await loop.run_in_executor(
                worker_pool,
                lambda: job_manager.execute_job_sync(job_id=job_id, options=parsed_opts),
            )
        except ArchivePasswordError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except ArchiveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        target = job_manager.get_download_target(job_id)
        if target is None:
            raise HTTPException(status_code=500, detail="Converted output ZIP was not found.")
        zip_path, filename, _ = target
        headers = {
            "X-CloudZip-Job-Id": job_id,
            "X-CloudZip-Backend": summary.stats.backend,
            "X-CloudZip-Input-Format": summary.stats.detected_input_format,
            "X-CloudZip-Compression-Method": summary.stats.compression_method.value,
            "X-CloudZip-Space-Saved-Percent": str(summary.stats.space_saved_percent),
            "X-CloudZip-SHA256": summary.stats.sha256,
        }
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=filename,
            headers=headers,
        )

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info,
        )
        schema["info"]["x-copyright"] = __copyright__
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        # Ensure streaming ProgressEvent and nested models are explicitly documented in schemas
        for extra_model in (ConversionOptions, RenameRule, ProgressEvent):
            model_schema = extra_model.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            defs = model_schema.pop("$defs", {})
            for def_name, def_val in defs.items():
                components.setdefault(def_name, def_val)
            components.setdefault(extra_model.__name__, model_schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


app = create_app()


def run_local() -> None:
    """Run the CloudZIP FastAPI server locally via Uvicorn."""
    uvicorn.run("cloudzip.api:app", host="0.0.0.0", port=8000, reload=False)
