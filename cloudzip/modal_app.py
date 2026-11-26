"""Modal.com serverless GPU deployment definition for CloudZIP.

Deploy permanently:
    uv run modal deploy modal_app.py

Serve live-reloading preview:
    uv run modal serve modal_app.py

Convert a local archive via Modal GPU RPC:
    uv run modal run modal_app.py --input-path ./sample.tar.zst --output-path ./sample.zip

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import modal

GPU_SPEC = os.environ.get("CLOUDZIP_GPU", "L4")
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install(
        "libarchive-tools",
        "libarchive-dev",
        "p7zip-full",
        "unar",
        "zstd",
        "lz4",
        "xz-utils",
        "bzip2",
        "gzip",
        "cpio",
        "binutils",
    )
    .uv_sync(uv_project_dir=PROJECT_ROOT, extra_options="--no-dev")
    .env({"CLOUDZIP_STORAGE_DIR": "/data/cloudzip"})
    .add_local_python_source("cloudzip")
)

app = modal.App(
    name="cloudzip",
    image=image,
    tags={
        "service": "cloudzip",
        "copyright": "Alexander_Scott",
        "engine": "nvidia-nvcomp",
    },
)

jobs_volume = modal.Volume.from_name("cloudzip-jobs", create_if_missing=True)
jobs_dict = modal.Dict.from_name("cloudzip-job-state", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=4.0,
    memory=16384,
    volumes={"/data/cloudzip": jobs_volume},
    scaledown_window=300,
    timeout=1800,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="cloudzip-api")
def web_api() -> Any:
    """Expose the full CloudZIP FastAPI service on Modal with NVIDIA GPU acceleration."""
    from cloudzip.api import create_app
    from cloudzip.config import get_settings
    from cloudzip.nvcomp_engine import NvcompCompressionEngine

    settings = get_settings()
    engine = NvcompCompressionEngine(require_gpu=settings.require_gpu)
    return create_app(
        settings=settings,
        engine=engine,
        modal_volume=jobs_volume,
        modal_dict=jobs_dict,
    )


@app.cls(
    image=image,
    gpu=GPU_SPEC,
    cpu=4.0,
    memory=16384,
    volumes={"/data/cloudzip": jobs_volume},
    scaledown_window=180,
    timeout=1800,
)
class CloudZipGpuWorker:
    """Dedicated Modal GPU worker class for programmatic RPC archive conversions."""

    @modal.enter()
    def warmup_nvcomp(self) -> None:
        from cloudzip.config import get_settings
        from cloudzip.nvcomp_engine import NvcompCompressionEngine

        self.settings = get_settings()
        self.engine = NvcompCompressionEngine(require_gpu=self.settings.require_gpu)

    @modal.method()
    def convert_archive_bytes(
        self,
        archive_bytes: bytes,
        original_filename: str,
        options_dict: dict[str, Any] | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Convert raw archive bytes into a GPU-compressed `.zip` and return `(zip_bytes, stats)`."""
        from cloudzip.models import ConversionOptions
        from cloudzip.pipeline import run_conversion_pipeline

        opts = ConversionOptions.model_validate(options_dict or {})
        with tempfile.TemporaryDirectory(prefix="cloudzip_rpc_") as tmp_dir:
            tmp_p = Path(tmp_dir)
            in_file = tmp_p / Path(original_filename).name
            out_file = tmp_p / "converted.zip"
            in_file.write_bytes(archive_bytes)
            summary = run_conversion_pipeline(
                job_id="rpc",
                source_archive_path=in_file,
                original_filename=original_filename,
                output_zip_path=out_file,
                options=opts,
                engine=self.engine,
                settings=self.settings,
                download_url="",
            )
            return out_file.read_bytes(), summary.stats.model_dump(mode="json")


@app.local_entrypoint()
def main(
    input_path: str,
    output_path: str = "",
    compression_method: str = "deflate",
    compression_level: int = 5,
    flatten_paths: bool = False,
) -> None:
    """CLI entrypoint for `modal run modal_app.py --input-path <archive>`."""
    from cloudzip.pipeline import derive_output_zip_filename

    src = Path(input_path).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"Input archive not found: {src}")

    dest = (
        Path(output_path).expanduser().resolve()
        if output_path
        else src.with_name(derive_output_zip_filename(src.name))
    )

    print(f"Uploading '{src.name}' ({src.stat().st_size:,} bytes) to Modal NVIDIA {GPU_SPEC} GPU...")
    worker = CloudZipGpuWorker()
    zip_bytes, stats = worker.convert_archive_bytes.remote(
        archive_bytes=src.read_bytes(),
        original_filename=src.name,
        options_dict={
            "compression_method": compression_method,
            "compression_level": compression_level,
            "flatten_paths": flatten_paths,
        },
    )
    dest.write_bytes(zip_bytes)
    print(
        f"Saved '{dest}' ({len(zip_bytes):,} bytes) | "
        f"Backend: {stats['backend']} | "
        f"Space saved: {stats['space_saved_percent']}% | "
        f"SHA-256: {stats['sha256']}"
    )
