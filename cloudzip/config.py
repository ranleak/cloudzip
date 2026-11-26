"""Runtime configuration for CloudZIP.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _is_modal_runtime() -> bool:
    return bool(
        os.environ.get("MODAL_IMAGE_ID")
        or os.environ.get("MODAL_TASK_ID")
        or os.environ.get("MODAL_ENVIRONMENT")
    )


def _default_storage_dir() -> Path:
    explicit = os.environ.get("CLOUDZIP_STORAGE_DIR")
    if explicit:
        return Path(explicit)
    if _is_modal_runtime():
        return Path("/data/cloudzip")
    return Path(tempfile.gettempdir()) / "cloudzip_storage"


@dataclass(slots=True)
class CloudZipSettings:
    """Centralized service configuration populated from environment variables."""

    app_name: str = field(default_factory=lambda: os.environ.get("CLOUDZIP_APP_NAME", "cloudzip"))
    gpu_type: str = field(default_factory=lambda: os.environ.get("CLOUDZIP_GPU", "L4"))
    storage_dir: Path = field(default_factory=_default_storage_dir)
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.environ.get("CLOUDZIP_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
    )
    max_uncompressed_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("CLOUDZIP_MAX_UNCOMPRESSED_BYTES", str(8 * 1024**3))
        )
    )
    max_archive_entries: int = field(
        default_factory=lambda: int(os.environ.get("CLOUDZIP_MAX_ARCHIVE_ENTRIES", "100000"))
    )
    job_ttl_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CLOUDZIP_JOB_TTL_SECONDS", "3600"))
    )
    require_gpu: bool = field(
        default_factory=lambda: os.environ.get("CLOUDZIP_REQUIRE_GPU", "false").lower()
        in {"1", "true", "yes"}
    )
    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.environ.get("CLOUDZIP_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )
    modal_volume_name: str = field(
        default_factory=lambda: os.environ.get("CLOUDZIP_VOLUME_NAME", "cloudzip-jobs")
    )
    modal_dict_name: str = field(
        default_factory=lambda: os.environ.get("CLOUDZIP_DICT_NAME", "cloudzip-job-state")
    )

    def ensure_dirs(self) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "jobs").mkdir(parents=True, exist_ok=True)
        return self.storage_dir


def get_settings() -> CloudZipSettings:
    settings = CloudZipSettings()
    settings.ensure_dirs()
    return settings
