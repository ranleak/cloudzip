"""Pydantic data models for CloudZIP API requests, options, progress events, and telemetry.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompressionMethod(str, Enum):
    """ZIP compression method backed by NVIDIA nvCOMP GPU acceleration."""

    DEFLATE = "deflate"
    ZSTD = "zstd"
    STORE = "store"


class EncryptionMethod(str, Enum):
    """Output ZIP encryption scheme."""

    NONE = "none"
    AES256 = "aes256"
    ZIPCRYPTO = "zipcrypto"


class Zip64Mode(str, Enum):
    """ZIP64 extended header policy."""

    AUTO = "auto"
    FORCE = "force"
    DISABLE = "disable"


class SortOrder(str, Enum):
    """Ordering of entries written into the output ZIP archive."""

    PATH = "path"
    EXTENSION = "extension"
    SIZE_ASC = "size_asc"
    SIZE_DESC = "size_desc"
    PRESERVE = "preserve"


class ConversionStage(str, Enum):
    """Lifecycle stage during archive extraction, manipulation, and GPU ZIP compression."""

    QUEUED = "queued"
    INITIALIZING = "initializing"
    DETECTING = "detecting"
    EXTRACTING = "extracting"
    MANIPULATING = "manipulating"
    COMPRESSING = "compressing"
    PACKAGING = "packaging"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class RenameRule(BaseModel):
    """Regular-expression path transformation rule applied to archive entry paths."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Python regular expression matched against the normalized POSIX archive path.",
        examples=["^legacy_src/"],
    )
    replacement: str = Field(
        default="",
        max_length=512,
        description="Replacement string (supports regex capture groups such as \\1).",
        examples=["src/"],
    )


class ConversionOptions(BaseModel):
    """Advanced archive manipulation and nvCOMP GPU ZIP compression options."""

    model_config = ConfigDict(extra="forbid")

    compression_method: CompressionMethod = Field(
        default=CompressionMethod.DEFLATE,
        description=(
            "Target ZIP compression codec. Defaults to 'deflate' (ZIP Method 8, RFC 1951) "
            "via NVIDIA nvCOMP BitstreamKind.RAW with algorithm_type=5 — the highest-ratio "
            "GPU-accelerated compression method universally compatible across all CPU-only "
            "client devices (Windows, macOS, iOS, Android, Linux, browsers). "
            "Select 'zstd' (ZIP Method 93) for modern Zstandard extractors or 'store' (Method 0)."
        ),
    )
    compression_level: int = Field(
        default=5,
        ge=1,
        le=5,
        description=(
            "NVIDIA nvCOMP compression ratio tier (1 = ultra-fast GPU throughput, "
            "2 = medium, 4 = high ratio beating zlib-6, 5 = maximum compression ratio / "
            "smallest output size). Defaults to 5."
        ),
    )
    smart_store: bool = Field(
        default=True,
        description=(
            "Automatically fall back to uncompressed STORE (Method 0) on a per-entry basis "
            "whenever compressed output is larger than or equal to the original file size, "
            "preventing expansion on already-compressed media."
        ),
    )
    gpu_batch_size_mb: int = Field(
        default=64,
        ge=4,
        le=512,
        description=(
            "Maximum batch payload size (MiB) dispatched in parallel to NVIDIA nvCOMP "
            "BatchArray GPU kernels."
        ),
    )
    zstd_chunk_size_mb: int = Field(
        default=4,
        ge=1,
        le=16,
        description=(
            "Per-frame chunk size (MiB) for parallel multi-frame GPU Zstandard compression "
            "when compression_method='zstd'."
        ),
    )

    # Path & entry manipulation
    flatten_paths: bool = Field(
        default=False,
        description=(
            "Flatten nested directory hierarchies into the root of the ZIP archive, "
            "automatically disambiguating filename collisions."
        ),
    )
    strip_components: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Strip N leading path components from every entry (similar to tar --strip-components).",
    )
    unwrap_single_root: bool = Field(
        default=False,
        description=(
            "If all archive contents reside under a single redundant top-level folder, "
            "automatically unwrap that folder."
        ),
    )
    root_prefix: str | None = Field(
        default=None,
        max_length=255,
        description="Optional root directory prefix prepended to every entry inside the output ZIP.",
        examples=["release-bundle"],
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        description="Optional glob patterns; when non-empty, only matching files are included.",
        examples=[["*.py", "*.md", "assets/**"]],
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Optional glob patterns of files or directories to exclude from the output ZIP.",
        examples=[["*.tmp", "*.log", "node_modules/**", ".git/**"]],
    )
    exclude_junk_files: bool = Field(
        default=False,
        description="Remove OS metadata clutter (__MACOSX/, .DS_Store, Thumbs.db, desktop.ini, ._*).",
    )
    exclude_hidden_files: bool = Field(
        default=False,
        description="Exclude dotfiles and hidden directories (segments starting with '.').",
    )
    rename_rules: list[RenameRule] = Field(
        default_factory=list,
        description="Ordered regex search-and-replace rules applied to normalized entry paths.",
    )
    deduplicate_identical: bool = Field(
        default=False,
        description="Omit byte-identical duplicate files based on SHA-256 content digest.",
    )
    include_empty_dirs: bool = Field(
        default=True,
        description="Preserve empty directory entries in the generated ZIP archive.",
    )
    sort_entries: SortOrder = Field(
        default=SortOrder.PATH,
        description="Entry ordering inside the ZIP Central Directory.",
    )

    # Nested extraction & passwords
    extract_nested_archives: bool = Field(
        default=False,
        description="Recursively unpack nested archives discovered inside the uploaded archive.",
    )
    max_nested_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum recursion depth when extract_nested_archives is enabled.",
    )
    source_password: str | None = Field(
        default=None,
        max_length=512,
        description="Password used to decrypt password-protected input archives (.zip, .7z, .rar).",
    )
    output_password: str | None = Field(
        default=None,
        max_length=512,
        description="Optional password to encrypt the output ZIP archive.",
    )
    encryption_method: EncryptionMethod = Field(
        default=EncryptionMethod.NONE,
        description=(
            "Encryption algorithm for output ZIP when output_password is supplied. "
            "'aes256' uses WinZip AE-2 256-bit AES-CTR + HMAC-SHA1 over GPU-compressed payloads."
        ),
    )

    # Metadata & format integrity
    preserve_timestamps: bool = Field(
        default=True,
        description="Preserve original file modification timestamps (DOS + UTC 0x5455 extra field).",
    )
    normalize_timestamps: bool = Field(
        default=False,
        description="Pin all entry timestamps to 2020-01-01T00:00:00Z for deterministic, reproducible ZIP builds.",
    )
    preserve_permissions: bool = Field(
        default=True,
        description="Preserve POSIX file permission bits in Central Directory external attributes.",
    )
    zip64_mode: Zip64Mode = Field(
        default=Zip64Mode.AUTO,
        description=(
            "ZIP64 policy: 'auto' writes clean PKZIP 2.0 headers under 4 GiB / 65,535 files "
            "and upgrades seamlessly to ZIP64 when needed."
        ),
    )
    archive_comment: str | None = Field(
        default=None,
        max_length=8192,
        description="Optional UTF-8 archive comment stored in the End of Central Directory record.",
    )
    output_filename: str | None = Field(
        default=None,
        max_length=255,
        description="Custom download filename for the resulting .zip archive.",
        examples=["converted-archive.zip"],
    )
    verify_integrity: bool = Field(
        default=True,
        description="Validate output ZIP Central Directory headers and checksums before finalizing.",
    )

    @field_validator("root_prefix")
    @classmethod
    def _clean_root_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.replace("\\", "/").strip().strip("/")
        if not cleaned:
            return None
        parts = [p for p in cleaned.split("/") if p not in {"", ".", ".."}]
        return "/".join(parts) if parts else None

    @model_validator(mode="after")
    def _validate_encryption(self) -> "ConversionOptions":
        if self.output_password:
            if self.encryption_method == EncryptionMethod.NONE:
                self.encryption_method = EncryptionMethod.AES256
        elif self.encryption_method != EncryptionMethod.NONE:
            raise ValueError(
                "output_password must be provided when encryption_method is not 'none'."
            )
        return self


class ArchiveEntryInfo(BaseModel):
    """Metadata for a single entry inside an inspected or converted archive."""

    path: str = Field(..., description="Normalized POSIX relative path inside the archive.")
    is_dir: bool = Field(..., description="True if entry represents a directory.")
    size_bytes: int = Field(..., ge=0, description="Uncompressed size in bytes.")
    compressed_bytes: int | None = Field(
        default=None, description="Compressed size in source archive if reported."
    )
    modified_iso: str | None = Field(
        default=None, description="Last-modified timestamp in ISO-8601 UTC format."
    )
    mode_octal: str | None = Field(
        default=None, description="POSIX file permissions in octal notation (e.g. '0o644')."
    )
    encrypted: bool = Field(default=False, description="Whether entry is encrypted in source.")


class ArchiveInspectionResponse(BaseModel):
    """Detailed inspection report for an uploaded archive of any format."""

    filename: str = Field(..., description="Uploaded archive filename.")
    detected_format: str = Field(
        ..., description="Canonical archive format identifier (e.g. 'tar.zst', '7z', 'rar', 'zip')."
    )
    extractor_used: str = Field(..., description="Backend used to inspect the archive.")
    archive_size_bytes: int = Field(..., ge=0, description="Uploaded archive file size in bytes.")
    total_entries: int = Field(..., ge=0, description="Total entries (files + directories).")
    file_count: int = Field(..., ge=0, description="Number of regular files.")
    dir_count: int = Field(..., ge=0, description="Number of directory entries.")
    total_uncompressed_bytes: int = Field(
        ..., ge=0, description="Sum of uncompressed file sizes in bytes."
    )
    has_encrypted_entries: bool = Field(
        ..., description="True if one or more entries require a password."
    )
    single_top_level_root: str | None = Field(
        default=None,
        description="Name of single top-level directory if all entries share one common root.",
    )
    entries: list[ArchiveEntryInfo] = Field(
        default_factory=list, description="Inspected archive entry manifest."
    )


class CompressionStats(BaseModel):
    """End-to-end compression, manipulation, and GPU execution telemetry."""

    input_filename: str = Field(..., description="Original uploaded archive filename.")
    output_filename: str = Field(..., description="Generated output .zip filename.")
    detected_input_format: str = Field(..., description="Detected input archive format.")
    compression_method: CompressionMethod = Field(
        ..., description="Requested primary ZIP compression method."
    )
    compression_level: int = Field(..., description="nvCOMP compression ratio tier used (1-5).")
    backend: str = Field(
        ...,
        description="Active compression engine ('nvcomp-cuda-gpu' on NVIDIA GPU or 'cpu-fallback').",
    )
    gpu_device_name: str | None = Field(
        default=None, description="NVIDIA GPU model used for nvCOMP compression, if active."
    )
    nvcomp_version: str | None = Field(
        default=None, description="Installed nvidia-nvcomp library version."
    )
    encrypted_output: bool = Field(
        default=False, description="True if output ZIP is password-encrypted."
    )
    encryption_method: EncryptionMethod = Field(
        default=EncryptionMethod.NONE, description="Encryption scheme applied to output ZIP."
    )
    zip64_used: bool = Field(
        default=False, description="True if ZIP64 extended headers were written."
    )
    original_archive_bytes: int = Field(..., ge=0, description="Size of uploaded source archive.")
    uncompressed_bytes: int = Field(
        ..., ge=0, description="Total uncompressed size of included files."
    )
    compressed_payload_bytes: int = Field(
        ..., ge=0, description="Total compressed payload size inside output ZIP."
    )
    output_zip_bytes: int = Field(..., ge=0, description="Final output .zip file size in bytes.")
    compression_ratio: float = Field(
        ...,
        ge=0.0,
        description="Ratio of compressed_payload_bytes / uncompressed_bytes (lower is smaller).",
    )
    space_saved_bytes: int = Field(
        ..., description="Bytes saved relative to uncompressed file data."
    )
    space_saved_percent: float = Field(
        ..., description="Percentage size reduction relative to uncompressed file data."
    )
    entries_discovered: int = Field(
        ..., ge=0, description="Total entries discovered in the uploaded archive."
    )
    entries_included: int = Field(
        ..., ge=0, description="Entries written into output ZIP after filtering/manipulation."
    )
    entries_filtered_out: int = Field(
        ..., ge=0, description="Entries excluded by filters, strip rules, or deduplication."
    )
    entries_gpu_compressed: int = Field(
        ..., ge=0, description="Files compressed with DEFLATE/ZSTD codec."
    )
    entries_smart_stored: int = Field(
        ..., ge=0, description="Files stored uncompressed via STORE (0-byte or Smart Store)."
    )
    gpu_compress_ms: float = Field(
        ..., ge=0.0, description="Time spent in compression kernels (milliseconds)."
    )
    total_elapsed_ms: float = Field(
        ..., ge=0.0, description="Total end-to-end conversion duration (milliseconds)."
    )
    throughput_mbps: float = Field(
        ..., ge=0.0, description="Effective compression throughput in MiB/s."
    )
    sha256: str = Field(..., description="SHA-256 hex digest of the generated .zip archive.")


class JobResultSummary(BaseModel):
    """Final output descriptor attached when a conversion job reaches 100%."""

    job_id: str = Field(..., description="Unique conversion job identifier.")
    download_url: str = Field(..., description="Relative API path to download the converted .zip.")
    output_filename: str = Field(..., description="Download filename of the converted .zip.")
    stats: CompressionStats = Field(..., description="Detailed compression and GPU telemetry.")


class ProgressEvent(BaseModel):
    """Real-time progress percentage event streamed over SSE or NDJSON."""

    job_id: str = Field(..., description="Unique conversion job identifier.")
    stage: ConversionStage = Field(..., description="Current pipeline stage.")
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Monotonic overall completion percentage from 0.0 to 100.0.",
    )
    message: str = Field(..., description="Human-readable status description.")
    current_entry: str | None = Field(
        default=None, description="Archive entry currently being processed, if applicable."
    )
    entries_processed: int = Field(default=0, ge=0, description="Entries processed in current/overall stage.")
    entries_total: int = Field(default=0, ge=0, description="Total entries targeted.")
    bytes_processed: int = Field(default=0, ge=0, description="Uncompressed bytes processed so far.")
    bytes_total: int = Field(default=0, ge=0, description="Total uncompressed bytes to process.")
    throughput_mbps: float | None = Field(
        default=None, description="Instantaneous compression throughput in MiB/s."
    )
    backend: str | None = Field(
        default=None, description="Compression backend ('nvcomp-cuda-gpu' or 'cpu-fallback')."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp when event was emitted.",
    )
    result: JobResultSummary | None = Field(
        default=None, description="Populated on the terminal 100% 'completed' event."
    )
    error: str | None = Field(
        default=None, description="Error details if stage is 'failed'."
    )


class JobCreatedResponse(BaseModel):
    """Response returned immediately when an asynchronous conversion job is created."""

    job_id: str = Field(..., description="Unique conversion job identifier.")
    status: ConversionStage = Field(..., description="Initial job status.")
    status_url: str = Field(..., description="Relative URL to poll job status.")
    progress_stream_url: str = Field(
        ..., description="Relative SSE URL streaming live 0-100% ProgressEvent updates."
    )
    download_url: str = Field(
        ..., description="Relative URL to download the .zip once conversion completes."
    )
    created_at: str = Field(..., description="UTC ISO-8601 job creation timestamp.")


class JobStatusResponse(BaseModel):
    """Current snapshot of a conversion job."""

    job_id: str = Field(..., description="Unique conversion job identifier.")
    status: ConversionStage = Field(..., description="Current job stage.")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Latest completion percentage.")
    message: str = Field(..., description="Latest status message.")
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp.")
    updated_at: str = Field(..., description="UTC ISO-8601 last-update timestamp.")
    download_ready: bool = Field(..., description="True when the output .zip is ready to download.")
    download_url: str | None = Field(default=None, description="Download endpoint when ready.")
    result: JobResultSummary | None = Field(
        default=None, description="Final result summary and stats when completed."
    )
    error: str | None = Field(default=None, description="Failure message if status is 'failed'.")


class GpuCapabilities(BaseModel):
    """NVIDIA GPU and nvCOMP codec runtime status."""

    nvcomp_installed: bool = Field(..., description="True if nvidia-nvcomp Python package is installed.")
    nvcomp_version: str | None = Field(default=None, description="nvidia-nvcomp package version.")
    gpu_available: bool = Field(
        ..., description="True if an NVIDIA CUDA GPU was detected and verified with nvCOMP."
    )
    gpu_device_name: str | None = Field(default=None, description="Detected NVIDIA GPU device name.")
    active_backend: Literal["nvcomp-cuda-gpu", "cpu-fallback"] = Field(
        ..., description="Active compression backend."
    )
    default_zip_method: str = Field(
        default="deflate (ZIP Method 8, RFC 1951, nvCOMP algorithm_type=5 BitstreamKind.RAW)",
        description="Default GPU-accelerated universally compatible ZIP compression strategy.",
    )
    supported_zip_methods: list[str] = Field(
        default_factory=lambda: ["deflate", "zstd", "store"],
        description="Available output ZIP compression methods.",
    )


class HealthResponse(BaseModel):
    """Service health and GPU runtime telemetry."""

    status: Literal["healthy"] = "healthy"
    service: str = "CloudZIP"
    version: str = "1.0.0"
    copyright: str = "Copyright (c) 2026 Alexander Scott"
    modal_runtime: bool = Field(..., description="True when executing inside Modal.com cloud.")
    gpu: GpuCapabilities = Field(..., description="NVIDIA nvCOMP & GPU telemetry.")


class FormatDescriptor(BaseModel):
    """Supported input archive or compressed stream format descriptor."""

    format_id: str = Field(..., description="Canonical format identifier.")
    extensions: list[str] = Field(..., description="Recognized file extensions.")
    description: str = Field(..., description="Format description.")
    password_decryption_supported: bool = Field(
        default=False, description="Whether encrypted input archives of this format are supported."
    )


class SupportedFormatsResponse(BaseModel):
    """Catalog of input archive formats and GPU output ZIP codecs."""

    copyright: str = "Copyright (c) 2026 Alexander Scott"
    input_formats: list[FormatDescriptor]
    output_compression_methods: list[dict[str, str | int | bool]]
