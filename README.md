# CloudZIP

**CloudZIP** is a serverless GPU-accelerated archive manipulation and universal `.zip` conversion cloud service built on **[Modal.com](https://modal.com)**, **FastAPI**, and **NVIDIA nvCOMP**.

Upload an archive in virtually any format (`.7z`, `.rar`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.tar.zst`, `.tar.lz4`, `.tar.br`, `.zip`, `.iso`, `.cab`, `.deb`, `.ar`, `.rpm`, `.cpio`, `.xar`, `.wim`, `.lzh`, `.gz`, `.bz2`, `.xz`, `.zst`, `.lz4`, `.br`, `.snappy`), apply rich path/entry manipulations, stream real-time `0%–100%` completion telemetry over Server-Sent Events (SSE) or NDJSON, and download a universally compatible, maximally compressed `.zip` archive.

> **Copyright (c) 2026 Alexander Scott.** Licensed under the [MIT License](./LICENSE).

---

## Why NVIDIA nvCOMP `Deflate` (`algorithm_type=5`, `BitstreamKind.RAW`)?

NVIDIA nvCOMP offers both GPU-specific codecs (`ANS`, `Bitcomp`, `Cascaded`, `GDeflate`) and industry-standard codecs (`Deflate`, `Zstd`, `LZ4`, `Snappy`, `Gzip`).

To satisfy **universal compatibility on CPU-only client devices** (Windows Explorer, macOS Finder / Archive Utility, iOS Files, Android Files, Linux `unzip`, Python `zipfile`, Java, and browser unzippers) **while maximizing compression ratio on the GPU**, CloudZIP defaults to:

1. **`nvidia.nvcomp.Codec(algorithm="Deflate", algorithm_type=5, bitstream_kind=BitstreamKind.RAW)`**
   - Emits pure **RFC 1951 raw DEFLATE** bitstreams (PKZIP Compression Method `8`) with zero proprietary nvCOMP headers.
   - Uses `algorithm_type=5` — nvCOMP's highest compression-ratio tier (exceeding standard `zlib` level 6+).
   - Executes across files in parallel on the GPU using `nvcomp.as_arrays(...).cuda()` (`BatchArray`) with thread-local CUDA stream isolation.
2. **Per-Entry Smart `STORE` Guard (PKZIP Method `0`)**
   - Automatically compares compressed output against raw input size per entry and switches already-compressed or tiny files to `STORE` (`Method 0`), guaranteeing `compressed_size <= uncompressed_size` on every file.
3. **Optional GPU Zstandard Mode (`compression_method="zstd"`, PKZIP Method `93`)**
   - Uses `nvidia.nvcomp.Codec(algorithm="Zstd", bitstream_kind=BitstreamKind.RAW)` with parallel multi-frame GPU chunking (`RFC 8878`) for modern CPU extractors (`7-Zip`, `WinRAR`, `libarchive`/`bsdtar`, `PeaZip`, Windows 11).

---

## Key Capabilities

- **Universal Input Format Support**: `.zip` (including WinZip AES-256, ZipCrypto, Zstd), `.7z` (including AES-256), `.rar` (v4 & v5), `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`, `.tar.xz`/`.txz`, `.tar.zst`/`.tzst`, `.tar.lz4`, `.tar.br`, `.tar.lzma`, `.deb`, `.ar`, `.cpio`, `.iso`, `.cab`, `.rpm`, `.xar`, `.wim`, `.lzh`/`.arj`, and single-stream `.gz`, `.bz2`, `.xz`, `.lzma`, `.zst`, `.lz4`, `.br`, `.snappy`.
- **Advanced Archive Manipulation**:
  - Strip `N` leading path components (`strip_components`) or unwrap redundant single top-level folders (`unwrap_single_root`).
  - Flatten nested directory hierarchies (`flatten_paths`) with automatic collision-safe renaming.
  - Prepend custom root folders (`root_prefix`) and apply ordered regex path rewrites (`rename_rules`).
  - Filter entries via `include_patterns` / `exclude_patterns` globs, remove OS clutter (`exclude_junk_files`), omit dotfiles (`exclude_hidden_files`), and deduplicate identical files by SHA-256 (`deduplicate_identical`).
  - Recursively unpack nested inner archives (`extract_nested_archives`).
  - Decrypt password-protected source archives (`source_password`) and encrypt output `.zip` archives with **WinZip AE-2 256-bit AES-CTR + HMAC-SHA1** (`output_password` + `encryption_method="aes256"`) *after* GPU compression.
  - Automatic **ZIP64** promotion (`zip64_mode="auto"`), deterministic reproducible builds (`normalize_timestamps`), POSIX permission preservation, and UTF-8 archive comments.
- **Real-Time Percentage (`0.0%–100.0%`) Progress Streaming**:
  - Async job workflow: `POST /api/v1/jobs` -> `GET /api/v1/jobs/{job_id}/progress` (browser `EventSource` SSE) -> `GET /api/v1/jobs/{job_id}/download`.
  - Single-request streaming workflow: `POST /api/v1/convert/stream` (SSE or `?stream_format=ndjson`).
  - Synchronous binary workflow: `POST /api/v1/convert` (`application/zip`).

---

## Quick Start with `uv`

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Test Suite Locally
```bash
uv run pytest -v
```

### 3. Run FastAPI Locally (CPU Fallback or Local CUDA GPU)
```bash
uv run cloudzip-api
# Interactive OpenAPI docs: http://localhost:8000/docs
```

### 4. Deploy to Modal.com Serverless GPU Cloud
```bash
# Authenticate once with Modal
uv run modal setup

# Live-reloading dev endpoint on NVIDIA L4 GPU
uv run modal serve modal_app.py

# Permanent production deployment
uv run modal deploy modal_app.py

# Or convert a local archive directly via Modal GPU RPC
uv run modal run modal_app.py --input-path ./archive.tar.zst --output-path ./archive.zip
```

Configure GPU hardware via `CLOUDZIP_GPU` (`L4` default, `T4`, `A10G`, `A100`, `H100`).

---

## API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, Modal runtime state, and NVIDIA nvCOMP GPU telemetry |
| `GET` | `/api/v1/formats` | Supported input archive formats and GPU output compression codecs |
| `GET` | `/api/v1/options/defaults` | Default `ConversionOptions` payload |
| `POST` | `/api/v1/inspect` | Inspect an archive's contents, directory structure, and format without converting |
| `POST` | `/api/v1/jobs` | Upload an archive and launch an async GPU conversion job (`202 Accepted`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status, completion percentage, and compression telemetry |
| `GET` | `/api/v1/jobs/{job_id}/progress` | Stream live `0%–100%` `ProgressEvent` frames (`text/event-stream` SSE or NDJSON) |
| `GET` | `/api/v1/jobs/{job_id}/download` | Download the finished `.zip` archive |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete job state and temporary artifacts |
| `POST` | `/api/v1/convert/stream` | Upload + stream `0%–100%` SSE/NDJSON progress in one HTTP call |
| `POST` | `/api/v1/convert` | Upload + receive converted `.zip` binary directly |

---

## License

Copyright (c) 2026 Alexander Scott. Released under the [MIT License](./LICENSE).
