"""End-to-end archive manipulation & nvCOMP GPU ZIP conversion pipeline with percentage streaming.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from cloudzip.config import CloudZipSettings
from cloudzip.extractor import ArchiveError, detect_archive_format, extract_archive
from cloudzip.manipulator import ManipulatedEntry, apply_manipulation_pipeline
from cloudzip.models import (
    CompressionStats,
    ConversionOptions,
    ConversionStage,
    JobResultSummary,
    ProgressEvent,
)
from cloudzip.nvcomp_engine import ZIP_METHOD_STORE, NvcompCompressionEngine
from cloudzip.zip_writer import CloudZipWriter, ZipMemberSpec, verify_zip_integrity

EventSink = Callable[[ProgressEvent], None]

COMPOUND_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
    ".tar.lz4",
    ".tar.br",
    ".tar.lzma",
    ".tar.snappy",
    ".tgz",
    ".tbz",
    ".tbz2",
    ".txz",
    ".tzst",
    ".tlz",
    ".tlz4",
    ".tbr",
    ".7z",
    ".rar",
    ".cbr",
    ".cbz",
    ".zip",
    ".jar",
    ".war",
    ".ear",
    ".apk",
    ".ipa",
    ".whl",
    ".epub",
    ".xpi",
    ".iso",
    ".udf",
    ".cab",
    ".deb",
    ".udeb",
    ".ar",
    ".rpm",
    ".cpio",
    ".xar",
    ".pkg",
    ".wim",
    ".swm",
    ".esd",
    ".lzh",
    ".lha",
    ".arj",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".lzma",
    ".zst",
    ".lz4",
    ".br",
    ".snappy",
    ".sz",
)


def derive_output_zip_filename(input_filename: str, custom_filename: str | None = None) -> str:
    """Compute a clean, safe `.zip` download filename."""
    if custom_filename:
        base = Path(custom_filename.replace("\\", "/")).name.strip()
        base = re.sub(r"[^\w.\- ()+\[\]]+", "_", base)
        if not base.lower().endswith(".zip"):
            base = f"{base}.zip"
        return base if base != ".zip" else "converted.zip"

    clean_in = Path(input_filename.replace("\\", "/")).name.strip() or "archive"
    lower = clean_in.lower()
    stem = clean_in
    for sfx in COMPOUND_ARCHIVE_SUFFIXES:
        if lower.endswith(sfx) and len(clean_in) > len(sfx):
            stem = clean_in[: -len(sfx)]
            break
    stem = re.sub(r"[^\w.\- ()+\[\]]+", "_", stem).strip("._ ") or "archive"
    return f"{stem}.zip"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _partition_file_batches(
    file_entries: list[ManipulatedEntry],
    max_batch_bytes: int,
    max_files_per_batch: int = 128,
) -> list[list[ManipulatedEntry]]:
    batches: list[list[ManipulatedEntry]] = []
    cur_batch: list[ManipulatedEntry] = []
    cur_bytes = 0

    for entry in file_entries:
        if cur_batch and (
            cur_bytes + entry.size_bytes > max_batch_bytes
            or len(cur_batch) >= max_files_per_batch
        ):
            batches.append(cur_batch)
            cur_batch = []
            cur_bytes = 0
        cur_batch.append(entry)
        cur_bytes += entry.size_bytes

    if cur_batch:
        batches.append(cur_batch)
    return batches


def run_conversion_pipeline(
    *,
    job_id: str,
    source_archive_path: Path,
    original_filename: str,
    output_zip_path: Path,
    options: ConversionOptions,
    engine: NvcompCompressionEngine,
    settings: CloudZipSettings,
    download_url: str,
    emit: EventSink | None = None,
) -> JobResultSummary:
    """Execute full archive -> manipulated nvCOMP GPU ZIP conversion with monotonic 0-100% progress."""
    t_start = time.perf_counter()
    highest_pct = 0.0

    def _send(
        stage: ConversionStage,
        pct: float,
        message: str,
        *,
        current_entry: str | None = None,
        entries_processed: int = 0,
        entries_total: int = 0,
        bytes_processed: int = 0,
        bytes_total: int = 0,
        throughput_mbps: float | None = None,
        result: JobResultSummary | None = None,
    ) -> None:
        nonlocal highest_pct
        clamped = round(max(highest_pct, min(100.0, pct)), 2)
        highest_pct = clamped
        if emit is not None:
            emit(
                ProgressEvent(
                    job_id=job_id,
                    stage=stage,
                    percentage=clamped,
                    message=message,
                    current_entry=current_entry,
                    entries_processed=entries_processed,
                    entries_total=entries_total,
                    bytes_processed=bytes_processed,
                    bytes_total=bytes_total,
                    throughput_mbps=round(throughput_mbps, 2)
                    if throughput_mbps is not None
                    else None,
                    backend=engine.active_backend,
                    result=result,
                )
            )

    original_archive_bytes = source_archive_path.stat().st_size
    output_filename = derive_output_zip_filename(original_filename, options.output_filename)

    _send(
        ConversionStage.INITIALIZING,
        1.0,
        f"Initializing CloudZIP conversion for '{original_filename}' ({original_archive_bytes:,} bytes).",
    )

    prelim_fmt = detect_archive_format(source_archive_path, original_filename)
    _send(
        ConversionStage.DETECTING,
        4.0,
        f"Detected source archive format '{prelim_fmt}'. Preparing extraction sandbox.",
    )

    with tempfile.TemporaryDirectory(prefix=f"cloudzip_work_{job_id}_") as work_tmp:
        staging_dir = Path(work_tmp) / "extracted"

        last_extract_emit = 0.0

        def _on_extract_progress(done: int, total: int, member_name: str) -> None:
            nonlocal last_extract_emit
            now = time.perf_counter()
            # Throttle intermediate extraction events to avoid flooding on 50k-file archives
            if done < total and (now - last_extract_emit) < 0.04:
                return
            last_extract_emit = now
            frac = done / max(1, total)
            _send(
                ConversionStage.EXTRACTING,
                5.0 + 25.0 * frac,
                f"Extracting archive entries ({done}/{total})...",
                current_entry=member_name,
                entries_processed=done,
                entries_total=total,
            )

        detected_fmt, _extractor_backend, discovered_items = extract_archive(
            source_archive_path,
            staging_dir,
            original_filename=original_filename,
            password=options.source_password,
            extract_nested=options.extract_nested_archives,
            max_nested_depth=options.max_nested_depth,
            max_uncompressed_bytes=settings.max_uncompressed_bytes,
            max_entries=settings.max_archive_entries,
            on_progress=_on_extract_progress,
        )

        entries_discovered = len(discovered_items)
        _send(
            ConversionStage.EXTRACTING,
            30.0,
            f"Extracted {entries_discovered} entries from '{detected_fmt}' archive.",
            entries_processed=entries_discovered,
            entries_total=entries_discovered,
        )

        _send(
            ConversionStage.MANIPULATING,
            32.0,
            "Applying path transformations, glob filters, rename rules, and batch schedule...",
        )
        kept_entries, filtered_out_count = apply_manipulation_pipeline(discovered_items, options)

        dir_entries = [e for e in kept_entries if e.is_dir]
        file_entries = [e for e in kept_entries if not e.is_dir]
        total_included = len(kept_entries)
        total_uncomp_bytes = sum(e.size_bytes for e in file_entries)

        _send(
            ConversionStage.MANIPULATING,
            35.0,
            (
                f"Prepared {total_included} entries ({len(file_entries)} files, "
                f"{len(dir_entries)} directories; {filtered_out_count} filtered out)."
            ),
            entries_processed=0,
            entries_total=total_included,
            bytes_processed=0,
            bytes_total=total_uncomp_bytes,
        )

        output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        batch_limit_bytes = options.gpu_batch_size_mb * 1024 * 1024
        zstd_chunk_bytes = options.zstd_chunk_size_mb * 1024 * 1024
        batches = _partition_file_batches(file_entries, max_batch_bytes=batch_limit_bytes)

        compressed_payload_bytes = 0
        gpu_compress_ms = 0.0
        entries_gpu_compressed = 0
        entries_smart_stored = 0
        processed_entries = 0
        processed_uncomp_bytes = 0

        with CloudZipWriter(
            output_zip_path,
            zip64_mode=options.zip64_mode,
            password=options.output_password,
            encryption_method=options.encryption_method,
            preserve_timestamps=options.preserve_timestamps,
            normalize_timestamps=options.normalize_timestamps,
            preserve_permissions=options.preserve_permissions,
            archive_comment=options.archive_comment,
        ) as zip_writer:
            # 1. Write directory entries
            for d_entry in dir_entries:
                zip_writer.write_member(
                    ZipMemberSpec(
                        arcname=d_entry.arcname,
                        is_dir=True,
                        zip_method_id=ZIP_METHOD_STORE,
                        payload=b"",
                        uncompressed_size=0,
                        crc32=0,
                        mtime=d_entry.mtime,
                        posix_mode=d_entry.posix_mode,
                    )
                )
                processed_entries += 1

            if dir_entries:
                _send(
                    ConversionStage.COMPRESSING,
                    37.0,
                    f"Wrote {len(dir_entries)} directory headers. Starting {options.compression_method.value.upper()} compression...",
                    entries_processed=processed_entries,
                    entries_total=total_included,
                    bytes_processed=0,
                    bytes_total=total_uncomp_bytes,
                )

            # 2. Compress & package file batches via nvCOMP GPU engine
            total_batches = max(1, len(batches))
            for batch_idx, batch in enumerate(batches, start=1):
                raw_buffers = [entry.disk_path.read_bytes() for entry in batch]
                batch_uncomp = sum(len(b) for b in raw_buffers)

                t_batch = time.perf_counter()
                comp_results = engine.compress_batch(
                    raw_buffers,
                    method=options.compression_method,
                    compression_level=options.compression_level,
                    smart_store=options.smart_store,
                    zstd_chunk_bytes=zstd_chunk_bytes,
                )
                batch_ms = (time.perf_counter() - t_batch) * 1000.0
                gpu_compress_ms += batch_ms
                batch_mbps = (
                    (batch_uncomp / (1024 * 1024)) / (batch_ms / 1000.0)
                    if batch_ms > 0 and batch_uncomp > 0
                    else 0.0
                )

                # Release raw buffers as we write compressed members
                del raw_buffers

                for entry, c_res in zip(batch, comp_results, strict=True):
                    written_comp = zip_writer.write_member(
                        ZipMemberSpec(
                            arcname=entry.arcname,
                            is_dir=False,
                            zip_method_id=c_res.zip_method_id,
                            payload=c_res.payload,
                            uncompressed_size=c_res.uncompressed_size,
                            crc32=c_res.crc32,
                            mtime=entry.mtime,
                            posix_mode=entry.posix_mode,
                        )
                    )
                    compressed_payload_bytes += written_comp
                    if c_res.zip_method_id == ZIP_METHOD_STORE:
                        entries_smart_stored += 1
                    else:
                        entries_gpu_compressed += 1
                    processed_entries += 1
                    processed_uncomp_bytes += entry.size_bytes

                if total_uncomp_bytes > 0:
                    comp_frac = processed_uncomp_bytes / total_uncomp_bytes
                else:
                    comp_frac = batch_idx / total_batches

                pct_now = 37.0 + 53.0 * comp_frac
                last_name = batch[-1].arcname if batch else None
                _send(
                    ConversionStage.COMPRESSING,
                    pct_now,
                    (
                        f"Compressed batch {batch_idx}/{total_batches} via {engine.active_backend} "
                        f"({processed_entries}/{total_included} entries)."
                    ),
                    current_entry=last_name,
                    entries_processed=processed_entries,
                    entries_total=total_included,
                    bytes_processed=processed_uncomp_bytes,
                    bytes_total=total_uncomp_bytes,
                    throughput_mbps=batch_mbps,
                )

            _send(
                ConversionStage.PACKAGING,
                92.0,
                "Writing ZIP Central Directory and finalizing archive container...",
                entries_processed=processed_entries,
                entries_total=total_included,
                bytes_processed=processed_uncomp_bytes,
                bytes_total=total_uncomp_bytes,
            )

        zip64_used = zip_writer.zip64_used

    if options.verify_integrity:
        _send(
            ConversionStage.VERIFYING,
            96.0,
            "Verifying output ZIP Central Directory headers and CRC-32 checksums...",
            entries_processed=total_included,
            entries_total=total_included,
            bytes_processed=total_uncomp_bytes,
            bytes_total=total_uncomp_bytes,
        )
        try:
            verify_zip_integrity(
                output_zip_path,
                password=options.output_password,
                encryption_method=options.encryption_method,
                compression_method_hint=options.compression_method.value,
            )
        except Exception as exc:
            raise ArchiveError(f"Output ZIP integrity verification failed: {exc}") from exc

    output_zip_bytes = output_zip_path.stat().st_size
    sha256_hex = _sha256_file(output_zip_path)
    total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if total_uncomp_bytes > 0:
        ratio = round(compressed_payload_bytes / total_uncomp_bytes, 4)
        saved_bytes = total_uncomp_bytes - compressed_payload_bytes
        saved_pct = round((saved_bytes / total_uncomp_bytes) * 100.0, 2)
    else:
        ratio = 1.0
        saved_bytes = 0
        saved_pct = 0.0

    overall_mbps = (
        round((total_uncomp_bytes / (1024 * 1024)) / (gpu_compress_ms / 1000.0), 2)
        if gpu_compress_ms > 0 and total_uncomp_bytes > 0
        else 0.0
    )

    stats = CompressionStats(
        input_filename=original_filename,
        output_filename=output_filename,
        detected_input_format=detected_fmt,
        compression_method=options.compression_method,
        compression_level=options.compression_level,
        backend=engine.active_backend,
        gpu_device_name=engine.gpu_device_name,
        nvcomp_version=engine.nvcomp_version,
        encrypted_output=bool(options.output_password),
        encryption_method=options.encryption_method if options.output_password else options.encryption_method.NONE,
        zip64_used=zip64_used,
        original_archive_bytes=original_archive_bytes,
        uncompressed_bytes=total_uncomp_bytes,
        compressed_payload_bytes=compressed_payload_bytes,
        output_zip_bytes=output_zip_bytes,
        compression_ratio=ratio,
        space_saved_bytes=saved_bytes,
        space_saved_percent=saved_pct,
        entries_discovered=entries_discovered,
        entries_included=total_included,
        entries_filtered_out=filtered_out_count,
        entries_gpu_compressed=entries_gpu_compressed,
        entries_smart_stored=entries_smart_stored,
        gpu_compress_ms=round(gpu_compress_ms, 2),
        total_elapsed_ms=round(total_elapsed_ms, 2),
        throughput_mbps=overall_mbps,
        sha256=sha256_hex,
    )

    summary = JobResultSummary(
        job_id=job_id,
        download_url=download_url,
        output_filename=output_filename,
        stats=stats,
    )

    _send(
        ConversionStage.COMPLETED,
        100.0,
        (
            f"Conversion complete! Produced '{output_filename}' ({output_zip_bytes:,} bytes, "
            f"{saved_pct:.1f}% payload reduction via {engine.active_backend})."
        ),
        entries_processed=total_included,
        entries_total=total_included,
        bytes_processed=total_uncomp_bytes,
        bytes_total=total_uncomp_bytes,
        throughput_mbps=overall_mbps,
        result=summary,
    )

    return summary
