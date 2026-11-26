"""NVIDIA nvCOMP GPU-accelerated ZIP payload compression engine.

Selects raw RFC 1951 DEFLATE (ZIP Method 8) via ``nvidia.nvcomp.Codec(algorithm="Deflate",
algorithm_type=5, bitstream_kind=BitstreamKind.RAW)`` by default — delivering the smallest
GPU-accelerated compressed size in nvCOMP while remaining 100% compatible with CPU-only
client operating systems and built-in ZIP extractors worldwide. Also supports GPU-accelerated
Zstandard (ZIP Method 93, RFC 8878) with parallel multi-frame batching, plus Smart Store
(ZIP Method 0) size guards and automatic CPU fallback when no CUDA GPU is attached.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import zstandard as zstd

from cloudzip.models import CompressionMethod, GpuCapabilities

ZIP_METHOD_STORE = 0
ZIP_METHOD_DEFLATE = 8
ZIP_METHOD_ZSTD = 93

# Maximum single-entry uncompressed buffer sent in one RAW Deflate GPU kernel call (256 MiB).
# Larger individual entries use Zstd multi-frame GPU batching (when zstd) or parallel sync-flush
# chunked Deflate so GPU VRAM never overflows even on multi-gigabyte archive members.
MAX_SINGLE_DEFLATE_GPU_BYTES = 256 * 1024 * 1024


@dataclass(slots=True)
class CompressedChunkResult:
    """Compressed payload and metadata ready for PKZIP Local File Header writing."""

    zip_method_id: int
    payload: bytes
    uncompressed_size: int
    crc32: int
    smart_stored: bool
    backend_used: str
    elapsed_ms: float


def _detect_nvcomp_version() -> str | None:
    for pkg_name in ("nvidia-nvcomp-cu12", "nvidia-nvcomp-cu13", "nvidia-nvcomp-cu11"):
        try:
            return importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _detect_gpu_device_name() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


def _load_nvcomp_module() -> Any | None:
    try:
        from nvidia import nvcomp  # type: ignore[import-not-found]

        return nvcomp
    except Exception:
        return None


class NvcompCompressionEngine:
    """Thread-safe GPU-first compression engine powered by NVIDIA nvCOMP."""

    def __init__(
        self,
        *,
        require_gpu: bool = False,
        nvcomp_module: Any | None = None,
        cpu_workers: int = 4,
    ) -> None:
        self._require_gpu = require_gpu
        self._nvcomp = nvcomp_module if nvcomp_module is not None else _load_nvcomp_module()
        self._nvcomp_version = _detect_nvcomp_version()
        self._tls = threading.local()
        self._cpu_workers = max(1, cpu_workers)
        self._gpu_available, self._gpu_device_name = self._probe_gpu()
        if self._require_gpu and not self._gpu_available:
            raise RuntimeError(
                "NVIDIA CUDA GPU with nvCOMP is required (CLOUDZIP_REQUIRE_GPU=true), "
                "but no active GPU device was detected."
            )

    def _probe_gpu(self) -> tuple[bool, str | None]:
        if self._nvcomp is None:
            return False, None
        try:
            probe_data = np.frombuffer(b"CloudZIP-nvCOMP-GPU-Probe" * 16, dtype=np.uint8)
            gpu_arr = self._nvcomp.as_array(probe_data).cuda()
            codec = self._nvcomp.Codec(
                algorithm="Deflate",
                algorithm_type=5,
                bitstream_kind=self._nvcomp.BitstreamKind.RAW,
            )
            encoded = codec.encode(gpu_arr)
            raw_stream = bytes(encoded.cpu())
            # Verify standard CPU zlib raw-deflate compatibility on the GPU output!
            restored = zlib.decompress(raw_stream, wbits=-15)
            if restored != bytes(probe_data):
                return False, None
            device_name = _detect_gpu_device_name() or "NVIDIA CUDA GPU"
            return True, device_name
        except Exception:
            return False, None

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    @property
    def active_backend(self) -> str:
        return "nvcomp-cuda-gpu" if self._gpu_available else "cpu-fallback"

    @property
    def gpu_device_name(self) -> str | None:
        return self._gpu_device_name

    @property
    def nvcomp_version(self) -> str | None:
        return self._nvcomp_version

    def capabilities(self) -> GpuCapabilities:
        return GpuCapabilities(
            nvcomp_installed=self._nvcomp is not None,
            nvcomp_version=self._nvcomp_version,
            gpu_available=self._gpu_available,
            gpu_device_name=self._gpu_device_name,
            active_backend="nvcomp-cuda-gpu" if self._gpu_available else "cpu-fallback",
        )

    def _get_thread_codec(self, algorithm: str, algorithm_type: int | None = None) -> Any:
        """Return a thread-local nvcomp.Codec instance (Codec is thread-local per nvCOMP spec)."""
        cache: dict[tuple[str, int | None], Any] | None = getattr(self._tls, "codecs", None)
        if cache is None:
            cache = {}
            self._tls.codecs = cache

        key = (algorithm, algorithm_type)
        codec = cache.get(key)
        if codec is None:
            kwargs: dict[str, Any] = {
                "algorithm": algorithm,
                "bitstream_kind": self._nvcomp.BitstreamKind.RAW,
            }
            if algorithm_type is not None:
                kwargs["algorithm_type"] = algorithm_type
            codec = self._nvcomp.Codec(**kwargs)
            cache[key] = codec
        return codec

    def compress_batch(
        self,
        buffers: list[bytes],
        *,
        method: CompressionMethod = CompressionMethod.DEFLATE,
        compression_level: int = 5,
        smart_store: bool = True,
        zstd_chunk_bytes: int = 4 * 1024 * 1024,
    ) -> list[CompressedChunkResult]:
        """Compress a batch of file byte buffers in parallel on NVIDIA GPU via nvCOMP.

        Empty buffers (0 bytes) and `method == STORE` use ZIP Method 0 directly.
        When `smart_store` is True, any entry whose compressed size >= uncompressed size
        automatically reverts to ZIP Method 0 (`STORE`) for optimal archive size.
        """
        if not buffers:
            return []

        t0 = time.perf_counter()
        results: list[CompressedChunkResult | None] = [None] * len(buffers)
        crcs = [zlib.crc32(buf) & 0xFFFFFFFF for buf in buffers]

        if method == CompressionMethod.STORE:
            elapsed_each = ((time.perf_counter() - t0) * 1000.0) / max(1, len(buffers))
            return [
                CompressedChunkResult(
                    zip_method_id=ZIP_METHOD_STORE,
                    payload=buf,
                    uncompressed_size=len(buf),
                    crc32=crcs[idx],
                    smart_stored=False,
                    backend_used="store-passthrough",
                    elapsed_ms=elapsed_each,
                )
                for idx, buf in enumerate(buffers)
            ]

        # Handle empty (0-byte) files immediately — nvCOMP requires uncompressed_size > 0
        non_empty_indices: list[int] = []
        for idx, buf in enumerate(buffers):
            if len(buf) == 0:
                results[idx] = CompressedChunkResult(
                    zip_method_id=ZIP_METHOD_STORE,
                    payload=b"",
                    uncompressed_size=0,
                    crc32=0,
                    smart_stored=True,
                    backend_used=self.active_backend,
                    elapsed_ms=0.0,
                )
            else:
                non_empty_indices.append(idx)

        if not non_empty_indices:
            return [r for r in results if r is not None]

        if method == CompressionMethod.DEFLATE:
            self._compress_deflate_entries(
                buffers=buffers,
                indices=non_empty_indices,
                crcs=crcs,
                compression_level=compression_level,
                smart_store=smart_store,
                out_results=results,
            )
        elif method == CompressionMethod.ZSTD:
            self._compress_zstd_entries(
                buffers=buffers,
                indices=non_empty_indices,
                crcs=crcs,
                compression_level=compression_level,
                smart_store=smart_store,
                zstd_chunk_bytes=zstd_chunk_bytes,
                out_results=results,
            )
        else:
            raise ValueError(f"Unsupported compression method: {method}")

        return [r for r in results if r is not None]

    def _compress_deflate_entries(
        self,
        *,
        buffers: list[bytes],
        indices: list[int],
        crcs: list[int],
        compression_level: int,
        smart_store: bool,
        out_results: list[CompressedChunkResult | None],
    ) -> None:
        algo_type = max(1, min(5, compression_level))
        gpu_indices = [
            i for i in indices if self._gpu_available and len(buffers[i]) <= MAX_SINGLE_DEFLATE_GPU_BYTES
        ]
        cpu_indices = [i for i in indices if i not in set(gpu_indices)]

        if gpu_indices:
            t_gpu = time.perf_counter()
            try:
                codec = self._get_thread_codec("Deflate", algorithm_type=algo_type)
                if len(gpu_indices) == 1:
                    only_idx = gpu_indices[0]
                    arr = np.frombuffer(buffers[only_idx], dtype=np.uint8)
                    encoded = codec.encode(self._nvcomp.as_array(arr).cuda()).cpu()
                    comp_payloads = [bytes(encoded)]
                else:
                    host_arrays = [np.frombuffer(buffers[i], dtype=np.uint8) for i in gpu_indices]
                    gpu_batch = self._nvcomp.as_arrays(host_arrays).cuda()
                    comp_batch = codec.encode(gpu_batch).cpu()
                    comp_payloads = [bytes(comp_batch[pos]) for pos in range(len(gpu_indices))]

                elapsed_each = ((time.perf_counter() - t_gpu) * 1000.0) / len(gpu_indices)
                for pos, orig_idx in enumerate(gpu_indices):
                    raw_buf = buffers[orig_idx]
                    comp_buf = comp_payloads[pos]
                    out_results[orig_idx] = self._finalize_entry(
                        raw_buf=raw_buf,
                        comp_buf=comp_buf,
                        target_zip_method=ZIP_METHOD_DEFLATE,
                        crc32=crcs[orig_idx],
                        smart_store=smart_store,
                        backend="nvcomp-cuda-gpu",
                        elapsed_ms=elapsed_each,
                    )
            except Exception:
                if self._require_gpu:
                    raise
                cpu_indices.extend(gpu_indices)

        if cpu_indices:
            zlib_level = {1: 1, 2: 4, 3: 6, 4: 7, 5: 9}.get(algo_type, 9)
            for orig_idx in cpu_indices:
                t_cpu = time.perf_counter()
                raw_buf = buffers[orig_idx]
                comp_buf = self._compress_deflate_cpu(raw_buf, level=zlib_level)
                elapsed_ms = (time.perf_counter() - t_cpu) * 1000.0
                out_results[orig_idx] = self._finalize_entry(
                    raw_buf=raw_buf,
                    comp_buf=comp_buf,
                    target_zip_method=ZIP_METHOD_DEFLATE,
                    crc32=crcs[orig_idx],
                    smart_store=smart_store,
                    backend="cpu-fallback",
                    elapsed_ms=elapsed_ms,
                )

    def _compress_deflate_cpu(self, data: bytes, *, level: int = 9) -> bytes:
        chunk_size = 4 * 1024 * 1024
        if len(data) <= chunk_size or self._cpu_workers == 1:
            comp = zlib.compressobj(level=level, method=zlib.DEFLATED, wbits=-15, memLevel=9)
            return comp.compress(data) + comp.flush(zlib.Z_FINISH)

        # Parallel multi-chunk RFC 1951 raw DEFLATE using Z_SYNC_FLUSH on non-final chunks
        slices = [data[offset : offset + chunk_size] for offset in range(0, len(data), chunk_size)]
        last_idx = len(slices) - 1

        def _encode_slice(pair: tuple[int, bytes]) -> bytes:
            idx, chunk = pair
            c = zlib.compressobj(level=level, method=zlib.DEFLATED, wbits=-15, memLevel=9)
            flush_mode = zlib.Z_FINISH if idx == last_idx else zlib.Z_SYNC_FLUSH
            return c.compress(chunk) + c.flush(flush_mode)

        with ThreadPoolExecutor(max_workers=self._cpu_workers) as pool:
            parts = list(pool.map(_encode_slice, enumerate(slices)))
        return b"".join(parts)

    def _compress_zstd_entries(
        self,
        *,
        buffers: list[bytes],
        indices: list[int],
        crcs: list[int],
        compression_level: int,
        smart_store: bool,
        zstd_chunk_bytes: int,
        out_results: list[CompressedChunkResult | None],
    ) -> None:
        safe_chunk = max(64 * 1024, min(16 * 1024 * 1024, zstd_chunk_bytes))
        if self._gpu_available:
            t_gpu = time.perf_counter()
            try:
                codec = self._get_thread_codec("Zstd")
                # Slice every file into <=16 MiB chunks so nvCOMP Zstd processes all chunks
                # in a single parallel BatchArray launch, then reassemble valid RFC 8878 frames!
                flat_chunks: list[np.ndarray] = []
                entry_ranges: list[tuple[int, int, int]] = []
                for orig_idx in indices:
                    raw_buf = buffers[orig_idx]
                    start_pos = len(flat_chunks)
                    for offset in range(0, len(raw_buf), safe_chunk):
                        flat_chunks.append(
                            np.frombuffer(raw_buf[offset : offset + safe_chunk], dtype=np.uint8)
                        )
                    entry_ranges.append((orig_idx, start_pos, len(flat_chunks)))

                if len(flat_chunks) == 1:
                    comp_frames = [bytes(codec.encode(self._nvcomp.as_array(flat_chunks[0]).cuda()).cpu())]
                else:
                    gpu_batch = self._nvcomp.as_arrays(flat_chunks).cuda()
                    comp_batch = codec.encode(gpu_batch).cpu()
                    comp_frames = [bytes(comp_batch[i]) for i in range(len(flat_chunks))]

                elapsed_each = ((time.perf_counter() - t_gpu) * 1000.0) / len(indices)
                for orig_idx, start_pos, end_pos in entry_ranges:
                    raw_buf = buffers[orig_idx]
                    comp_buf = b"".join(comp_frames[start_pos:end_pos])
                    out_results[orig_idx] = self._finalize_entry(
                        raw_buf=raw_buf,
                        comp_buf=comp_buf,
                        target_zip_method=ZIP_METHOD_ZSTD,
                        crc32=crcs[orig_idx],
                        smart_store=smart_store,
                        backend="nvcomp-cuda-gpu",
                        elapsed_ms=elapsed_each,
                    )
                return
            except Exception:
                if self._require_gpu:
                    raise

        zstd_level = {1: 3, 2: 6, 3: 10, 4: 15, 5: 19}.get(compression_level, 19)
        compressor = zstd.ZstdCompressor(level=zstd_level, threads=-1)
        for orig_idx in indices:
            t_cpu = time.perf_counter()
            raw_buf = buffers[orig_idx]
            comp_buf = compressor.compress(raw_buf)
            elapsed_ms = (time.perf_counter() - t_cpu) * 1000.0
            out_results[orig_idx] = self._finalize_entry(
                raw_buf=raw_buf,
                comp_buf=comp_buf,
                target_zip_method=ZIP_METHOD_ZSTD,
                crc32=crcs[orig_idx],
                smart_store=smart_store,
                backend="cpu-fallback",
                elapsed_ms=elapsed_ms,
            )

    @staticmethod
    def _finalize_entry(
        *,
        raw_buf: bytes,
        comp_buf: bytes,
        target_zip_method: int,
        crc32: int,
        smart_store: bool,
        backend: str,
        elapsed_ms: float,
    ) -> CompressedChunkResult:
        if smart_store and len(comp_buf) >= len(raw_buf):
            return CompressedChunkResult(
                zip_method_id=ZIP_METHOD_STORE,
                payload=raw_buf,
                uncompressed_size=len(raw_buf),
                crc32=crc32,
                smart_stored=True,
                backend_used=backend,
                elapsed_ms=elapsed_ms,
            )
        return CompressedChunkResult(
            zip_method_id=target_zip_method,
            payload=comp_buf,
            uncompressed_size=len(raw_buf),
            crc32=crc32,
            smart_stored=False,
            backend_used=backend,
            elapsed_ms=elapsed_ms,
        )
