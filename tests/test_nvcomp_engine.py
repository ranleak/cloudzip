"""Unit tests for NVIDIA nvCOMP GPU compression engine & CPU fallback.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import io
import os
import threading
import zlib
from enum import IntEnum

import numpy as np
import pytest
import zstandard as zstd

from cloudzip.models import CompressionMethod
from cloudzip.nvcomp_engine import (
    ZIP_METHOD_DEFLATE,
    ZIP_METHOD_STORE,
    ZIP_METHOD_ZSTD,
    NvcompCompressionEngine,
)


class _FakeBitstreamKind(IntEnum):
    NVCOMP_NATIVE = 0
    RAW = 1
    WITH_UNCOMPRESSED_SIZE = 2


class _FakeArray:
    def __init__(self, payload: bytes, on_gpu: bool = False) -> None:
        self._payload = payload
        self._on_gpu = on_gpu

    def cuda(self) -> "_FakeArray":
        return _FakeArray(self._payload, on_gpu=True)

    def cpu(self) -> "_FakeArray":
        return _FakeArray(self._payload, on_gpu=False)

    def __bytes__(self) -> bytes:
        return self._payload


class _FakeBatchArray:
    def __init__(self, items: list[bytes], on_gpu: bool = False) -> None:
        self._items = items
        self._on_gpu = on_gpu

    def cuda(self) -> "_FakeBatchArray":
        return _FakeBatchArray(self._items, on_gpu=True)

    def cpu(self) -> "_FakeBatchArray":
        return _FakeBatchArray(self._items, on_gpu=False)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> _FakeArray:
        return _FakeArray(self._items[idx], on_gpu=self._on_gpu)


class _FakeNvcompModule:
    """Simulates `nvidia.nvcomp` GPU execution while validating Codec kwargs & thread safety."""

    BitstreamKind = _FakeBitstreamKind

    def __init__(self) -> None:
        self.codec_creations: list[tuple[int, dict[str, object]]] = []
        self._lock = threading.Lock()

    def as_array(self, arr: np.ndarray) -> _FakeArray:
        return _FakeArray(arr.tobytes())

    def as_arrays(self, arrays: list[np.ndarray]) -> _FakeBatchArray:
        return _FakeBatchArray([a.tobytes() for a in arrays])

    def Codec(self, **kwargs: object) -> "_FakeCodec":
        tid = threading.get_ident()
        with self._lock:
            self.codec_creations.append((tid, dict(kwargs)))
        return _FakeCodec(tid, **kwargs)


class _FakeCodec:
    def __init__(self, owner_tid: int, **kwargs: object) -> None:
        self.owner_tid = owner_tid
        self.algorithm = str(kwargs.get("algorithm", ""))
        self.algorithm_type = kwargs.get("algorithm_type")
        self.bitstream_kind = kwargs.get("bitstream_kind")
        if self.bitstream_kind != _FakeBitstreamKind.RAW:
            raise AssertionError("ZIP payloads MUST use BitstreamKind.RAW (no nvCOMP header)!")

    def _encode_one(self, raw: bytes) -> bytes:
        if len(raw) == 0:
            raise ValueError("nvCOMP requires uncompressed_size > 0")
        if self.algorithm == "Deflate":
            lvl = {1: 1, 2: 4, 3: 6, 4: 7, 5: 9}.get(int(self.algorithm_type or 1), 9)
            c = zlib.compressobj(level=lvl, method=zlib.DEFLATED, wbits=-15)
            return c.compress(raw) + c.flush(zlib.Z_FINISH)
        if self.algorithm == "Zstd":
            return zstd.ZstdCompressor(level=10).compress(raw)
        raise ValueError(f"Unexpected algorithm {self.algorithm}")

    def encode(self, src: _FakeArray | _FakeBatchArray) -> _FakeArray | _FakeBatchArray:
        if threading.get_ident() != self.owner_tid:
            raise RuntimeError("Thread-safety violation: Codec shared across threads!")
        if isinstance(src, _FakeArray):
            assert src._on_gpu
            return _FakeArray(self._encode_one(src._payload), on_gpu=True)
        assert src._on_gpu
        return _FakeBatchArray([self._encode_one(item) for item in src._items], on_gpu=True)


def test_nvcomp_gpu_deflate_level5_and_smart_store() -> None:
    fake_nvcomp = _FakeNvcompModule()
    engine = NvcompCompressionEngine(require_gpu=True, nvcomp_module=fake_nvcomp)
    assert engine.gpu_available is True
    assert engine.active_backend == "nvcomp-cuda-gpu"

    compressible = b"NVIDIA nvCOMP GPU-accelerated ZIP compression! " * 500
    incompressible = os.urandom(4096)
    empty = b""

    results = engine.compress_batch(
        [compressible, incompressible, empty],
        method=CompressionMethod.DEFLATE,
        compression_level=5,
        smart_store=True,
    )
    assert len(results) == 3

    # Entry 0: compressed via GPU Deflate level 5 (ZIP Method 8)
    assert results[0].zip_method_id == ZIP_METHOD_DEFLATE
    assert results[0].smart_stored is False
    assert results[0].backend_used == "nvcomp-cuda-gpu"
    assert len(results[0].payload) < len(compressible) // 10
    assert zlib.decompress(results[0].payload, wbits=-15) == compressible

    # Entry 1: random bytes -> Smart Store automatically selects ZIP Method 0!
    assert results[1].zip_method_id == ZIP_METHOD_STORE
    assert results[1].smart_stored is True
    assert results[1].payload == incompressible

    # Entry 2: 0-byte empty file -> ZIP Method 0 without calling nvCOMP on empty buffer
    assert results[2].zip_method_id == ZIP_METHOD_STORE
    assert results[2].payload == b""
    assert results[2].crc32 == 0

    # Verify algorithm_type=5 and BitstreamKind.RAW were passed to nvCOMP Codec
    assert any(
        kw.get("algorithm") == "Deflate"
        and kw.get("algorithm_type") == 5
        and kw.get("bitstream_kind") == _FakeBitstreamKind.RAW
        for _, kw in fake_nvcomp.codec_creations
    )


def test_nvcomp_gpu_zstd_multiframe_concatenation() -> None:
    fake_nvcomp = _FakeNvcompModule()
    engine = NvcompCompressionEngine(require_gpu=True, nvcomp_module=fake_nvcomp)

    large_data = (b"Multi-frame GPU Zstandard payload segment! " * 8000)  # ~344 KB
    results = engine.compress_batch(
        [large_data],
        method=CompressionMethod.ZSTD,
        compression_level=5,
        smart_store=True,
        zstd_chunk_bytes=64 * 1024,  # Force multiple 64 KiB frames in BatchArray
    )
    assert len(results) == 1
    assert results[0].zip_method_id == ZIP_METHOD_ZSTD
    assert results[0].backend_used == "nvcomp-cuda-gpu"

    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(results[0].payload)) as reader:
        decompressed = reader.read()
    assert decompressed == large_data


def test_nvcomp_thread_local_codec_safety() -> None:
    fake_nvcomp = _FakeNvcompModule()
    engine = NvcompCompressionEngine(require_gpu=True, nvcomp_module=fake_nvcomp)
    errors: list[Exception] = []

    def _worker(tid: int) -> None:
        try:
            sample = f"Thread-{tid}-payload-".encode() * 400
            res = engine.compress_batch([sample], method=CompressionMethod.DEFLATE)
            assert zlib.decompress(res[0].payload, wbits=-15) == sample
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_cpu_fallback_and_require_gpu_guard() -> None:
    with pytest.raises(RuntimeError, match="NVIDIA CUDA GPU with nvCOMP is required"):
        NvcompCompressionEngine(require_gpu=True, nvcomp_module=None)

    cpu_engine = NvcompCompressionEngine(require_gpu=False, nvcomp_module=None)
    assert cpu_engine.gpu_available is False
    assert cpu_engine.active_backend == "cpu-fallback"

    big_text = b"Parallel sync-flush raw DEFLATE verification block! " * 100_000  # ~5.2 MB
    res = cpu_engine.compress_batch([big_text], method=CompressionMethod.DEFLATE, compression_level=5)
    assert res[0].zip_method_id == ZIP_METHOD_DEFLATE
    assert zlib.decompress(res[0].payload, wbits=-15) == big_text
