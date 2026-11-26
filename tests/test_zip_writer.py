"""Unit tests for spec-compliant PKZIP / ZIP64 / AES-256 / Zstd binary writer.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pyzipper

from cloudzip.models import CompressionMethod, EncryptionMethod, Zip64Mode
from cloudzip.nvcomp_engine import ZIP_METHOD_STORE, NvcompCompressionEngine
from cloudzip.zip_writer import CloudZipWriter, ZipMemberSpec, verify_zip_integrity


def test_zip_writer_deflate_unicode_zip64_and_reproducible_hash(tmp_path: Path) -> None:
    engine = NvcompCompressionEngine(require_gpu=False, nvcomp_module=None)
    content_a = b"Hello CPU-compatible ZIP world! " * 300
    content_b = "Unicode greeting: café 🚀 日本語".encode("utf-8") * 50

    comp_results = engine.compress_batch(
        [content_a, content_b],
        method=CompressionMethod.DEFLATE,
        compression_level=5,
    )

    hashes: list[str] = []
    for idx in range(2):
        out_zip = tmp_path / f"repro_{idx}.zip"
        with CloudZipWriter(
            out_zip,
            zip64_mode=Zip64Mode.FORCE,
            normalize_timestamps=True,
            preserve_permissions=True,
            archive_comment="CloudZIP Test Archive - Copyright Alexander Scott",
        ) as writer:
            writer.write_member(
                ZipMemberSpec(
                    arcname="empty_folder",
                    is_dir=True,
                    zip_method_id=ZIP_METHOD_STORE,
                    payload=b"",
                    uncompressed_size=0,
                    crc32=0,
                    mtime=datetime.now(timezone.utc),
                    posix_mode=0o755,
                )
            )
            writer.write_member(
                ZipMemberSpec(
                    arcname="docs/readme.txt",
                    is_dir=False,
                    zip_method_id=comp_results[0].zip_method_id,
                    payload=comp_results[0].payload,
                    uncompressed_size=comp_results[0].uncompressed_size,
                    crc32=comp_results[0].crc32,
                    mtime=datetime.now(timezone.utc),
                    posix_mode=0o755,
                )
            )
            writer.write_member(
                ZipMemberSpec(
                    arcname="ユニコード/🚀_notes.txt",
                    is_dir=False,
                    zip_method_id=comp_results[1].zip_method_id,
                    payload=comp_results[1].payload,
                    uncompressed_size=comp_results[1].uncompressed_size,
                    crc32=comp_results[1].crc32,
                    mtime=datetime.now(timezone.utc),
                    posix_mode=0o640,
                )
            )
        assert writer.zip64_used is True
        verify_zip_integrity(out_zip)
        hashes.append(hashlib.sha256(out_zip.read_bytes()).hexdigest())

        with zipfile.ZipFile(out_zip, "r") as zf:
            assert zf.testzip() is None
            assert zf.comment == b"CloudZIP Test Archive - Copyright Alexander Scott"
            assert zf.read("docs/readme.txt") == content_a
            assert zf.read("ユニコード/🚀_notes.txt") == content_b
            assert "empty_folder/" in zf.namelist()

    # Deterministic reproducible builds must produce identical SHA-256 digests
    assert hashes[0] == hashes[1]


def test_zip_writer_aes256_and_zipcrypto_encryption(tmp_path: Path) -> None:
    engine = NvcompCompressionEngine(require_gpu=False, nvcomp_module=None)
    secret_payload = b"Top-secret GPU-compressed archive content! " * 250
    comp = engine.compress_batch([secret_payload], method=CompressionMethod.DEFLATE)[0]

    for method in (EncryptionMethod.AES256, EncryptionMethod.ZIPCRYPTO):
        enc_zip = tmp_path / f"encrypted_{method.value}.zip"
        with CloudZipWriter(
            enc_zip,
            password="CorrectHorseBatteryStaple!",
            encryption_method=method,
        ) as writer:
            writer.write_member(
                ZipMemberSpec(
                    arcname="vault/secret.txt",
                    is_dir=False,
                    zip_method_id=comp.zip_method_id,
                    payload=comp.payload,
                    uncompressed_size=comp.uncompressed_size,
                    crc32=comp.crc32,
                    mtime=datetime.now(timezone.utc),
                )
            )

        verify_zip_integrity(
            enc_zip,
            password="CorrectHorseBatteryStaple!",
            encryption_method=method,
        )
        with pyzipper.AESZipFile(enc_zip, "r") as azf:
            azf.setpassword(b"CorrectHorseBatteryStaple!")
            assert azf.read("vault/secret.txt") == secret_payload


def test_zip_writer_zstd_method93_verification(tmp_path: Path) -> None:
    engine = NvcompCompressionEngine(require_gpu=False, nvcomp_module=None)
    payload = b"Zstandard Method 93 inside PKWARE ZIP container! " * 400
    comp = engine.compress_batch([payload], method=CompressionMethod.ZSTD)[0]

    zstd_zip = tmp_path / "zstd93.zip"
    with CloudZipWriter(zstd_zip) as writer:
        writer.write_member(
            ZipMemberSpec(
                arcname="data/frame.bin",
                is_dir=False,
                zip_method_id=comp.zip_method_id,
                payload=comp.payload,
                uncompressed_size=comp.uncompressed_size,
                crc32=comp.crc32,
                mtime=datetime.now(timezone.utc),
            )
        )

    verify_zip_integrity(zstd_zip, compression_method_hint="zstd")
