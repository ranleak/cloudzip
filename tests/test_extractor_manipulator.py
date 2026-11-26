"""Unit tests for universal multi-format archive extraction & manipulation options.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import tarfile
import zipfile
from pathlib import Path

import brotli
import cramjam
import lz4.frame
import py7zr
import pytest
import pyzipper
import zstandard as zstd

from cloudzip.extractor import (
    ArchivePasswordError,
    detect_archive_format,
    extract_archive,
    sanitize_archive_path,
)
from cloudzip.manipulator import apply_manipulation_pipeline
from cloudzip.models import ConversionOptions, RenameRule

SAMPLE_FILES = {
    "project-v1/src/main.py": b"print('Hello from CloudZIP!')\n" * 40,
    "project-v1/docs/guide.md": b"# User Guide\nDetailed instructions.\n" * 30,
    "project-v1/__MACOSX/._main.py": b"junk",
    "project-v1/.DS_Store": b"junk_ds",
    "project-v1/.hidden/secret.env": b"TOKEN=123",
}


def _make_tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in SAMPLE_FILES.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize(
    ("filename", "builder", "expected_fmt"),
    [
        ("bundle.tar", lambda raw: raw, "tar"),
        ("bundle.tar.gz", gzip.compress, "tar.gz"),
        ("bundle.tgz", gzip.compress, "tar.gz"),
        ("bundle.tar.bz2", bz2.compress, "tar.bz2"),
        ("bundle.tar.xz", lzma.compress, "tar.xz"),
        ("bundle.tar.zst", lambda raw: zstd.ZstdCompressor().compress(raw), "tar.zst"),
        ("bundle.tar.lz4", lz4.frame.compress, "tar.lz4"),
        ("bundle.tar.br", brotli.compress, "tar.br"),
        # Even without `.tar` in filename, stream-wrapped tar is auto-unwrapped!
        ("mystery_upload.bin", lambda raw: zstd.ZstdCompressor().compress(raw), "tar.zst"),
    ],
)
def test_extract_compressed_tar_variants(
    tmp_path: Path, filename: str, builder: object, expected_fmt: str
) -> None:
    tar_raw = _make_tar_bytes()
    archive_file = tmp_path / filename
    archive_file.write_bytes(builder(tar_raw))  # type: ignore[operator]

    out_dir = tmp_path / f"out_{filename}"
    fmt, _backend, items = extract_archive(
        archive_file, out_dir, original_filename=filename
    )
    assert fmt == expected_fmt
    by_path = {it.rel_path: it for it in items if not it.is_dir}
    assert by_path["project-v1/src/main.py"].disk_path.read_bytes() == SAMPLE_FILES[
        "project-v1/src/main.py"
    ]
    assert by_path["project-v1/docs/guide.md"].disk_path.read_bytes() == SAMPLE_FILES[
        "project-v1/docs/guide.md"
    ]


def test_extract_7z_and_encrypted_7z_and_aes_zip(tmp_path: Path) -> None:
    # 1. Plain & encrypted .7z
    src_file = tmp_path / "hello.txt"
    src_file.write_bytes(b"7-Zip extraction test! " * 100)

    enc_7z = tmp_path / "encrypted.7z"
    with py7zr.SevenZipFile(enc_7z, "w", password="7zSecretPassword") as sz:
        sz.write(src_file, arcname="nested/hello.txt")

    assert detect_archive_format(enc_7z) == "7z"
    with pytest.raises(ArchivePasswordError):
        extract_archive(enc_7z, tmp_path / "fail_7z")

    fmt, _, items = extract_archive(
        enc_7z, tmp_path / "ok_7z", password="7zSecretPassword"
    )
    assert fmt == "7z"
    files = [i for i in items if not i.is_dir]
    assert len(files) == 1
    assert files[0].rel_path == "nested/hello.txt"
    assert files[0].disk_path.read_bytes() == src_file.read_bytes()

    # 2. WinZip AES-256 encrypted source .zip
    aes_zip = tmp_path / "aes_input.zip"
    with pyzipper.AESZipFile(
        aes_zip, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
    ) as azf:
        azf.setpassword(b"ZipAesPass")
        azf.writestr("secure/doc.txt", b"AES-encrypted input ZIP content!")

    with pytest.raises(ArchivePasswordError):
        extract_archive(aes_zip, tmp_path / "fail_aes")

    _, _, z_items = extract_archive(
        aes_zip, tmp_path / "ok_aes", password="ZipAesPass"
    )
    z_files = [i for i in z_items if not i.is_dir]
    assert z_files[0].disk_path.read_bytes() == b"AES-encrypted input ZIP content!"


def test_extract_deb_cpio_and_single_streams(tmp_path: Path) -> None:
    # 1. Single-file Snappy stream
    snappy_path = tmp_path / "report.csv.snappy"
    csv_bytes = b"id,name\n1,CloudZIP\n2,nvCOMP\n" * 50
    snappy_path.write_bytes(bytes(cramjam.snappy.compress(csv_bytes)))
    _, _, s_items = extract_archive(snappy_path, tmp_path / "out_snappy")
    assert len(s_items) == 1
    assert s_items[0].rel_path == "report.csv"
    assert s_items[0].disk_path.read_bytes() == csv_bytes

    # 2. Debian .deb (Unix `!<arch>\n` wrapping `data.tar.gz`)
    data_tgz = gzip.compress(_make_tar_bytes())
    deb_buf = bytearray(b"!<arch>\n")
    for member_name, member_bytes in [
        ("debian-binary", b"2.0\n"),
        ("data.tar.gz", data_tgz),
    ]:
        hdr = (
            f"{member_name:<16}"
            f"{'1700000000':<12}"
            f"{'0':<6}{'0':<6}"
            f"{'100644':<8}"
            f"{len(member_bytes):<10}`\n"
        ).encode("ascii")
        deb_buf.extend(hdr)
        deb_buf.extend(member_bytes)
        if len(member_bytes) % 2 == 1:
            deb_buf.extend(b"\n")

    deb_path = tmp_path / "pkg.deb"
    deb_path.write_bytes(bytes(deb_buf))
    fmt, _, deb_items = extract_archive(deb_path, tmp_path / "out_deb")
    assert fmt == "deb"
    assert any(i.rel_path == "project-v1/src/main.py" for i in deb_items)

    # 3. SVR4 `070701` newc CPIO archive
    def _cpio_entry(name: str, mode: int, payload: bytes, ino: int = 1) -> bytes:
        name_b = name.encode("utf-8") + b"\x00"
        hdr = (
            "070701"
            f"{ino:08x}{mode:08x}{0:08x}{0:08x}{1:08x}{1700000000:08x}"
            f"{len(payload):08x}{0:08x}{0:08x}{0:08x}{0:08x}"
            f"{len(name_b):08x}{0:08x}"
        ).encode("ascii")
        pad_n = b"\x00" * ((4 - ((110 + len(name_b)) % 4)) % 4)
        pad_d = b"\x00" * ((4 - (len(payload) % 4)) % 4)
        return hdr + name_b + pad_n + payload + pad_d

    cpio_bytes = (
        _cpio_entry("etc/app.conf", 0o100644, b"port=8080\n", ino=1)
        + _cpio_entry("TRAILER!!!", 0, b"", ino=0)
    )
    cpio_path = tmp_path / "initramfs.cpio"
    cpio_path.write_bytes(cpio_bytes)
    c_fmt, _, c_items = extract_archive(cpio_path, tmp_path / "out_cpio")
    assert c_fmt == "cpio"
    c_files = [i for i in c_items if not i.is_dir]
    assert c_files[0].rel_path == "etc/app.conf"
    assert c_files[0].disk_path.read_bytes() == b"port=8080\n"


def test_zip_slip_sanitization_and_manipulation_options(tmp_path: Path) -> None:
    assert sanitize_archive_path("../../../etc/passwd") == "etc/passwd"
    assert sanitize_archive_path("C:\\Windows\\System32\\..\\cmd.exe") == "Windows/cmd.exe"

    # Build a tar with collisions, duplicates, junk, hidden files, and single root
    tar_path = tmp_path / "messy.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        entries = {
            "wrapper/a/app.py": b"same_code",
            "wrapper/b/app.py": b"different_code",
            "wrapper/c/dup.py": b"same_code",
            "wrapper/logs/debug.log": b"noisy_log",
            "wrapper/.DS_Store": b"mac_junk",
            "wrapper/.git/HEAD": b"ref: refs/heads/main",
        }
        for path_str, payload in entries.items():
            ti = tarfile.TarInfo(name=path_str)
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))

    _, _, extracted = extract_archive(tar_path, tmp_path / "staged")

    opts = ConversionOptions(
        unwrap_single_root=True,
        exclude_junk_files=True,
        exclude_hidden_files=True,
        exclude_patterns=["*.log"],
        deduplicate_identical=True,
        flatten_paths=True,
        root_prefix="clean-bundle",
        rename_rules=[RenameRule(pattern=r"\.py$", replacement=".python")],
    )
    kept, filtered_count = apply_manipulation_pipeline(extracted, opts)
    names = [e.arcname for e in kept]
    # `a/app.py` and `b/app.py` both rename to `app.python`, flatten, collide -> `app.python` & `app_1.python`!
    # `c/dup.py` has identical SHA-256 to `a/app.py`, so deduplicated!
    # `.DS_Store`, `.git/HEAD`, `debug.log`, and empty dirs filtered out!
    assert names == ["clean-bundle/app.python", "clean-bundle/app_1.python"]
    assert filtered_count >= 4
