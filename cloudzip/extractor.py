"""Universal multi-format archive detector, inspector, and secure extractor.

Supports ZIP (including WinZip AES-256, ZipCrypto, and Zstd Method 93), TAR and all
compressed TAR variants (.tar.gz/.tgz, .tar.bz2/.tbz2, .tar.xz/.txz, .tar.zst/.tzst,
.tar.lz4, .tar.br, .tar.snappy, .tar.lzma), 7-Zip (.7z), RAR v4/v5 (.rar/.cbr), ISO/UDF,
CAB, DEB, AR, RPM, CPIO, XAR, WIM, LZH/LHA, ARJ, and standalone compressed streams
(.gz, .bz2, .xz, .lzma, .zst, .lz4, .br, .snappy), with Zip-Slip traversal protection
and decompression-bomb resource guards.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import brotli
import cramjam
import lz4.frame
import py7zr
import pyzipper
import rarfile
import zstandard as zstd

from cloudzip.models import ArchiveEntryInfo, ArchiveInspectionResponse, FormatDescriptor

ProgressCallback = Callable[[int, int, str], None]

SUPPORTED_INPUT_FORMATS: list[FormatDescriptor] = [
    FormatDescriptor(
        format_id="zip",
        extensions=[".zip", ".jar", ".war", ".ear", ".apk", ".ipa", ".whl", ".epub", ".cbz", ".xpi"],
        description="PKZIP / ZIP64 / JAR / APK / Wheel / EPUB (Store, Deflate, BZip2, LZMA, Zstd, AES-256, ZipCrypto)",
        password_decryption_supported=True,
    ),
    FormatDescriptor(
        format_id="7z",
        extensions=[".7z"],
        description="7-Zip archive (LZMA, LZMA2, BZip2, Deflate, PPMd, Zstd, AES-256)",
        password_decryption_supported=True,
    ),
    FormatDescriptor(
        format_id="rar",
        extensions=[".rar", ".cbr"],
        description="RAR v4 / RAR v5 archive (via libarchive / unar / 7z / rarfile)",
        password_decryption_supported=True,
    ),
    FormatDescriptor(
        format_id="tar",
        extensions=[".tar"],
        description="POSIX / GNU / PAX uncompressed tape archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.gz",
        extensions=[".tar.gz", ".tgz", ".taz"],
        description="Gzip-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.bz2",
        extensions=[".tar.bz2", ".tbz", ".tbz2"],
        description="BZip2-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.xz",
        extensions=[".tar.xz", ".txz"],
        description="XZ / LZMA2-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.zst",
        extensions=[".tar.zst", ".tzst"],
        description="Zstandard-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.lz4",
        extensions=[".tar.lz4", ".tlz4"],
        description="LZ4-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.br",
        extensions=[".tar.br", ".tbr"],
        description="Brotli-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="tar.lzma",
        extensions=[".tar.lzma", ".tlz"],
        description="LZMA-compressed TAR archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="deb",
        extensions=[".deb", ".udeb", ".ar", ".a"],
        description="Debian binary package & Unix ar archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="cpio",
        extensions=[".cpio"],
        description="POSIX / SVR4 CPIO archive",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="iso",
        extensions=[".iso", ".udf", ".img"],
        description="ISO-9660 / UDF optical disc image (via libarchive / 7z)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="cab",
        extensions=[".cab"],
        description="Microsoft Cabinet archive (via libarchive / 7z / unar)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="rpm",
        extensions=[".rpm"],
        description="RPM Package Manager archive (via libarchive / 7z)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="xar",
        extensions=[".xar", ".pkg"],
        description="eXtensible ARchive format (via libarchive / 7z / unar)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="wim",
        extensions=[".wim", ".swm", ".esd"],
        description="Windows Imaging Format (via libarchive / 7z)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="lzh",
        extensions=[".lzh", ".lha", ".arj"],
        description="Legacy LZH / LHA / ARJ archives (via libarchive / 7z / unar)",
        password_decryption_supported=False,
    ),
    FormatDescriptor(
        format_id="stream",
        extensions=[".gz", ".bz2", ".xz", ".lzma", ".zst", ".lz4", ".br", ".snappy", ".sz"],
        description="Single-file compressed streams (Gzip, BZip2, XZ, LZMA, Zstandard, LZ4, Brotli, Snappy)",
        password_decryption_supported=False,
    ),
]


class ArchiveError(Exception):
    """Base error raised when an input archive is invalid, unsupported, or encrypted without key."""


class ArchivePasswordError(ArchiveError):
    """Raised when an archive requires a password or the provided password was rejected."""


@dataclass(slots=True)
class ExtractedItem:
    """Extracted filesystem item with preserved metadata."""

    rel_path: str
    disk_path: Path
    is_dir: bool
    size_bytes: int
    mtime: datetime
    posix_mode: int


def sanitize_archive_path(raw_path: str) -> str:
    """Sanitize a member path against Zip-Slip (`..`), absolute paths, drive letters, and nulls."""
    cleaned = raw_path.replace("\x00", "").replace("\\", "/")
    # Strip Windows drive prefix such as C:/
    cleaned = re.sub(r"^[A-Za-z]:/+", "", cleaned)
    cleaned = cleaned.lstrip("/")
    segments: list[str] = []
    for seg in cleaned.split("/"):
        if seg in {"", "."}:
            continue
        if seg == "..":
            if segments:
                segments.pop()
            continue
        segments.append(seg)
    return "/".join(segments)


def _safe_join(root: Path, rel_posix: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / rel_posix).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ArchiveError(f"Blocked unsafe path traversal outside staging root: '{rel_posix}'")
    return candidate


def detect_archive_format(archive_path: Path, original_filename: str | None = None) -> str:
    """Detect archive format using magic bytes first, refined by compound filename suffixes."""
    name_lower = (original_filename or archive_path.name).lower()

    with open(archive_path, "rb") as fp:
        head = fp.read(4096)
        iso_sig = b""
        for iso_off in (0x8001, 0x8801, 0x9001):
            fp.seek(iso_off)
            probe = fp.read(5)
            if probe == b"CD001":
                iso_sig = probe
                break

    # Compound tar suffix hints
    is_tar_named = any(
        name_lower.endswith(sfx)
        for sfx in (
            ".tar.gz",
            ".tgz",
            ".taz",
            ".tar.bz2",
            ".tbz",
            ".tbz2",
            ".tar.xz",
            ".txz",
            ".tar.zst",
            ".tzst",
            ".tar.lz4",
            ".tlz4",
            ".tar.br",
            ".tbr",
            ".tar.lzma",
            ".tlz",
            ".tar.snappy",
        )
    )

    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "zip"
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if head.startswith((b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")):
        return "rar"
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz" if is_tar_named else "gz"
    if head.startswith(b"BZh"):
        return "tar.bz2" if is_tar_named else "bz2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "tar.xz" if is_tar_named else "xz"
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        return "tar.zst" if is_tar_named else "zst"
    if head.startswith(b"\x04\x22\x4d\x18"):
        return "tar.lz4" if is_tar_named else "lz4"
    if head.startswith(b"\xff\x06\x00\x00sNaPpY"):
        return "tar.snappy" if is_tar_named else "snappy"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    if head.startswith(b"!<arch>\n"):
        return "deb" if name_lower.endswith((".deb", ".udeb")) else "ar"
    if head.startswith(b"\xed\xab\xee\xdb"):
        return "rpm"
    if head.startswith(b"MSCF"):
        return "cab"
    if head.startswith(b"xar!"):
        return "xar"
    if head.startswith(b"MSWIM\x00\x00\x00"):
        return "wim"
    if head.startswith((b"070701", b"070702", b"070707", b"\xc7\x71", b"\x71\xc7")):
        return "cpio"
    if iso_sig == b"CD001":
        return "iso"

    # Check tarfile directly (handles old V7 tar without 'ustar' magic)
    try:
        if tarfile.is_tarfile(archive_path):
            return "tar"
    except Exception:
        pass

    # Extension-based fallback for streams lacking unambiguous magic (brotli, raw lzma, lzh, arj)
    ext_map = [
        ((".tar.br", ".tbr"), "tar.br"),
        ((".br",), "br"),
        ((".tar.lzma", ".tlz"), "tar.lzma"),
        ((".lzma",), "lzma"),
        ((".snappy", ".sz"), "snappy"),
        ((".zip", ".jar", ".war", ".apk", ".whl", ".epub", ".cbz", ".ipa"), "zip"),
        ((".7z",), "7z"),
        ((".rar", ".cbr"), "rar"),
        ((".iso", ".udf"), "iso"),
        ((".cab",), "cab"),
        ((".rpm",), "rpm"),
        ((".deb", ".udeb"), "deb"),
        ((".ar",), "ar"),
        ((".cpio",), "cpio"),
        ((".xar", ".pkg"), "xar"),
        ((".wim", ".swm", ".esd"), "wim"),
        ((".lzh", ".lha"), "lzh"),
        ((".arj",), "arj"),
    ]
    for suffixes, fmt in ext_map:
        if name_lower.endswith(suffixes):
            return fmt

    return "auto"


def _stem_without_stream_ext(filename: str) -> str:
    lower = filename.lower()
    for ext in (".gz", ".bz2", ".xz", ".lzma", ".zst", ".lz4", ".br", ".snappy", ".sz"):
        if lower.endswith(ext) and len(filename) > len(ext):
            return filename[: -len(ext)]
    if lower.endswith(".tgz"):
        return filename[:-4] + ".tar"
    if lower.endswith((".tbz", ".txz", ".tlz")):
        return filename[:-4] + ".tar"
    if lower.endswith((".tbz2", ".tzst", ".tlz4")):
        return filename[:-5] + ".tar"
    return f"{filename}.extracted"


def _decompress_stream_to_file(
    src_path: Path,
    dest_path: Path,
    codec: str,
    *,
    max_bytes: int,
) -> int:
    """Decompress a single-stream compressed file (`gz`, `bz2`, `xz`, `lzma`, `zst`, `lz4`, `br`, `snappy`)."""
    written = 0
    chunk_sz = 1024 * 1024

    def _copy_limited(reader: io.BufferedReader | object, out_fp: io.BufferedWriter) -> None:
        nonlocal written
        while True:
            block = reader.read(chunk_sz)  # type: ignore[attr-defined]
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise ArchiveError(
                    f"Decompressed stream exceeds maximum allowed size ({max_bytes} bytes)."
                )
            out_fp.write(block)

    with open(dest_path, "wb") as out_fp:
        if codec == "gz":
            with gzip.open(src_path, "rb") as gz_in:
                _copy_limited(gz_in, out_fp)
        elif codec == "bz2":
            with bz2.open(src_path, "rb") as bz_in:
                _copy_limited(bz_in, out_fp)
        elif codec in {"xz", "lzma"}:
            with lzma.open(src_path, "rb") as lz_in:
                _copy_limited(lz_in, out_fp)
        elif codec == "zst":
            dctx = zstd.ZstdDecompressor()
            with open(src_path, "rb") as raw_in, dctx.stream_reader(raw_in) as zs_in:
                _copy_limited(zs_in, out_fp)
        elif codec == "lz4":
            with lz4.frame.open(src_path, "rb") as lz4_in:
                _copy_limited(lz4_in, out_fp)
        elif codec == "br":
            dec = brotli.Decompressor()
            with open(src_path, "rb") as raw_in:
                while True:
                    chunk = raw_in.read(chunk_sz)
                    if not chunk:
                        break
                    piece = dec.process(chunk)
                    written += len(piece)
                    if written > max_bytes:
                        raise ArchiveError(
                            f"Decompressed Brotli stream exceeds limit ({max_bytes} bytes)."
                        )
                    out_fp.write(piece)
        elif codec == "snappy":
            raw_data = src_path.read_bytes()
            try:
                plain = bytes(cramjam.snappy.decompress(raw_data))
            except Exception:
                plain = bytes(cramjam.snappy.decompress_raw(raw_data))
            written = len(plain)
            if written > max_bytes:
                raise ArchiveError(f"Decompressed Snappy stream exceeds limit ({max_bytes} bytes).")
            out_fp.write(plain)
        else:
            raise ArchiveError(f"Unknown stream codec '{codec}'.")
    return written


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    original_filename: str | None = None,
    password: str | None = None,
    extract_nested: bool = False,
    max_nested_depth: int = 2,
    max_uncompressed_bytes: int = 8 * 1024**3,
    max_entries: int = 100_000,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, str, list[ExtractedItem]]:
    """Extract an archive of any supported format into `dest_dir` and return `(format_id, backend, items)`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fmt = detect_archive_format(archive_path, original_filename)

    extractor_used = _extract_once(
        archive_path=archive_path,
        dest_dir=dest_dir,
        detected_fmt=fmt,
        original_filename=original_filename or archive_path.name,
        password=password,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_entries=max_entries,
        on_progress=on_progress,
    )

    # Upgrade reported format if a single compressed stream turned out to wrap a TAR archive
    if fmt in {"gz", "bz2", "xz", "lzma", "zst", "lz4", "br", "snappy"} and extractor_used.endswith(
        "+tar"
    ):
        fmt = f"tar.{fmt}"

    if extract_nested:
        _unpack_nested_archives(
            dest_dir,
            max_depth=max_nested_depth,
            password=password,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_entries=max_entries,
        )

    items = _collect_extracted_items(
        dest_dir,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_entries=max_entries,
    )
    return fmt, extractor_used, items


def _extract_once(
    *,
    archive_path: Path,
    dest_dir: Path,
    detected_fmt: str,
    original_filename: str,
    password: str | None,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> str:
    stream_Map = {
        "tar.gz": "gz",
        "gz": "gz",
        "tar.bz2": "bz2",
        "bz2": "bz2",
        "tar.xz": "xz",
        "xz": "xz",
        "tar.lzma": "lzma",
        "lzma": "lzma",
        "tar.zst": "zst",
        "zst": "zst",
        "tar.lz4": "lz4",
        "lz4": "lz4",
        "tar.br": "br",
        "br": "br",
        "tar.snappy": "snappy",
        "snappy": "snappy",
    }

    if detected_fmt == "zip":
        _extract_zip(
            archive_path,
            dest_dir,
            password=password,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_entries=max_entries,
            on_progress=on_progress,
        )
        return "python-zipfile/pyzipper"

    if detected_fmt == "tar":
        _extract_tar(
            archive_path,
            dest_dir,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_entries=max_entries,
            on_progress=on_progress,
        )
        return "python-tarfile"

    if detected_fmt in stream_Map:
        codec = stream_Map[detected_fmt]
        with tempfile.TemporaryDirectory(prefix="cloudzip_stream_") as tmp_s:
            inner_name = sanitize_archive_path(_stem_without_stream_ext(original_filename)) or "payload"
            inner_name = Path(inner_name).name
            temp_out = Path(tmp_s) / inner_name
            _decompress_stream_to_file(
                archive_path,
                temp_out,
                codec,
                max_bytes=max_uncompressed_bytes,
            )
            if tarfile.is_tarfile(temp_out):
                _extract_tar(
                    temp_out,
                    dest_dir,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                    max_entries=max_entries,
                    on_progress=on_progress,
                )
                return f"stream-{codec}+tar"
            final_file = _safe_join(dest_dir, inner_name)
            final_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_out), str(final_file))
            if on_progress:
                on_progress(1, 1, inner_name)
            return f"stream-{codec}"

    if detected_fmt == "7z":
        try:
            _extract_7z_python(
                archive_path,
                dest_dir,
                password=password,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_entries=max_entries,
                on_progress=on_progress,
            )
            return "py7zr"
        except ArchivePasswordError:
            raise
        except Exception as py7z_err:
            backend = _try_system_or_libarchive_extract(
                archive_path, dest_dir, password=password
            )
            if backend:
                return backend
            raise ArchiveError(f"Failed to extract 7z archive: {py7z_err}") from py7z_err

    if detected_fmt in {"deb", "ar"}:
        try:
            _extract_ar_or_deb(
                archive_path,
                dest_dir,
                unwrap_deb_data=(detected_fmt == "deb"),
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_entries=max_entries,
                on_progress=on_progress,
            )
            return "python-ar/deb"
        except Exception:
            backend = _try_system_or_libarchive_extract(archive_path, dest_dir, password=password)
            if backend:
                return backend
            raise

    if detected_fmt == "cpio":
        try:
            _extract_cpio_newc(
                archive_path,
                dest_dir,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_entries=max_entries,
                on_progress=on_progress,
            )
            return "python-cpio"
        except Exception:
            backend = _try_system_or_libarchive_extract(archive_path, dest_dir, password=password)
            if backend:
                return backend
            raise

    if detected_fmt == "rar":
        backend = _try_rar_extract(archive_path, dest_dir, password=password)
        if backend:
            return backend
        raise ArchiveError(
            "Unable to extract RAR archive (verify password or ensure libarchive/unar/7z is installed)."
        )

    # Universal fallback chain: libarchive -> bsdtar -> 7z/7zz -> unar -> shutil
    backend = _try_system_or_libarchive_extract(archive_path, dest_dir, password=password)
    if backend:
        return backend

    raise ArchiveError(
        f"Unsupported or corrupted archive file '{original_filename}' (detected='{detected_fmt}')."
    )


def _extract_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    password: str | None,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> None:
    pwd_bytes = password.encode("utf-8") if password else None
    dctx = zstd.ZstdDecompressor()

    try:
        zf_probe = zipfile.ZipFile(archive_path, "r")
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Invalid or corrupted ZIP archive: {exc}") from exc

    with zf_probe:
        infos = zf_probe.infolist()
        if len(infos) > max_entries:
            raise ArchiveError(f"Archive entry count ({len(infos)}) exceeds limit ({max_entries}).")

        needs_aes = any(i.compress_type == 99 for i in infos)
        has_encrypted = any(bool(i.flag_bits & 0x1) for i in infos)
        if has_encrypted and not pwd_bytes:
            raise ArchivePasswordError("Source ZIP archive is password-protected; provide source_password.")

    opener = pyzipper.AESZipFile if (needs_aes or has_encrypted) else zipfile.ZipFile
    total_bytes = 0
    total_count = len(infos)

    try:
        with opener(archive_path, "r") as zf, open(archive_path, "rb") as raw_fp:
            if pwd_bytes:
                zf.setpassword(pwd_bytes)
            for idx, info in enumerate(zf.infolist(), start=1):
                rel = sanitize_archive_path(info.filename)
                if not rel:
                    continue
                target = _safe_join(dest_dir, rel)

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    total_bytes += info.file_size
                    if total_bytes > max_uncompressed_bytes:
                        raise ArchiveError(
                            f"Uncompressed archive size exceeds limit ({max_uncompressed_bytes} bytes)."
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if info.compress_type == 93 and not (info.flag_bits & 0x1):
                        # Zstandard Method 93 ZIP entry
                        raw_fp.seek(info.header_offset)
                        lh = raw_fp.read(30)
                        _, _, _, _, _, _, _, _, _, fn_len, ex_len = struct.unpack("<4sHHHHHIIIHH", lh)
                        raw_fp.seek(info.header_offset + 30 + fn_len + ex_len)
                        comp_data = raw_fp.read(info.compress_size)
                        with dctx.stream_reader(io.BytesIO(comp_data)) as reader, open(
                            target, "wb"
                        ) as out_f:
                            shutil.copyfileobj(reader, out_f)
                    else:
                        with zf.open(info) as src_f, open(target, "wb") as out_f:
                            shutil.copyfileobj(src_f, out_f, length=1024 * 1024)

                # Restore timestamp and POSIX permissions
                try:
                    dt = datetime(*info.date_time, tzinfo=timezone.utc)
                    ts = dt.timestamp()
                    os.utime(target, (ts, ts))
                except Exception:
                    pass
                unix_mode = (info.external_attr >> 16) & 0o777
                if unix_mode:
                    try:
                        os.chmod(target, unix_mode | (0o700 if info.is_dir() else 0o600))
                    except Exception:
                        pass

                if on_progress:
                    on_progress(idx, total_count, rel)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "password" in msg or "bad password" in msg or "encrypted" in msg:
            raise ArchivePasswordError(f"Incorrect or missing password for ZIP archive: {exc}") from exc
        raise ArchiveError(f"Failed extracting ZIP archive: {exc}") from exc


def _extract_tar(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> None:
    total_bytes = 0
    try:
        with tarfile.open(archive_path, mode="r:*") as tf:
            members = tf.getmembers()
            if len(members) > max_entries:
                raise ArchiveError(f"TAR entry count ({len(members)}) exceeds limit ({max_entries}).")
            total = len(members)
            for idx, member in enumerate(members, start=1):
                rel = sanitize_archive_path(member.name)
                if not rel:
                    continue
                target = _safe_join(dest_dir, rel)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    total_bytes += max(0, member.size)
                    if total_bytes > max_uncompressed_bytes:
                        raise ArchiveError(
                            f"TAR uncompressed size exceeds limit ({max_uncompressed_bytes} bytes)."
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted_fp = tf.extractfile(member)
                    if extracted_fp is not None:
                        with extracted_fp, open(target, "wb") as out_f:
                            shutil.copyfileobj(extracted_fp, out_f, length=1024 * 1024)
                else:
                    # Skip special devices, FIFOs, and symlinks pointing outside sandbox
                    continue

                if member.mtime:
                    try:
                        os.utime(target, (member.mtime, member.mtime))
                    except Exception:
                        pass
                if member.mode:
                    try:
                        os.chmod(
                            target,
                            (member.mode & 0o777) | (0o700 if member.isdir() else 0o600),
                        )
                    except Exception:
                        pass

                if on_progress:
                    on_progress(idx, total, rel)
    except tarfile.TarError as exc:
        raise ArchiveError(f"Failed reading TAR archive: {exc}") from exc


def _extract_7z_python(
    archive_path: Path,
    dest_dir: Path,
    *,
    password: str | None,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> None:
    try:
        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as sz:
            if sz.needs_password() and not password:
                raise ArchivePasswordError("7z archive is encrypted; provide source_password.")
            infos = sz.list()
            if len(infos) > max_entries:
                raise ArchiveError(f"7z entry count ({len(infos)}) exceeds limit ({max_entries}).")
            total_uncomp = sum(max(0, info.uncompressed or 0) for info in infos)
            if total_uncomp > max_uncompressed_bytes:
                raise ArchiveError(
                    f"7z uncompressed size ({total_uncomp}) exceeds limit ({max_uncompressed_bytes})."
                )
            with tempfile.TemporaryDirectory(prefix="cloudzip_7z_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                sz.extractall(path=tmp_path)
                all_paths = list(tmp_path.rglob("*"))
                total = max(1, len(all_paths))
                for idx, item in enumerate(all_paths, start=1):
                    if item.is_symlink():
                        continue
                    rel = sanitize_archive_path(str(item.relative_to(tmp_path)))
                    if not rel:
                        continue
                    target = _safe_join(dest_dir, rel)
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif item.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(item), str(target))
                    if on_progress:
                        on_progress(idx, total, rel)
    except py7zr.PasswordRequired as exc:
        raise ArchivePasswordError("7z archive requires a password; provide source_password.") from exc
    except ArchivePasswordError:
        raise
    except Exception as exc:
        if password and ("password" in str(exc).lower() or "corrupt" in str(exc).lower() or "lzma" in str(exc).lower()):
            raise ArchivePasswordError(f"Invalid password or encrypted 7z stream error: {exc}") from exc
        raise


def _extract_ar_or_deb(
    archive_path: Path,
    dest_dir: Path,
    *,
    unwrap_deb_data: bool,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> None:
    """Pure-Python Unix `ar` / Debian `.deb` archive reader."""
    with open(archive_path, "rb") as fp:
        magic = fp.read(8)
        if magic != b"!<arch>\n":
            raise ArchiveError("Invalid Unix ar/deb magic header.")

        gnu_Name_table: bytes = b""
        count = 0
        total_bytes = 0
        staged_members: list[tuple[str, int, int, bytes]] = []

        while True:
            hdr = fp.read(60)
            if not hdr:
                break
            if len(hdr) < 60 or hdr[58:60] != b"`\n":
                raise ArchiveError("Malformed Unix ar member header.")
            raw_name = hdr[0:16].decode("latin-1", errors="replace").strip()
            mtime_str = hdr[16:28].decode("ascii", errors="ignore").strip()
            mode_str = hdr[40:48].decode("ascii", errors="ignore").strip()
            size_str = hdr[48:58].decode("ascii", errors="ignore").strip()
            size = int(size_str) if size_str else 0
            mtime = int(mtime_str) if mtime_str.isdigit() else int(datetime.now(timezone.utc).timestamp())
            mode = int(mode_str, 8) if mode_str and all(c in "01234567" for c in mode_str) else 0o644

            data = fp.read(size)
            if size % 2 == 1:
                fp.read(1)  # 2-byte alignment padding

            if raw_name == "//":
                gnu_Name_table = data
                continue
            if raw_name in {"/", "/SYM64/"}:
                continue

            if raw_name.startswith("#1/"):
                ext_len = int(raw_name[3:])
                real_name = data[:ext_len].rstrip(b"\x00").decode("utf-8", errors="replace")
                data = data[ext_len:]
            elif raw_name.startswith("/") and raw_name[1:].isdigit() and gnu_Name_table:
                off = int(raw_name[1:])
                end = gnu_Name_table.find(b"/\n", off)
                if end == -1:
                    end = gnu_Name_table.find(b"\n", off)
                real_name = gnu_Name_table[off:end].decode("utf-8", errors="replace")
            else:
                real_name = raw_name.rstrip("/")

            count += 1
            if count > max_entries:
                raise ArchiveError(f"AR archive entry count exceeds {max_entries}.")
            total_bytes += len(data)
            if total_bytes > max_uncompressed_bytes:
                raise ArchiveError("AR archive exceeds uncompressed size limit.")
            staged_members.append((real_name, mtime, mode, data))

    # If .deb package contains data.tar.*, extract the package filesystem cleanly
    if unwrap_deb_data:
        for m_name, _, _, m_data in staged_members:
            if m_name.startswith("data.tar"):
                with tempfile.NamedTemporaryFile(suffix=f"_{m_name}", delete=False) as tf_tmp:
                    tf_tmp.write(m_data)
                    tmp_p = Path(tf_tmp.name)
                try:
                    sub_fmt = detect_archive_format(tmp_p, m_name)
                    _extract_once(
                        archive_path=tmp_p,
                        dest_dir=dest_dir,
                        detected_fmt=sub_fmt,
                        original_filename=m_name,
                        password=None,
                        max_uncompressed_bytes=max_uncompressed_bytes,
                        max_entries=max_entries,
                        on_progress=on_progress,
                    )
                    return
                finally:
                    tmp_p.unlink(missing_ok=True)

    total = max(1, len(staged_members))
    for idx, (m_name, mtime, mode, m_data) in enumerate(staged_members, start=1):
        rel = sanitize_archive_path(m_name)
        if not rel:
            continue
        out_p = _safe_join(dest_dir, rel)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(m_data)
        try:
            os.utime(out_p, (mtime, mtime))
            os.chmod(out_p, (mode & 0o777) | 0o600)
        except Exception:
            pass
        if on_progress:
            on_progress(idx, total, rel)


def _extract_cpio_newc(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_uncompressed_bytes: int,
    max_entries: int,
    on_progress: ProgressCallback | None,
) -> None:
    """Pure-Python SVR4 `070701` / `070702` ASCII CPIO reader."""
    with open(archive_path, "rb") as fp:
        count = 0
        total_bytes = 0
        while True:
            hdr = fp.read(110)
            if not hdr:
                break
            if len(hdr) < 110 or hdr[:6] not in {b"070701", b"070702"}:
                raise ArchiveError("Not an SVR4 newc/crc CPIO stream.")
            mode = int(hdr[14:22], 16)
            mtime = int(hdr[46:54], 16)
            filesize = int(hdr[54:62], 16)
            namesize = int(hdr[94:102], 16)

            name_bytes = fp.read(namesize)
            # Align (110 + namesize) to 4-byte boundary
            pad_name = (4 - ((110 + namesize) % 4)) % 4
            if pad_name:
                fp.read(pad_name)

            raw_name = name_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
            if raw_name == "TRAILER!!!":
                break

            count += 1
            if count > max_entries:
                raise ArchiveError(f"CPIO entry count exceeds limit ({max_entries}).")

            total_bytes += filesize
            if total_bytes > max_uncompressed_bytes:
                raise ArchiveError("CPIO uncompressed size exceeds limit.")

            file_data = fp.read(filesize)
            pad_data = (4 - (filesize % 4)) % 4
            if pad_data:
                fp.read(pad_data)

            rel = sanitize_archive_path(raw_name)
            if not rel:
                continue
            target = _safe_join(dest_dir, rel)
            if stat.S_ISDIR(mode):
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(file_data)
            else:
                continue
            try:
                os.utime(target, (mtime, mtime))
                os.chmod(target, (mode & 0o777) | (0o700 if stat.S_ISDIR(mode) else 0o600))
            except Exception:
                pass
            if on_progress:
                on_progress(count, count, rel)


def _try_rar_extract(archive_path: Path, dest_dir: Path, *, password: str | None) -> str | None:
    backend = _try_system_or_libarchive_extract(archive_path, dest_dir, password=password)
    if backend:
        return backend
    try:
        with rarfile.RarFile(archive_path) as rf:
            if rf.needs_password() and not password:
                raise ArchivePasswordError("RAR archive is encrypted; provide source_password.")
            if password:
                rf.setpassword(password)
            for info in rf.infolist():
                rel = sanitize_archive_path(info.filename)
                if not rel:
                    continue
                target = _safe_join(dest_dir, rel)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with rf.open(info) as src_f, open(target, "wb") as out_f:
                        shutil.copyfileobj(src_f, out_f)
        return "python-rarfile"
    except ArchivePasswordError:
        raise
    except Exception:
        return None


def _try_system_or_libarchive_extract(
    archive_path: Path,
    dest_dir: Path,
    *,
    password: str | None,
) -> str | None:
    """Attempt extraction via libarchive-c, bsdtar, 7z/7zz, or unar."""
    # 1. libarchive-c native shared library
    try:
        import libarchive  # type: ignore[import-untyped]

        with tempfile.TemporaryDirectory(prefix="cloudzip_la_") as tmp_dir:
            tmp_p = Path(tmp_dir)
            kwargs = {"passphrase": password} if password else {}
            with libarchive.file_reader(str(archive_path), **kwargs) as archive:
                for entry in archive:
                    pathname = getattr(entry, "pathname", None) or ""
                    rel = sanitize_archive_path(pathname)
                    if not rel:
                        continue
                    out_p = _safe_join(tmp_p, rel)
                    if entry.isdir:
                        out_p.mkdir(parents=True, exist_ok=True)
                    elif entry.isfile:
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_p, "wb") as f_out:
                            for block in entry.get_blocks():
                                f_out.write(block)
                    if entry.mtime:
                        try:
                            os.utime(out_p, (entry.mtime, entry.mtime))
                        except Exception:
                            pass
            _move_safe_tree(tmp_p, dest_dir)
            return "libarchive-c"
    except Exception as exc:
        if "passphrase" in str(exc).lower() or "password" in str(exc).lower():
            raise ArchivePasswordError(f"Archive requires valid password: {exc}") from exc

    # 2. 7z / 7zz CLI (supports 7z, rar, iso, cab, wim, rpm, deb, xar, lzh, arj, dmg, zstd...)
    for seven_bin in ("7zz", "7z", "7za"):
        bin_path = shutil.which(seven_bin)
        if not bin_path:
            continue
        with tempfile.TemporaryDirectory(prefix="cloudzip_7zcli_") as tmp_dir:
            pw_arg = f"-p{password}" if password else "-p-"
            cmd = [bin_path, "x", "-y", pw_arg, f"-o{tmp_dir}", str(archive_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                _move_safe_tree(Path(tmp_dir), dest_dir)
                return f"{seven_bin}-cli"
            combined = (proc.stdout + proc.stderr).lower()
            if "wrong password" in combined or "encrypted" in combined:
                raise ArchivePasswordError("Archive is password-protected or password was rejected.")

    # 3. unar (The Unarchiver CLI — handles RAR5, StuffIt, ALZip, LZH, ARJ, CAB, XAR, ISO)
    unar_bin = shutil.which("unar")
    if unar_bin:
        with tempfile.TemporaryDirectory(prefix="cloudzip_unar_") as tmp_dir:
            cmd = [unar_bin, "-f", "-D", "-o", tmp_dir]
            if password:
                cmd.extend(["-p", password])
            cmd.append(str(archive_path))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                _move_safe_tree(Path(tmp_dir), dest_dir)
                return "unar-cli"
            if "password" in (proc.stdout + proc.stderr).lower():
                raise ArchivePasswordError("Archive requires valid password.")

    # 4. bsdtar CLI
    bsdtar_bin = shutil.which("bsdtar")
    if bsdtar_bin and not password:
        with tempfile.TemporaryDirectory(prefix="cloudzip_bsdtar_") as tmp_dir:
            cmd = [bsdtar_bin, "-xf", str(archive_path), "-C", tmp_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                _move_safe_tree(Path(tmp_dir), dest_dir)
                return "bsdtar-cli"

    return None


def _move_safe_tree(src_root: Path, dest_root: Path) -> None:
    for item in sorted(src_root.rglob("*")):
        if item.is_symlink():
            continue
        rel = sanitize_archive_path(str(item.relative_to(src_root)))
        if not rel:
            continue
        target = _safe_join(dest_root, rel)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))


def _unpack_nested_archives(
    root_dir: Path,
    *,
    max_depth: int,
    password: str | None,
    max_uncompressed_bytes: int,
    max_entries: int,
) -> None:
    nested_exts = (
        ".zip",
        ".7z",
        ".rar",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
        ".tar.zst",
        ".tzst",
        ".tar.lz4",
        ".cpio",
    )
    for _depth in range(max_depth):
        candidates = [
            p
            for p in list(root_dir.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name.lower().endswith(nested_exts)
        ]
        if not candidates:
            break
        unpacked_any = False
        for nested_file in candidates:
            stem = nested_file.name
            for ext in sorted(nested_exts, key=len, reverse=True):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)] or "nested"
                    break
            nested_dest = nested_file.parent / f"{stem}_extracted"
            try:
                sub_fmt = detect_archive_format(nested_file, nested_file.name)
                _extract_once(
                    archive_path=nested_file,
                    dest_dir=nested_dest,
                    detected_fmt=sub_fmt,
                    original_filename=nested_file.name,
                    password=password,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                    max_entries=max_entries,
                    on_progress=None,
                )
                nested_file.unlink()
                unpacked_any = True
            except Exception:
                shutil.rmtree(nested_dest, ignore_errors=True)
        if not unpacked_any:
            break


def _collect_extracted_items(
    root_dir: Path,
    *,
    max_uncompressed_bytes: int,
    max_entries: int,
) -> list[ExtractedItem]:
    items: list[ExtractedItem] = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        filenames.sort()
        cur_dir = Path(dirpath)
        if cur_dir != root_dir:
            rel_dir = sanitize_archive_path(str(cur_dir.relative_to(root_dir)))
            if rel_dir:
                st = cur_dir.stat()
                items.append(
                    ExtractedItem(
                        rel_path=rel_dir,
                        disk_path=cur_dir,
                        is_dir=True,
                        size_bytes=0,
                        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                        posix_mode=stat.S_IMODE(st.st_mode) or 0o755,
                    )
                )
        for fn in filenames:
            fpath = cur_dir / fn
            if fpath.is_symlink() or not fpath.is_file():
                continue
            rel_file = sanitize_archive_path(str(fpath.relative_to(root_dir)))
            if not rel_file:
                continue
            st = fpath.stat()
            total_bytes += st.st_size
            if total_bytes > max_uncompressed_bytes:
                raise ArchiveError(
                    f"Extracted payload ({total_bytes} bytes) exceeds limit ({max_uncompressed_bytes} bytes)."
                )
            items.append(
                ExtractedItem(
                    rel_path=rel_file,
                    disk_path=fpath,
                    is_dir=False,
                    size_bytes=st.st_size,
                    mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                    posix_mode=stat.S_IMODE(st.st_mode) or 0o644,
                )
            )
            if len(items) > max_entries:
                raise ArchiveError(f"Extracted entry count exceeds limit ({max_entries}).")

    return items


def inspect_archive(
    archive_path: Path,
    *,
    original_filename: str | None = None,
    password: str | None = None,
    max_uncompressed_bytes: int = 8 * 1024**3,
    max_entries: int = 100_000,
) -> ArchiveInspectionResponse:
    """Inspect any uploaded archive format and return its manifest and summary metadata."""
    fname = original_filename or archive_path.name
    archive_bytes = archive_path.stat().st_size
    detected_fmt = detect_archive_format(archive_path, fname)

    # Fast non-extracting inspection path for standard ZIP archives
    if detected_fmt == "zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                infos = zf.infolist()
                entries: list[ArchiveEntryInfo] = []
                has_enc = False
                roots: set[str] = set()
                file_cnt = 0
                dir_cnt = 0
                uncomp_sum = 0
                for info in infos:
                    rel = sanitize_archive_path(info.filename)
                    if not rel:
                        continue
                    is_d = info.is_dir()
                    enc = bool(info.flag_bits & 0x1)
                    has_enc = has_enc or enc
                    roots.add(rel.split("/")[0])
                    if is_d:
                        dir_cnt += 1
                    else:
                        file_cnt += 1
                        uncomp_sum += info.file_size
                    try:
                        mod_iso = datetime(*info.date_time, tzinfo=timezone.utc).isoformat()
                    except Exception:
                        mod_iso = None
                    mode_bits = (info.external_attr >> 16) & 0o777
                    entries.append(
                        ArchiveEntryInfo(
                            path=rel + ("/" if is_d else ""),
                            is_dir=is_d,
                            size_bytes=0 if is_d else info.file_size,
                            compressed_bytes=0 if is_d else info.compress_size,
                            modified_iso=mod_iso,
                            mode_octal=oct(mode_bits) if mode_bits else None,
                            encrypted=enc,
                        )
                    )
                single_root = next(iter(roots)) if len(roots) == 1 and dir_cnt + file_cnt > 1 else None
                return ArchiveInspectionResponse(
                    filename=fname,
                    detected_format="zip",
                    extractor_used="python-zipfile-header-scan",
                    archive_size_bytes=archive_bytes,
                    total_entries=len(entries),
                    file_count=file_cnt,
                    dir_count=dir_cnt,
                    total_uncompressed_bytes=uncomp_sum,
                    has_encrypted_entries=has_enc,
                    single_top_level_root=single_root,
                    entries=entries,
                )
        except zipfile.BadZipFile as exc:
            raise ArchiveError(f"Corrupted ZIP archive: {exc}") from exc

    # General inspection via temporary staging extraction (supports every format uniformly)
    with tempfile.TemporaryDirectory(prefix="cloudzip_inspect_") as tmp_dir:
        try:
            fmt, backend, items = extract_archive(
                archive_path,
                Path(tmp_dir),
                original_filename=fname,
                password=password,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_entries=max_entries,
            )
        except ArchivePasswordError:
            return ArchiveInspectionResponse(
                filename=fname,
                detected_format=detected_fmt,
                extractor_used="encrypted-header-detect",
                archive_size_bytes=archive_bytes,
                total_entries=0,
                file_count=0,
                dir_count=0,
                total_uncompressed_bytes=0,
                has_encrypted_entries=True,
                single_top_level_root=None,
                entries=[],
            )

        entries = []
        roots = set()
        file_cnt = 0
        dir_cnt = 0
        uncomp_sum = 0
        for item in items:
            roots.add(item.rel_path.split("/")[0])
            if item.is_dir:
                dir_cnt += 1
            else:
                file_cnt += 1
                uncomp_sum += item.size_bytes
            entries.append(
                ArchiveEntryInfo(
                    path=item.rel_path + ("/" if item.is_dir else ""),
                    is_dir=item.is_dir,
                    size_bytes=item.size_bytes,
                    compressed_bytes=None,
                    modified_iso=item.mtime.isoformat(),
                    mode_octal=oct(item.posix_mode),
                    encrypted=bool(password),
                )
            )
        single_root = next(iter(roots)) if len(roots) == 1 and len(entries) > 1 else None
        return ArchiveInspectionResponse(
            filename=fname,
            detected_format=fmt,
            extractor_used=backend,
            archive_size_bytes=archive_bytes,
            total_entries=len(entries),
            file_count=file_cnt,
            dir_count=dir_cnt,
            total_uncompressed_bytes=uncomp_sum,
            has_encrypted_entries=bool(password),
            single_top_level_root=single_root,
            entries=entries,
        )
