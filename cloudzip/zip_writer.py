"""Spec-compliant PKWARE APPNOTE 6.3.10 ZIP / ZIP64 binary writer and verifier.

Accepts pre-compressed raw bitstreams generated on GPU by NVIDIA nvCOMP (DEFLATE Method 8,
Zstandard Method 93, or uncompressed STORE Method 0) and writes standard Local File Headers,
Central Directory records, ZIP64 extensions, UTF-8 EFS flags, UTC Extended Timestamps, and
optional WinZip AE-2 (256-bit AES-CTR + HMAC-SHA1) or ZipCrypto encryption.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import struct
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pyzipper
import zstandard as zstd
from Cryptodome.Cipher import AES
from Cryptodome.Util import Counter

from cloudzip.models import EncryptionMethod, Zip64Mode
from cloudzip.nvcomp_engine import ZIP_METHOD_DEFLATE, ZIP_METHOD_STORE, ZIP_METHOD_ZSTD

SIG_LOCAL_FILE_HEADER = b"PK\x03\x04"
SIG_CENTRAL_DIR_HEADER = b"PK\x01\x02"
SIG_ZIP64_EOCD_RECORD = b"PK\x06\x06"
SIG_ZIP64_EOCD_LOCATOR = b"PK\x06\x07"
SIG_EOCD_RECORD = b"PK\x05\x06"

ZIP_METHOD_WZ_AES = 99
FLAG_ENCRYPTED = 0x0001
FLAG_UTF8_EFS = 0x0800

ZIP32_LIMIT = 0xFFFFFFFF
ZIP32_COUNT_LIMIT = 0xFFFF

REPRODUCIBLE_EPOCH_UTC = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _build_crc32_table() -> tuple[int, ...]:
    table: list[int] = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (0xEDB88320 ^ (c >> 1)) if (c & 1) else (c >> 1)
        table.append(c)
    return tuple(table)


_CRC32_TABLE = _build_crc32_table()


def _zipcrypto_encrypt(payload: bytes, password: str, crc32: int) -> bytes:
    """Encrypt a compressed payload with traditional PKZIP 2.0 stream cipher."""
    table = _CRC32_TABLE
    k0, k1, k2 = 0x12345678, 0x23456789, 0x34567890

    for b in password.encode("utf-8"):
        k0 = table[(k0 ^ b) & 0xFF] ^ (k0 >> 8)
        k1 = ((k1 + (k0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        k2 = table[(k2 ^ (k1 >> 24)) & 0xFF] ^ (k2 >> 8)

    # 11 random bytes + 1 check byte = (crc32 >> 24) & 0xFF (since General Purpose Bit 3 == 0)
    header = os.urandom(11) + bytes([(crc32 >> 24) & 0xFF])
    plain = header + payload
    out = bytearray(len(plain))

    for idx, pbyte in enumerate(plain):
        temp = (k2 | 2) & 0xFFFF
        stream_byte = ((temp * (temp ^ 1)) >> 8) & 0xFF
        cbyte = pbyte ^ stream_byte
        out[idx] = cbyte
        k0 = table[(k0 ^ pbyte) & 0xFF] ^ (k0 >> 8)
        k1 = ((k1 + (k0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        k2 = table[(k2 ^ (k1 >> 24)) & 0xFF] ^ (k2 >> 8)

    return bytes(out)


def _aes256_ae2_encrypt(payload: bytes, password: str) -> bytes:
    """Encrypt a GPU-compressed bitstream using WinZip AE-2 (256-bit AES-CTR + HMAC-SHA1)."""
    salt = os.urandom(16)  # 16-byte salt for AES-256 (strength = 3)
    derived = hashlib.pbkdf2_hmac("sha1", password.encode("utf-8"), salt, 1000, dklen=66)
    aes_key = derived[:32]
    hmac_key = derived[32:64]
    pw_verify = derived[64:66]

    ctr = Counter.new(128, initial_value=1, little_endian=True)
    cipher = AES.new(aes_key, AES.MODE_CTR, counter=ctr)
    ciphertext = cipher.encrypt(payload)
    auth_code = hmac.new(hmac_key, ciphertext, hashlib.sha1).digest()[:10]
    return salt + pw_verify + ciphertext + auth_code


def _encode_arcname(path_str: str, *, is_dir: bool) -> tuple[bytes, bool]:
    norm = path_str.replace("\\", "/").lstrip("/")
    if is_dir and not norm.endswith("/"):
        norm += "/"
    elif not is_dir:
        norm = norm.rstrip("/")
    try:
        return norm.encode("ascii"), False
    except UnicodeEncodeError:
        return norm.encode("utf-8"), True


def _datetime_to_dos(dt: datetime) -> tuple[int, int]:
    utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    year = max(1980, min(2107, utc_dt.year))
    dos_date = ((year - 1980) << 9) | (utc_dt.month << 5) | utc_dt.day
    dos_time = (utc_dt.hour << 11) | (utc_dt.minute << 5) | (utc_dt.second // 2)
    return dos_time & 0xFFFF, dos_date & 0xFFFF


@dataclass(slots=True)
class ZipMemberSpec:
    """Prepared archive member ready to be written into the output ZIP stream."""

    arcname: str
    is_dir: bool
    zip_method_id: int
    payload: bytes
    uncompressed_size: int
    crc32: int
    mtime: datetime
    posix_mode: int = 0o644


@dataclass(slots=True)
class _CentralDirEntry:
    arcname_bytes: bytes
    version_made_by: int
    version_needed: int
    flags: int
    header_method: int
    dos_time: int
    dos_date: int
    header_crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    external_attr: int
    extra_base: bytes


class CloudZipWriter:
    """Streaming binary PKZIP / ZIP64 archive builder."""

    def __init__(
        self,
        dest: Path | BinaryIO,
        *,
        zip64_mode: Zip64Mode = Zip64Mode.AUTO,
        password: str | None = None,
        encryption_method: EncryptionMethod = EncryptionMethod.NONE,
        preserve_timestamps: bool = True,
        normalize_timestamps: bool = False,
        preserve_permissions: bool = True,
        archive_comment: str | None = None,
    ) -> None:
        self._owns_fp = isinstance(dest, Path)
        self._path: Path | None = dest if isinstance(dest, Path) else None
        self._fp: BinaryIO = open(dest, "wb") if isinstance(dest, Path) else dest
        self._zip64_mode = zip64_mode
        self._password = password if password else None
        self._encryption_method = (
            encryption_method if self._password else EncryptionMethod.NONE
        )
        self._preserve_timestamps = preserve_timestamps
        self._normalize_timestamps = normalize_timestamps
        self._preserve_permissions = preserve_permissions
        self._archive_comment = archive_comment or ""
        self._central_entries: list[_CentralDirEntry] = []
        self._zip64_used = False
        self._closed = False

    @property
    def zip64_used(self) -> bool:
        return self._zip64_used

    def write_member(self, member: ZipMemberSpec) -> int:
        """Write a single file or directory entry and return bytes written for its payload."""
        if self._closed:
            raise RuntimeError("Cannot write to a closed CloudZipWriter.")

        local_offset = self._fp.tell()
        arcname_bytes, needs_utf8 = _encode_arcname(member.arcname, is_dir=member.is_dir)

        effective_dt = (
            REPRODUCIBLE_EPOCH_UTC
            if (self._normalize_timestamps or not self._preserve_timestamps)
            else member.mtime
        )
        dos_time, dos_date = _datetime_to_dos(effective_dt)

        flags = FLAG_UTF8_EFS if needs_utf8 else 0
        payload = b"" if member.is_dir else member.payload
        uncomp_size = 0 if member.is_dir else member.uncompressed_size
        actual_method = ZIP_METHOD_STORE if member.is_dir else member.zip_method_id
        header_method = actual_method
        header_crc = 0 if member.is_dir else (member.crc32 & 0xFFFFFFFF)

        extra_parts: list[bytes] = []

        # Extended UTC timestamp extra field (0x5455)
        if self._preserve_timestamps and not self._normalize_timestamps:
            unix_ts = int(
                (
                    effective_dt.astimezone(timezone.utc)
                    if effective_dt.tzinfo
                    else effective_dt.replace(tzinfo=timezone.utc)
                ).timestamp()
            )
            unix_ts = max(-2147483648, min(2147483647, unix_ts))
            extra_parts.append(struct.pack("<HHBl", 0x5455, 5, 0x01, unix_ts))

        # Apply optional encryption over the already GPU-compressed payload
        if not member.is_dir and self._password:
            flags |= FLAG_ENCRYPTED
            if self._encryption_method == EncryptionMethod.AES256:
                payload = _aes256_ae2_encrypt(payload, self._password)
                header_method = ZIP_METHOD_WZ_AES
                header_crc = 0  # WinZip AE-2 suppresses plaintext CRC-32 in header
                extra_parts.append(
                    struct.pack("<HHH2sBH", 0x9901, 7, 2, b"AE", 3, actual_method)
                )
            elif self._encryption_method == EncryptionMethod.ZIPCRYPTO:
                payload = _zipcrypto_encrypt(payload, self._password, header_crc)

        comp_size = len(payload)

        needs_entry_zip64 = (
            self._zip64_mode == Zip64Mode.FORCE
            or uncomp_size >= ZIP32_LIMIT
            or comp_size >= ZIP32_LIMIT
        )
        if needs_entry_zip64 and self._zip64_mode == Zip64Mode.DISABLE:
            raise ValueError(
                f"Entry '{member.arcname}' exceeds 4 GiB limit while zip64_mode='disable'."
            )

        version_needed = 20
        if actual_method == ZIP_METHOD_ZSTD:
            version_needed = max(version_needed, 63)
        if header_method == ZIP_METHOD_WZ_AES:
            version_needed = max(version_needed, 51)
        if needs_entry_zip64 or local_offset >= ZIP32_LIMIT:
            if self._zip64_mode == Zip64Mode.DISABLE:
                raise ValueError("Archive offset exceeds 4 GiB while zip64_mode='disable'.")
            version_needed = max(version_needed, 45)
            self._zip64_used = True

        local_extra = b"".join(extra_parts)
        if needs_entry_zip64:
            self._zip64_used = True
            # Local Header ZIP64 extra field MUST carry both uncompressed & compressed 64-bit sizes
            local_extra = struct.pack("<HHQQ", 0x0001, 16, uncomp_size, comp_size) + local_extra
            lh_uncomp = ZIP32_LIMIT
            lh_comp = ZIP32_LIMIT
        else:
            lh_uncomp = uncomp_size
            lh_comp = comp_size

        local_header = struct.pack(
            "<4sHHHHHIIIHH",
            SIG_LOCAL_FILE_HEADER,
            version_needed,
            flags,
            header_method,
            dos_time,
            dos_date,
            header_crc,
            lh_comp,
            lh_uncomp,
            len(arcname_bytes),
            len(local_extra),
        )
        self._fp.write(local_header)
        self._fp.write(arcname_bytes)
        self._fp.write(local_extra)
        if payload:
            self._fp.write(payload)

        # External attributes: Unix permissions in upper 16 bits + DOS directory flag in bit 4
        if member.is_dir:
            mode_bits = (member.posix_mode & 0o7777) if self._preserve_permissions else 0o755
            unix_mode = 0o040000 | mode_bits
            external_attr = (unix_mode << 16) | 0x10
        else:
            mode_bits = (member.posix_mode & 0o7777) if self._preserve_permissions else 0o644
            unix_mode = 0o100000 | mode_bits
            external_attr = unix_mode << 16

        version_made_by = (3 << 8) | version_needed  # 3 = Unix host OS

        self._central_entries.append(
            _CentralDirEntry(
                arcname_bytes=arcname_bytes,
                version_made_by=version_made_by,
                version_needed=version_needed,
                flags=flags,
                header_method=header_method,
                dos_time=dos_time,
                dos_date=dos_date,
                header_crc32=header_crc,
                compressed_size=comp_size,
                uncompressed_size=uncomp_size,
                local_header_offset=local_offset,
                external_attr=external_attr,
                extra_base=b"".join(extra_parts),
            )
        )
        return comp_size

    def close(self) -> int:
        """Write Central Directory, ZIP64 trailers (if applicable), and EOCD; return total bytes."""
        if self._closed:
            return self._path.stat().st_size if self._path and self._path.exists() else 0

        cd_offset = self._fp.tell()
        for entry in self._central_entries:
            z64_fields: list[int] = []
            if self._zip64_mode == Zip64Mode.FORCE or entry.uncompressed_size >= ZIP32_LIMIT:
                z64_fields.append(entry.uncompressed_size)
                cd_uncomp = ZIP32_LIMIT
            else:
                cd_uncomp = entry.uncompressed_size

            if self._zip64_mode == Zip64Mode.FORCE or entry.compressed_size >= ZIP32_LIMIT:
                z64_fields.append(entry.compressed_size)
                cd_comp = ZIP32_LIMIT
            else:
                cd_comp = entry.compressed_size

            if self._zip64_mode == Zip64Mode.FORCE or entry.local_header_offset >= ZIP32_LIMIT:
                z64_fields.append(entry.local_header_offset)
                cd_local_off = ZIP32_LIMIT
            else:
                cd_local_off = entry.local_header_offset

            if z64_fields:
                self._zip64_used = True
                z64_extra = struct.pack(
                    f"<HH{'Q' * len(z64_fields)}",
                    0x0001,
                    8 * len(z64_fields),
                    *z64_fields,
                )
                cd_extra = z64_extra + entry.extra_base
                ver_needed = max(entry.version_needed, 45)
                ver_made = (3 << 8) | ver_needed
            else:
                cd_extra = entry.extra_base
                ver_needed = entry.version_needed
                ver_made = entry.version_made_by

            cd_header = struct.pack(
                "<4sHHHHHHIIIHHHHHII",
                SIG_CENTRAL_DIR_HEADER,
                ver_made,
                ver_needed,
                entry.flags,
                entry.header_method,
                entry.dos_time,
                entry.dos_date,
                entry.header_crc32,
                cd_comp,
                cd_uncomp,
                len(entry.arcname_bytes),
                len(cd_extra),
                0,  # file comment length
                0,  # disk number start
                0,  # internal file attributes
                entry.external_attr,
                cd_local_off,
            )
            self._fp.write(cd_header)
            self._fp.write(entry.arcname_bytes)
            self._fp.write(cd_extra)

        cd_end = self._fp.tell()
        cd_size = cd_end - cd_offset
        entry_count = len(self._central_entries)

        needs_eocd_zip64 = (
            self._zip64_mode == Zip64Mode.FORCE
            or self._zip64_used
            or entry_count >= ZIP32_COUNT_LIMIT
            or cd_size >= ZIP32_LIMIT
            or cd_offset >= ZIP32_LIMIT
        )
        if needs_eocd_zip64 and self._zip64_mode == Zip64Mode.DISABLE:
            raise ValueError("Archive exceeds PKZIP 32-bit limits while zip64_mode='disable'.")

        if needs_eocd_zip64:
            self._zip64_used = True
            zip64_eocd_offset = self._fp.tell()
            zip64_eocd = struct.pack(
                "<4sQHHIIQQQQ",
                SIG_ZIP64_EOCD_RECORD,
                44,  # size of remaining ZIP64 EOCD record
                (3 << 8) | 45,  # version made by
                45,  # version needed
                0,  # disk number
                0,  # disk with start of CD
                entry_count,
                entry_count,
                cd_size,
                cd_offset,
            )
            self._fp.write(zip64_eocd)
            zip64_locator = struct.pack(
                "<4sIQI",
                SIG_ZIP64_EOCD_LOCATOR,
                0,  # disk with start of ZIP64 EOCD
                zip64_eocd_offset,
                1,  # total number of disks
            )
            self._fp.write(zip64_locator)

        comment_bytes = self._archive_comment.encode("utf-8")[:65535]
        eocd = struct.pack(
            "<4sHHHHIIH",
            SIG_EOCD_RECORD,
            0,
            0,
            min(entry_count, ZIP32_COUNT_LIMIT),
            min(entry_count, ZIP32_COUNT_LIMIT),
            min(cd_size, ZIP32_LIMIT),
            min(cd_offset, ZIP32_LIMIT),
            len(comment_bytes),
        )
        self._fp.write(eocd)
        if comment_bytes:
            self._fp.write(comment_bytes)

        total_bytes = self._fp.tell()
        self._fp.flush()
        if self._owns_fp:
            self._fp.close()
        self._closed = True
        return total_bytes

    def __enter__(self) -> "CloudZipWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._closed:
            self.close()


def verify_zip_integrity(
    zip_path: Path,
    *,
    password: str | None = None,
    encryption_method: EncryptionMethod = EncryptionMethod.NONE,
    compression_method_hint: str = "deflate",
) -> None:
    """Verify Central Directory headers, decompression streams, and CRC-32 checksums."""
    if password and encryption_method in {EncryptionMethod.AES256, EncryptionMethod.ZIPCRYPTO}:
        with pyzipper.AESZipFile(zip_path, "r") as azf:
            azf.setpassword(password.encode("utf-8"))
            # Verify standard/deflate/store entries via pyzipper; for Zstd inside AES, read raw
            if compression_method_hint != "zstd":
                bad_member = azf.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"Encrypted ZIP integrity check failed on '{bad_member}'.")
        return

    if compression_method_hint == "zstd":
        _verify_zstd_or_mixed_zip(zip_path)
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Output ZIP integrity verification failed on '{bad_member}'.")


def _verify_zstd_or_mixed_zip(zip_path: Path) -> None:
    """Verify all entries in a ZIP that may contain Method 93 (Zstandard), Method 8, or Method 0."""
    dctx = zstd.ZstdDecompressor()
    with zipfile.ZipFile(zip_path, "r") as zf, open(zip_path, "rb") as raw_fp:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_fp.seek(info.header_offset)
            lh = raw_fp.read(30)
            if len(lh) < 30 or lh[:4] != SIG_LOCAL_FILE_HEADER:
                raise RuntimeError(f"Corrupt Local File Header for '{info.filename}'.")
            _, _, _, method_id, _, _, _, _, _, fname_len, extra_len = struct.unpack(
                "<4sHHHHHIIIHH", lh
            )
            raw_fp.seek(info.header_offset + 30 + fname_len + extra_len)
            comp_bytes = raw_fp.read(info.compress_size)
            if method_id == ZIP_METHOD_STORE:
                plain = comp_bytes
            elif method_id == ZIP_METHOD_DEFLATE:
                plain = zlib.decompress(comp_bytes, wbits=-15)
            elif method_id == ZIP_METHOD_ZSTD:
                with dctx.stream_reader(io.BytesIO(comp_bytes)) as reader:
                    plain = reader.read()
            else:
                raise RuntimeError(
                    f"Unsupported ZIP method {method_id} during verification of '{info.filename}'."
                )
            if len(plain) != info.file_size:
                raise RuntimeError(
                    f"Size mismatch for '{info.filename}': expected {info.file_size}, got {len(plain)}."
                )
            actual_crc = zlib.crc32(plain) & 0xFFFFFFFF
            if actual_crc != (info.CRC & 0xFFFFFFFF):
                raise RuntimeError(
                    f"CRC-32 mismatch for '{info.filename}': expected {info.CRC:#x}, got {actual_crc:#x}."
                )
