"""Archive entry filtering, path transformation, deduplication, and ordering pipeline.

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cloudzip.extractor import ExtractedItem, sanitize_archive_path
from cloudzip.models import ConversionOptions, SortOrder

OS_JUNK_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini", "icon\r"})


@dataclass(slots=True)
class ManipulatedEntry:
    """Archive entry after filtering, path rewriting, and collision resolution."""

    arcname: str
    disk_path: Path
    is_dir: bool
    size_bytes: int
    mtime: datetime
    posix_mode: int


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    norm = pattern.replace("\\", "/").strip("/")
    # Translate **, *, ? cleanly into a full-string regex
    i = 0
    n = len(norm)
    parts: list[str] = ["^"]
    while i < n:
        ch = norm[i]
        if ch == "*":
            if i + 1 < n and norm[i + 1] == "*":
                i += 2
                if i < n and norm[i] == "/":
                    i += 1
                    parts.append("(?:.+/)?")
                else:
                    parts.append(".*")
            else:
                i += 1
                parts.append("[^/]*")
        elif ch == "?":
            i += 1
            parts.append("[^/]")
        else:
            parts.append(re.escape(ch))
            i += 1
    parts.append("$")
    return re.compile("".join(parts))


def _matches_any_glob(rel_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    basename = rel_path.rsplit("/", 1)[-1]
    for pat in patterns:
        clean_pat = pat.replace("\\", "/").strip()
        if not clean_pat:
            continue
        if "/" not in clean_pat.strip("/"):
            if fnmatch.fnmatch(basename, clean_pat) or fnmatch.fnmatch(rel_path, clean_pat):
                return True
        if _glob_to_regex(clean_pat).match(rel_path):
            return True
    return False


def _is_os_junk(rel_path: str) -> bool:
    segments = rel_path.split("/")
    if "__MACOSX" in segments:
        return True
    basename = segments[-1]
    if basename.lower() in OS_JUNK_NAMES:
        return True
    if basename.startswith("._"):
        return True
    return False


def _is_hidden_path(rel_path: str) -> bool:
    return any(seg.startswith(".") and seg not in {".", ".."} for seg in rel_path.split("/"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _disambiguate_path(candidate: str, used_lower: set[str]) -> str:
    if candidate.lower() not in used_lower:
        used_lower.add(candidate.lower())
        return candidate

    parent, _, name = candidate.rpartition("/")
    if "." in name and not name.startswith("."):
        stem, ext = name.rsplit(".", 1)
        ext_dot = f".{ext}"
    else:
        stem, ext_dot = name, ""

    counter = 1
    while True:
        alt_name = f"{stem}_{counter}{ext_dot}"
        alt_full = f"{parent}/{alt_name}" if parent else alt_name
        if alt_full.lower() not in used_lower:
            used_lower.add(alt_full.lower())
            return alt_full
        counter += 1


def apply_manipulation_pipeline(
    items: list[ExtractedItem],
    options: ConversionOptions,
) -> tuple[list[ManipulatedEntry], int]:
    """Transform extracted archive items according to `options`; return `(kept_entries, filtered_count)`."""
    if not items:
        return [], 0

    compiled_renames = [
        (re.compile(rule.pattern), rule.replacement) for rule in options.rename_rules
    ]

    # Detect single common top-level directory across files/non-root entries if unwrap_single_root
    unwrap_prefix: str | None = None
    if options.unwrap_single_root:
        file_or_deep = [it.rel_path for it in items if "/" in it.rel_path or not it.is_dir]
        if file_or_deep and all("/" in p for p in file_or_deep):
            roots = {p.split("/", 1)[0] for p in file_or_deep}
            if len(roots) == 1:
                unwrap_prefix = next(iter(roots))

    filtered_out = 0
    seen_hashes: set[str] = set()
    used_file_paths: set[str] = set()
    used_dir_paths: set[str] = set()
    file_entries: list[ManipulatedEntry] = []
    dir_candidates: list[ManipulatedEntry] = []

    for item in items:
        rel = item.rel_path

        # 1. OS junk & hidden filters on original path
        if options.exclude_junk_files and _is_os_junk(rel):
            filtered_out += 1
            continue
        if options.exclude_hidden_files and _is_hidden_path(rel):
            filtered_out += 1
            continue

        # 2. Unwrap single root folder
        if unwrap_prefix:
            if rel == unwrap_prefix:
                filtered_out += 1
                continue
            if rel.startswith(f"{unwrap_prefix}/"):
                rel = rel[len(unwrap_prefix) + 1 :]

        # 3. Strip N leading path components
        if options.strip_components > 0:
            segs = rel.split("/")
            if len(segs) <= options.strip_components:
                filtered_out += 1
                continue
            rel = "/".join(segs[options.strip_components :])

        # 4. Apply regex rename rules
        for regex, repl in compiled_renames:
            rel = regex.sub(repl, rel)
        rel = sanitize_archive_path(rel)
        if not rel:
            filtered_out += 1
            continue

        # 5. Include / exclude glob filters
        if options.exclude_patterns and _matches_any_glob(rel, options.exclude_patterns):
            filtered_out += 1
            continue
        if not item.is_dir and options.include_patterns and not _matches_any_glob(
            rel, options.include_patterns
        ):
            filtered_out += 1
            continue

        # 6. Flatten paths if requested
        if options.flatten_paths:
            if item.is_dir:
                filtered_out += 1
                continue
            rel = rel.rsplit("/", 1)[-1]

        # 7. Prepend root_prefix if configured
        if options.root_prefix:
            rel = f"{options.root_prefix}/{rel}"

        # 8. Content deduplication for regular files
        if not item.is_dir and options.deduplicate_identical:
            digest = _file_sha256(item.disk_path)
            if digest in seen_hashes:
                filtered_out += 1
                continue
            seen_hashes.add(digest)

        if item.is_dir:
            if not options.include_empty_dirs:
                filtered_out += 1
                continue
            if rel.lower() not in used_dir_paths:
                used_dir_paths.add(rel.lower())
                dir_candidates.append(
                    ManipulatedEntry(
                        arcname=rel,
                        disk_path=item.disk_path,
                        is_dir=True,
                        size_bytes=0,
                        mtime=item.mtime,
                        posix_mode=item.posix_mode,
                    )
                )
        else:
            unique_rel = _disambiguate_path(rel, used_file_paths)
            file_entries.append(
                ManipulatedEntry(
                    arcname=unique_rel,
                    disk_path=item.disk_path,
                    is_dir=False,
                    size_bytes=item.size_bytes,
                    mtime=item.mtime,
                    posix_mode=item.posix_mode,
                )
            )

    # Keep only genuinely empty directories (directories that contain files are implicitly
    # represented by their child file paths, avoiding redundant directory header bloat, OR
    # keep explicit empty leaf directories when include_empty_dirs=True).
    non_empty_ancestors: set[str] = set()
    for fe in file_entries:
        parts = fe.arcname.split("/")
        for depth in range(1, len(parts)):
            non_empty_ancestors.add("/".join(parts[:depth]).lower())
    for dc in dir_candidates:
        parts = dc.arcname.split("/")
        for depth in range(1, len(parts)):
            non_empty_ancestors.add("/".join(parts[:depth]).lower())

    kept_empty_dirs = [
        dc
        for dc in dir_candidates
        if dc.arcname.lower() not in non_empty_ancestors
        and (not options.include_patterns or _matches_any_glob(dc.arcname, options.include_patterns))
    ]

    combined = kept_empty_dirs + file_entries

    if options.sort_entries == SortOrder.PATH:
        combined.sort(key=lambda e: (not e.is_dir, e.arcname))
    elif options.sort_entries == SortOrder.EXTENSION:
        combined.sort(
            key=lambda e: (
                e.is_dir,
                e.arcname.rsplit(".", 1)[-1].lower() if "." in e.arcname.rsplit("/", 1)[-1] else "",
                e.arcname,
            )
        )
    elif options.sort_entries == SortOrder.SIZE_ASC:
        combined.sort(key=lambda e: (e.size_bytes, e.arcname))
    elif options.sort_entries == SortOrder.SIZE_DESC:
        combined.sort(key=lambda e: (-e.size_bytes, e.arcname))
    # SortOrder.PRESERVE keeps discovery order

    return combined, filtered_out
