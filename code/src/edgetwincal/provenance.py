"""Provenance-complete, corruption-detecting caches for EdgeTwinCal.

The confirmatory campaign must never decide that a cache is reusable from a
schema version alone.  This module therefore separates the *cache key* (all
inputs that can affect frozen feature extraction) from payload integrity data,
stores both in a single binary envelope, and validates them before returning
any bytes.

The envelope is intentionally dependency free.  Callers may put an ``npz``, a
``torch.save`` blob, or another serialization in the payload; this module does
not deserialize untrusted payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .paths import PROJECT_ROOT


CACHE_SCHEMA_VERSION = 3
CACHE_SUFFIX = ".etcache"
_MAGIC = b"EDGETWINCAL_CACHE_V3\n"
_HEADER_LENGTH = struct.Struct(">Q")
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


# Every item below participates in the filename digest.  The superset of the
# handoff minimum is deliberate: APN patch/mode, protocol, and group identity
# are extraction-relevant and must not be silently aliased.
REQUIRED_KEY_FIELDS = (
    "schema_version",
    "method_version",
    "project_commit",
    "apn_commit",
    "apn_patch_sha256",
    "apn_mode",
    "checkpoint_sha256",
    "resolved_config_sha256",
    "dataset_id",
    "dataset_raw_sha256",
    "dataset_processed_sha256",
    "protocol_manifest_sha256",
    "split_manifest_sha256",
    "sample_ids_sha256",
    "group_ids_sha256",
    "normalizer_manifest_sha256",
    "loader_source_sha256",
    "extractor_source_sha256",
    "seed",
    "history_length",
    "horizon",
    "shapes",
    "dtypes",
    "mask_sha256",
)
PAYLOAD_FIELDS = ("payload_sha256", "payload_num_bytes")
REQUIRED_MANIFEST_FIELDS = REQUIRED_KEY_FIELDS + PAYLOAD_FIELDS


class CacheError(RuntimeError):
    """Base class for cache safety failures."""


class CacheManifestError(CacheError):
    """A manifest is missing, non-canonical, or structurally invalid."""


class StaleCacheError(CacheError):
    """A valid cache was produced from provenance other than the expected one."""

    def __init__(self, mismatches: Mapping[str, tuple[Any, Any]]) -> None:
        self.mismatches = dict(mismatches)
        summary = ", ".join(sorted(self.mismatches))
        super().__init__(f"Stale cache provenance fields: {summary}")


class CorruptCacheError(CacheError):
    """A cache fails envelope, manifest, filename, or payload validation."""


class PartialCacheError(CorruptCacheError):
    """A cache is truncated or otherwise incomplete."""


class CacheLockTimeout(CacheError):
    """A same-directory cache lock could not be acquired in time."""


class CacheBoundaryError(CacheError):
    """A requested cache path escapes its explicitly allowed root."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CacheManifestError(f"Value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_plain_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CacheManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CacheManifestError(f"{name} must be a non-empty trimmed string")
    return value


def _require_hash(name: str, value: Any, pattern: re.Pattern[str]) -> str:
    text = _require_text(name, value).lower()
    if pattern.fullmatch(text) is None:
        expected = 64 if pattern is _SHA256_RE else 40
        raise CacheManifestError(f"{name} must be a {expected}-character hex digest")
    return text


def _normalize_shapes(value: Any) -> Mapping[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise CacheManifestError("shapes must be a non-empty mapping")
    normalized: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_shape in value.items():
        name = _require_text("shape name", raw_name)
        if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Sequence):
            raise CacheManifestError(f"shapes[{name!r}] must be a sequence")
        shape = tuple(
            _require_plain_int(f"shapes[{name!r}] dimension", dim, minimum=0)
            for dim in raw_shape
        )
        if not shape:
            raise CacheManifestError(f"shapes[{name!r}] must not be empty")
        normalized[name] = shape
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_dtypes(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise CacheManifestError("dtypes must be a non-empty mapping")
    normalized = {
        _require_text("dtype name", name): _require_text(f"dtypes[{name!r}]", dtype)
        for name, dtype in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class CacheExpectation:
    """A validated cache key, without a payload hash known in advance."""

    _fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self._fields) != set(REQUIRED_KEY_FIELDS):
            missing = sorted(set(REQUIRED_KEY_FIELDS) - set(self._fields))
            extra = sorted(set(self._fields) - set(REQUIRED_KEY_FIELDS))
            raise CacheManifestError(
                f"Cache expectation fields differ; missing={missing}, extra={extra}"
            )
        # CacheManifest.build is the single field validator.  A zero-byte
        # payload is used only to validate and canonicalize the key fields.
        checked = CacheManifest.build(payload=b"", **dict(self._fields))
        object.__setattr__(self, "_fields", MappingProxyType(checked.key_dict()))

    def key_dict(self) -> dict[str, Any]:
        return _copy_key_dict(self._fields)

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.key_dict()))


@dataclass(frozen=True)
class CacheManifest:
    """Complete provenance and payload-integrity metadata for one cache."""

    schema_version: int
    method_version: str
    project_commit: str
    apn_commit: str
    apn_patch_sha256: str
    apn_mode: str
    checkpoint_sha256: str
    resolved_config_sha256: str
    dataset_id: str
    dataset_raw_sha256: str
    dataset_processed_sha256: str
    protocol_manifest_sha256: str
    split_manifest_sha256: str
    sample_ids_sha256: str
    group_ids_sha256: str
    normalizer_manifest_sha256: str
    loader_source_sha256: str
    extractor_source_sha256: str
    seed: int
    history_length: int
    horizon: int
    shapes: Mapping[str, tuple[int, ...]]
    dtypes: Mapping[str, str]
    mask_sha256: str
    payload_sha256: str
    payload_num_bytes: int

    def __post_init__(self) -> None:
        schema = _require_plain_int("schema_version", self.schema_version, minimum=0)
        if schema != CACHE_SCHEMA_VERSION:
            raise CacheManifestError(
                f"schema_version must be {CACHE_SCHEMA_VERSION}; schema-2 is legacy-only"
            )
        object.__setattr__(self, "method_version", _require_text("method_version", self.method_version))
        object.__setattr__(self, "project_commit", _require_hash("project_commit", self.project_commit, _GIT_SHA1_RE))
        object.__setattr__(self, "apn_commit", _require_hash("apn_commit", self.apn_commit, _GIT_SHA1_RE))
        for name in (
            "apn_patch_sha256",
            "checkpoint_sha256",
            "resolved_config_sha256",
            "dataset_raw_sha256",
            "dataset_processed_sha256",
            "protocol_manifest_sha256",
            "split_manifest_sha256",
            "sample_ids_sha256",
            "group_ids_sha256",
            "normalizer_manifest_sha256",
            "loader_source_sha256",
            "extractor_source_sha256",
            "mask_sha256",
            "payload_sha256",
        ):
            object.__setattr__(self, name, _require_hash(name, getattr(self, name), _SHA256_RE))
        object.__setattr__(self, "apn_mode", _require_text("apn_mode", self.apn_mode))
        object.__setattr__(self, "dataset_id", _require_text("dataset_id", self.dataset_id))
        object.__setattr__(self, "seed", _require_plain_int("seed", self.seed, minimum=0))
        object.__setattr__(self, "history_length", _require_plain_int("history_length", self.history_length, minimum=1))
        object.__setattr__(self, "horizon", _require_plain_int("horizon", self.horizon, minimum=1))
        object.__setattr__(self, "payload_num_bytes", _require_plain_int("payload_num_bytes", self.payload_num_bytes, minimum=0))
        object.__setattr__(self, "shapes", _normalize_shapes(self.shapes))
        object.__setattr__(self, "dtypes", _normalize_dtypes(self.dtypes))
        if set(self.shapes) != set(self.dtypes):
            raise CacheManifestError("shapes and dtypes must name exactly the same tensors")

    @classmethod
    def build(cls, *, payload: bytes | bytearray | memoryview, **key_fields: Any) -> "CacheManifest":
        """Build a complete manifest and calculate payload integrity fields."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        missing = sorted(set(REQUIRED_KEY_FIELDS) - set(key_fields))
        extra = sorted(set(key_fields) - set(REQUIRED_KEY_FIELDS))
        if missing or extra:
            raise CacheManifestError(
                f"Cache key fields differ; missing={missing}, extra={extra}"
            )
        raw = bytes(payload)
        return cls(
            **key_fields,
            payload_sha256=_sha256_bytes(raw),
            payload_num_bytes=len(raw),
        )

    @classmethod
    def expectation(cls, **key_fields: Any) -> CacheExpectation:
        """Validate extraction provenance before trying to load a cache."""

        return CacheExpectation(key_fields)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheManifest":
        """Load a disk manifest, rejecting missing, extra, and noncanonical data."""

        if not isinstance(value, Mapping):
            raise CacheManifestError("Cache manifest must be a mapping")
        missing = sorted(set(REQUIRED_MANIFEST_FIELDS) - set(value))
        extra = sorted(set(value) - set(REQUIRED_MANIFEST_FIELDS))
        if missing or extra:
            raise CacheManifestError(
                f"Cache manifest fields differ; missing={missing}, extra={extra}"
            )
        manifest = cls(**dict(value))
        if manifest.to_dict() != dict(value):
            raise CacheManifestError("Cache manifest is not in canonical form")
        return manifest

    def key_dict(self) -> dict[str, Any]:
        return _copy_key_dict({name: getattr(self, name) for name in REQUIRED_KEY_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        result = self.key_dict()
        result.update(
            payload_sha256=self.payload_sha256,
            payload_num_bytes=self.payload_num_bytes,
        )
        return result

    def digest(self) -> str:
        """Return the canonical extraction-key SHA256 used in the filename."""

        return _sha256_bytes(_canonical_json_bytes(self.key_dict()))

    def manifest_digest(self) -> str:
        """Return a digest that also commits to payload identity and size."""

        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def expectation_view(self) -> CacheExpectation:
        return CacheExpectation(self.key_dict())

    def validate_against(self, expected: "CacheManifest | CacheExpectation") -> None:
        """Compare every expected field and report all mismatches at once."""

        if isinstance(expected, CacheManifest):
            expected_fields = expected.to_dict()
            actual_fields = self.to_dict()
        elif isinstance(expected, CacheExpectation):
            expected_fields = expected.key_dict()
            actual_fields = self.key_dict()
        else:
            raise TypeError("expected must be CacheManifest or CacheExpectation")
        mismatches = {
            name: (expected_fields[name], actual_fields[name])
            for name in expected_fields
            if _canonical_json_bytes(expected_fields[name])
            != _canonical_json_bytes(actual_fields[name])
        }
        if mismatches:
            raise StaleCacheError(mismatches)

    def validate_payload(self, payload: bytes | bytearray | memoryview) -> None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        raw = bytes(payload)
        if len(raw) != self.payload_num_bytes:
            raise PartialCacheError(
                f"Payload length is {len(raw)}, expected {self.payload_num_bytes}"
            )
        actual = _sha256_bytes(raw)
        if actual != self.payload_sha256:
            raise CorruptCacheError(
                f"Payload SHA256 mismatch: expected {self.payload_sha256}, got {actual}"
            )


def _copy_key_dict(fields: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: fields[name] for name in REQUIRED_KEY_FIELDS}
    result["shapes"] = {name: list(shape) for name, shape in fields["shapes"].items()}
    result["dtypes"] = dict(fields["dtypes"])
    return result


def resolve_cache_path(
    path: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
    create_parent: bool = False,
) -> Path:
    """Resolve ``path`` and prove it remains under ``root`` before any write."""

    root_path = Path(root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise CacheBoundaryError(f"Cache path escapes root: {resolved}") from exc
    if resolved == root_path:
        raise CacheBoundaryError("Cache path must name a file below the cache root")
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # Resolve again after creation so an existing directory link cannot
        # turn a lexically safe path into an external write target.
        resolved = resolved.resolve(strict=False)
        try:
            resolved.relative_to(root_path)
        except ValueError as exc:
            raise CacheBoundaryError(f"Cache path escapes root: {resolved}") from exc
    return resolved


def cache_file_path(
    directory: str | Path,
    stem: str,
    manifest: CacheManifest | CacheExpectation,
    *,
    root: str | Path = PROJECT_ROOT,
) -> Path:
    """Construct ``<stem>.<key-digest>.etcache`` inside the allowed root."""

    if _SAFE_STEM_RE.fullmatch(stem) is None:
        raise CacheManifestError(f"Unsafe cache filename stem: {stem!r}")
    directory_path = resolve_cache_path(
        Path(directory) / "directory-sentinel",
        root=root,
        create_parent=True,
    ).parent
    return resolve_cache_path(
        directory_path / f"{stem}.{manifest.digest()}{CACHE_SUFFIX}",
        root=root,
    )


class _CacheLock:
    """Small O_EXCL lock scoped to the cache directory and exact cache name."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        poll_seconds: float,
        stale_after_seconds: float | None,
    ) -> None:
        if timeout_seconds < 0 or poll_seconds <= 0:
            raise ValueError("Invalid cache lock timing")
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.stale_after_seconds = stale_after_seconds
        self.token = secrets.token_hex(16)
        self._owned = False

    def __enter__(self) -> "_CacheLock":
        deadline = time.monotonic() + self.timeout_seconds
        record = _canonical_json_bytes(
            {"pid": os.getpid(), "token": self.token, "created_unix_ns": time.time_ns()}
        )
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if self._remove_stale_lock():
                    continue
                if time.monotonic() >= deadline:
                    raise CacheLockTimeout(f"Timed out acquiring cache lock: {self.path}")
                time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))
                continue
            try:
                os.write(descriptor, record)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._owned = True
            return self

    def _remove_stale_lock(self) -> bool:
        if self.stale_after_seconds is None:
            return False
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return True
        if age <= self.stale_after_seconds:
            return False
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._owned:
            return
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if record.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            # Never delete a lock that can no longer be proven to be ours.
            pass
        finally:
            self._owned = False


def _require_digest_filename(path: Path, digest: str) -> None:
    if digest not in path.name.split("."):
        raise CorruptCacheError(
            f"Cache filename does not contain its canonical key digest: {path.name}"
        )


def write_cache_atomic(
    path: str | Path,
    manifest: CacheManifest,
    payload: bytes | bytearray | memoryview,
    *,
    root: str | Path = PROJECT_ROOT,
    lock_timeout_seconds: float = 30.0,
    lock_poll_seconds: float = 0.05,
    stale_lock_seconds: float | None = 600.0,
) -> Path:
    """Write one complete cache envelope using a same-directory atomic replace."""

    if not isinstance(manifest, CacheManifest):
        raise TypeError("manifest must be CacheManifest")
    raw_payload = bytes(payload)
    manifest.validate_payload(raw_payload)
    destination = resolve_cache_path(path, root=root, create_parent=True)
    _require_digest_filename(destination, manifest.digest())
    header = {
        "cache_key_sha256": manifest.digest(),
        "manifest": manifest.to_dict(),
    }
    raw_header = _canonical_json_bytes(header)
    if len(raw_header) > _MAX_HEADER_BYTES:
        raise CacheManifestError("Cache header exceeds the safety limit")
    lock_path = destination.with_name(destination.name + ".lock")
    temporary_path: Path | None = None
    with _CacheLock(
        lock_path,
        timeout_seconds=lock_timeout_seconds,
        poll_seconds=lock_poll_seconds,
        stale_after_seconds=stale_lock_seconds,
    ):
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(_MAGIC)
                stream.write(_HEADER_LENGTH.pack(len(raw_header)))
                stream.write(raw_header)
                stream.write(raw_payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return destination


def read_cache(
    path: str | Path,
    expected: CacheManifest | CacheExpectation,
    *,
    root: str | Path = PROJECT_ROOT,
) -> tuple[CacheManifest, bytes]:
    """Validate a cache envelope and return bytes only after every check passes."""

    if not isinstance(expected, (CacheManifest, CacheExpectation)):
        raise TypeError("expected must be CacheManifest or CacheExpectation")
    source = resolve_cache_path(path, root=root)
    with source.open("rb") as stream:
        magic = stream.read(len(_MAGIC))
        if len(magic) != len(_MAGIC):
            raise PartialCacheError("Cache ended inside the magic header")
        if magic != _MAGIC:
            raise CorruptCacheError("Cache magic/version is invalid")
        length_bytes = stream.read(_HEADER_LENGTH.size)
        if len(length_bytes) != _HEADER_LENGTH.size:
            raise PartialCacheError("Cache ended before the manifest length")
        (header_length,) = _HEADER_LENGTH.unpack(length_bytes)
        if header_length == 0 or header_length > _MAX_HEADER_BYTES:
            raise CorruptCacheError(f"Invalid cache header length: {header_length}")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise PartialCacheError("Cache ended inside the manifest header")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptCacheError(f"Cache header is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(header, Mapping) or set(header) != {"cache_key_sha256", "manifest"}:
            raise CorruptCacheError("Cache envelope header fields are invalid")
        if _canonical_json_bytes(header) != raw_header:
            raise CorruptCacheError("Cache envelope header is not canonical JSON")
        try:
            manifest = CacheManifest.from_dict(header["manifest"])
        except CacheManifestError as exc:
            raise CorruptCacheError(f"Invalid cache manifest: {exc}") from exc
        if header["cache_key_sha256"] != manifest.digest():
            raise CorruptCacheError("Cache key digest does not match its manifest")
        _require_digest_filename(source, manifest.digest())
        manifest.validate_against(expected)
        payload = stream.read()
    manifest.validate_payload(payload)
    return manifest, payload


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CACHE_SUFFIX",
    "REQUIRED_KEY_FIELDS",
    "REQUIRED_MANIFEST_FIELDS",
    "CacheError",
    "CacheManifestError",
    "StaleCacheError",
    "CorruptCacheError",
    "PartialCacheError",
    "CacheLockTimeout",
    "CacheBoundaryError",
    "CacheExpectation",
    "CacheManifest",
    "resolve_cache_path",
    "cache_file_path",
    "write_cache_atomic",
    "read_cache",
]
