from __future__ import annotations

import copy
from pathlib import Path

import pytest

from edgetwincal.provenance import (
    CACHE_SCHEMA_VERSION,
    REQUIRED_KEY_FIELDS,
    CacheBoundaryError,
    CacheLockTimeout,
    CacheManifest,
    CacheManifestError,
    CorruptCacheError,
    PartialCacheError,
    StaleCacheError,
    cache_file_path,
    read_cache,
    resolve_cache_path,
    write_cache_atomic,
)


def _hash(character: str) -> str:
    return character * 64


def _commit(character: str) -> str:
    return character * 40


@pytest.fixture
def key_fields() -> dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method_version": "msn2026_v1",
        "project_commit": _commit("1"),
        "apn_commit": _commit("2"),
        "apn_patch_sha256": _hash("3"),
        "apn_mode": "upstream_parity",
        "checkpoint_sha256": _hash("4"),
        "resolved_config_sha256": _hash("5"),
        "dataset_id": "P12",
        "dataset_raw_sha256": _hash("6"),
        "dataset_processed_sha256": _hash("7"),
        "protocol_manifest_sha256": _hash("8"),
        "split_manifest_sha256": _hash("9"),
        "sample_ids_sha256": _hash("a"),
        "group_ids_sha256": _hash("b"),
        "normalizer_manifest_sha256": _hash("c"),
        "loader_source_sha256": _hash("d"),
        "extractor_source_sha256": _hash("e"),
        "seed": 2024,
        "history_length": 36,
        "horizon": 3,
        "shapes": {
            "forecast": [17, 3, 36],
            "latent": [17, 36, 24],
            "mask": [17, 3, 36],
        },
        "dtypes": {
            "forecast": "float32",
            "latent": "float32",
            "mask": "bool",
        },
        "mask_sha256": _hash("f"),
    }


def _manifest(key_fields: dict[str, object], payload: bytes = b"frozen features") -> CacheManifest:
    return CacheManifest.build(payload=payload, **copy.deepcopy(key_fields))


def _mutate(fields: dict[str, object], name: str) -> None:
    if name == "schema_version":
        fields[name] = 2
    elif name in {"project_commit", "apn_commit"}:
        fields[name] = _commit("0")
    elif name.endswith("sha256"):
        fields[name] = _hash("0")
    elif name in {"seed", "history_length", "horizon"}:
        fields[name] = int(fields[name]) + 1
    elif name == "shapes":
        shapes = copy.deepcopy(fields[name])
        assert isinstance(shapes, dict)
        shapes["latent"][2] += 1
        fields[name] = shapes
    elif name == "dtypes":
        dtypes = copy.deepcopy(fields[name])
        assert isinstance(dtypes, dict)
        dtypes["latent"] = "float64"
        fields[name] = dtypes
    else:
        fields[name] = f"changed_{fields[name]}"


def test_round_trip_validates_manifest_filename_and_payload(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = b"\x00\x01serialized frozen feature tensors\xff"
    manifest = _manifest(key_fields, payload)
    path = cache_file_path("cache/features", "p12-seed2024", manifest, root=root)

    written = write_cache_atomic(path, manifest, payload, root=root)
    loaded_manifest, loaded_payload = read_cache(
        written,
        manifest.expectation_view(),
        root=root,
    )

    assert written == path
    assert loaded_manifest.to_dict() == manifest.to_dict()
    assert loaded_payload == payload
    assert manifest.digest() in path.name.split(".")
    assert not list(path.parent.glob("*.lock"))
    assert not list(path.parent.glob("*.tmp"))


def test_digest_is_canonical_and_payload_independent(key_fields: dict[str, object]) -> None:
    first = _manifest(key_fields, b"payload one")
    reordered = copy.deepcopy(key_fields)
    reordered["shapes"] = dict(reversed(list(reordered["shapes"].items())))
    reordered["dtypes"] = dict(reversed(list(reordered["dtypes"].items())))
    second = _manifest(reordered, b"payload two")

    assert first.digest() == second.digest()
    assert first.expectation_view().digest() == first.digest()
    assert first.manifest_digest() != second.manifest_digest()
    assert first.to_dict()["shapes"] == {
        "forecast": [17, 3, 36],
        "latent": [17, 36, 24],
        "mask": [17, 3, 36],
    }


@pytest.mark.parametrize("field", REQUIRED_KEY_FIELDS)
def test_every_cache_key_field_is_required_and_strictly_checked(
    field: str,
    key_fields: dict[str, object],
) -> None:
    expected = _manifest(key_fields).expectation_view()
    changed = copy.deepcopy(key_fields)
    _mutate(changed, field)

    if field == "schema_version":
        with pytest.raises(CacheManifestError, match="legacy-only"):
            CacheManifest.expectation(**changed)
        return

    actual = _manifest(changed)
    with pytest.raises(StaleCacheError) as caught:
        actual.validate_against(expected)
    assert set(caught.value.mismatches) == {field}
    expected_value, actual_value = caught.value.mismatches[field]
    assert expected_value != actual_value


@pytest.mark.parametrize("field", REQUIRED_KEY_FIELDS)
def test_missing_cache_key_field_is_rejected(
    field: str,
    key_fields: dict[str, object],
) -> None:
    incomplete = copy.deepcopy(key_fields)
    del incomplete[field]
    with pytest.raises(CacheManifestError, match="missing="):
        CacheManifest.build(payload=b"payload", **incomplete)


def test_extra_key_and_manifest_fields_are_rejected(key_fields: dict[str, object]) -> None:
    extra_key = {**key_fields, "untracked_setting": "unsafe"}
    with pytest.raises(CacheManifestError, match="extra="):
        CacheManifest.build(payload=b"payload", **extra_key)

    manifest_dict = _manifest(key_fields).to_dict()
    manifest_dict["untracked_setting"] = "unsafe"
    with pytest.raises(CacheManifestError, match="extra="):
        CacheManifest.from_dict(manifest_dict)


def test_payload_hash_and_size_are_strictly_checked(key_fields: dict[str, object]) -> None:
    first = _manifest(key_fields, b"payload-one")
    second = _manifest(key_fields, b"payload-two-is-longer")
    with pytest.raises(StaleCacheError) as caught:
        first.validate_against(second)
    assert set(caught.value.mismatches) == {"payload_sha256", "payload_num_bytes"}

    with pytest.raises(PartialCacheError, match="Payload length"):
        first.validate_payload(b"short")
    same_size_corruption = bytearray(b"payload-one")
    same_size_corruption[-1] ^= 1
    with pytest.raises(CorruptCacheError, match="Payload SHA256 mismatch"):
        first.validate_payload(same_size_corruption)


def test_corrupt_payload_is_rejected_before_bytes_are_returned(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = b"payload with integrity"
    manifest = _manifest(key_fields, payload)
    path = cache_file_path("cache", "features", manifest, root=root)
    write_cache_atomic(path, manifest, payload, root=root)
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)

    with pytest.raises(CorruptCacheError, match="Payload SHA256 mismatch"):
        read_cache(path, manifest.expectation_view(), root=root)


@pytest.mark.parametrize("cut", [1, 8, 22, 30, -1])
def test_partial_cache_is_rejected(
    cut: int,
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = b"payload long enough to truncate"
    manifest = _manifest(key_fields, payload)
    path = cache_file_path("cache", "features", manifest, root=root)
    write_cache_atomic(path, manifest, payload, root=root)
    complete = path.read_bytes()
    partial = complete[:cut] if cut >= 0 else complete[:cut]
    path.write_bytes(partial)

    with pytest.raises(PartialCacheError):
        read_cache(path, manifest.expectation_view(), root=root)


def test_manifest_header_corruption_and_wrong_filename_are_rejected(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = _manifest(key_fields)
    path = cache_file_path("cache", "features", manifest, root=root)
    write_cache_atomic(path, manifest, b"frozen features", root=root)

    wrong_name = path.with_name("features." + _hash("0") + ".etcache")
    path.replace(wrong_name)
    with pytest.raises(CorruptCacheError, match="filename"):
        read_cache(wrong_name, manifest.expectation_view(), root=root)

    wrong_name.replace(path)
    damaged = bytearray(path.read_bytes())
    damaged[0] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(CorruptCacheError, match="magic"):
        read_cache(path, manifest.expectation_view(), root=root)


def test_stale_cache_reports_the_exact_changed_fields_on_load(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    actual = _manifest(key_fields)
    path = cache_file_path("cache", "features", actual, root=root)
    write_cache_atomic(path, actual, b"frozen features", root=root)
    expected_fields = copy.deepcopy(key_fields)
    expected_fields["checkpoint_sha256"] = _hash("0")
    expected_fields["seed"] = 2025
    expected = CacheManifest.expectation(**expected_fields)

    with pytest.raises(StaleCacheError) as caught:
        read_cache(path, expected, root=root)
    assert set(caught.value.mismatches) == {"checkpoint_sha256", "seed"}


def test_atomic_writer_uses_lock_and_preserves_existing_cache_on_contention(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first_payload = b"frozen features"
    manifest = _manifest(key_fields, first_payload)
    path = cache_file_path("cache", "features", manifest, root=root)
    write_cache_atomic(path, manifest, first_payload, root=root)
    original = path.read_bytes()
    lock_path = path.with_name(path.name + ".lock")
    lock_path.write_text("occupied", encoding="utf-8")

    try:
        with pytest.raises(CacheLockTimeout):
            write_cache_atomic(
                path,
                manifest,
                first_payload,
                root=root,
                lock_timeout_seconds=0.01,
                lock_poll_seconds=0.002,
                stale_lock_seconds=None,
            )
        assert path.read_bytes() == original
        assert not list(path.parent.glob("*.tmp"))
    finally:
        lock_path.unlink(missing_ok=True)


def test_root_boundary_blocks_read_write_and_directory_escape(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside" / "features.etcache"
    manifest = _manifest(key_fields)

    with pytest.raises(CacheBoundaryError, match="escapes root"):
        resolve_cache_path(outside, root=root, create_parent=True)
    with pytest.raises(CacheBoundaryError, match="escapes root"):
        write_cache_atomic(outside, manifest, b"frozen features", root=root)
    with pytest.raises(CacheBoundaryError, match="escapes root"):
        read_cache(outside, manifest.expectation_view(), root=root)
    with pytest.raises(CacheBoundaryError, match="escapes root"):
        cache_file_path("../outside", "features", manifest, root=root)
    assert not outside.exists()


def test_filename_digest_is_mandatory_before_write(
    tmp_path: Path,
    key_fields: dict[str, object],
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = _manifest(key_fields)
    unsafe_alias = root / "cache" / "features.etcache"
    with pytest.raises(CorruptCacheError, match="filename"):
        write_cache_atomic(unsafe_alias, manifest, b"frozen features", root=root)
    assert not unsafe_alias.exists()


def test_disk_manifest_must_be_canonical(key_fields: dict[str, object]) -> None:
    value = _manifest(key_fields).to_dict()
    value["sample_ids_sha256"] = value["sample_ids_sha256"].upper()
    with pytest.raises(CacheManifestError, match="canonical"):
        CacheManifest.from_dict(value)
