from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import generate_asset


def _mesh(face_count):
    return SimpleNamespace(faces=np.zeros((face_count, 3), dtype=np.int64))


def test_default_has_no_decimation_limit():
    assert generate_asset._parse_target("none") is None
    assert generate_asset._parse_target("0") is None
    assert generate_asset._parse_target("250000") == 250000


def test_oom_diagnostic_recommends_safe_single_process_retry():
    message = generate_asset._watchdog_message(RuntimeError("MPS backend out of memory"))
    assert "--pipeline-type 512" in message
    assert "one TRELLIS process" in message


def test_auto_backend_selects_mps_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(generate_asset.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(generate_asset.platform, "machine", lambda: "arm64")
    assert generate_asset._resolve_backend("auto") == "mps"


def test_auto_backend_preserves_linux_cuda_route(monkeypatch):
    monkeypatch.setattr(generate_asset.platform, "system", lambda: "Linux")
    monkeypatch.setattr(generate_asset.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert generate_asset._resolve_backend("auto") == "cuda"


def test_auto_fallback_order_keeps_full_resolution_first():
    assert list(generate_asset._attempt_schedule("auto", None, 800_000)) == [
        ("metal", None),
        ("kdtree", None),
        ("metal", 200_000),
        ("kdtree", 200_000),
    ]


def test_simulated_bvh_failure_uses_full_kdtree(tmp_path: Path):
    calls = []

    def fake_export(mesh, path, *, baker, target, texture_size):
        calls.append((baker, target, texture_size))
        if baker == "metal":
            raise RuntimeError("simulated Metal BVH failure")
        path.write_bytes(b"candidate")
        return object(), False

    _, chosen, attempts = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="auto",
        requested_target=None,
        texture_size=1024,
        export_fn=fake_export,
    )

    assert calls == [("metal", None, 1024), ("kdtree", None, 1024)]
    assert chosen["baker"] == "kdtree"
    assert chosen["target_faces"] is None
    assert [attempt["status"] for attempt in attempts] == ["failed", "ok"]


def test_safety_candidate_is_only_after_both_full_attempts(tmp_path: Path):
    calls = []

    def fake_export(mesh, path, *, baker, target, texture_size):
        calls.append((baker, target))
        if target is None:
            raise RuntimeError("full-resolution baker failure")
        path.write_bytes(b"candidate")
        return object(), True

    _, chosen, _ = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="auto",
        requested_target=None,
        texture_size=512,
        export_fn=fake_export,
    )

    assert calls == [("metal", None), ("kdtree", None), ("metal", 200_000)]
    assert chosen["technical_safety_target"] is True
    assert chosen["pre_simplified_before_bvh"] is True


def test_explicit_kdtree_never_loads_metal():
    assert list(generate_asset._attempt_schedule("kdtree", None, 1000)) == [("kdtree", None)]


def test_explicit_200k_target_is_not_mislabeled_as_safety_fallback(tmp_path: Path):
    def fake_export(mesh, path, *, baker, target, texture_size):
        path.write_bytes(b"candidate")
        return object(), True

    _, chosen, _ = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="kdtree",
        requested_target=200_000,
        texture_size=512,
        export_fn=fake_export,
    )

    assert chosen["target_faces"] == 200_000
    assert chosen["technical_safety_target"] is False


def test_raw_export_preserves_full_topology_and_writes_vertex_normals(tmp_path: Path):
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ]
    )
    output = tmp_path / "raw_full.glb"
    stats = generate_asset._export_raw(SimpleNamespace(vertices=vertices, faces=faces), output)

    from trellis2.gltf_validation import inspect_glb

    validated = inspect_glb(output, require_pbr=False)
    assert stats["vertices"] == validated["vertices"] == 8
    assert stats["triangles"] == validated["triangles"] == 12
