"""Tests for parallel load_references: equivalence + cache validity."""

import shutil

import numpy as np
import pytest

from scanner import load_references


def test_load_references_parallel_equals_sequential(
    tmp_path, sample_face_path, monkeypatch
):
    """For 12+ ref images, parallel loader must produce same encodings as sequential."""
    rdir = tmp_path / "refs"
    alice = rdir / "alice"
    bob = rdir / "bob"
    alice.mkdir(parents=True)
    bob.mkdir()
    for i in range(7):
        shutil.copy(sample_face_path, alice / f"alice_{i}.jpg")
    for i in range(7):
        shutil.copy(sample_face_path, bob / f"bob_{i}.jpg")

    cache_file = rdir / ".encodings_cache.pkl"

    monkeypatch.setenv("FACE_FINDER_PARALLEL", "0")
    if cache_file.exists():
        cache_file.unlink()
    sequential = load_references(str(rdir), tolerance=0.6)

    monkeypatch.setenv("FACE_FINDER_PARALLEL", "1")
    cache_file.unlink()
    parallel = load_references(str(rdir), tolerance=0.6)

    assert sorted(sequential.keys()) == sorted(parallel.keys())
    for name in sequential:
        assert len(sequential[name]) == len(parallel[name])
        for seq_enc, par_enc in zip(sequential[name], parallel[name]):
            assert np.allclose(seq_enc, par_enc, atol=1e-6)
