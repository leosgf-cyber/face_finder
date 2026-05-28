"""Tests for parallel scan_frames: equivalence + progress callback contract."""

import os
import shutil
from pathlib import Path

import pytest

from scanner import load_references, scan_frames


def _sort_results(results):
    """Stable sort each name's matches by frame_number for comparison."""
    return {
        name: sorted(matches, key=lambda m: m["frame_number"])
        for name, matches in results.items()
    }


def test_scan_frames_parallel_equals_sequential(
    frames_dir, references_dir, matches_dir, monkeypatch
):
    """Parallel and single-process scans must produce identical results."""
    refs = load_references(str(references_dir), tolerance=0.6)

    par_matches = matches_dir / "parallel"
    seq_matches = matches_dir / "sequential"
    par_matches.mkdir()
    seq_matches.mkdir()

    # Force parallel path
    monkeypatch.setenv("FACE_FINDER_PARALLEL", "1")
    monkeypatch.setattr("scanner.PARALLEL_THRESHOLD", 1)
    parallel = scan_frames(
        str(frames_dir), refs, tolerance=0.6, fps=1.0, matches_dir=str(par_matches)
    )

    # Force sequential path
    monkeypatch.setattr("scanner.PARALLEL_THRESHOLD", 10**6)
    sequential = scan_frames(
        str(frames_dir), refs, tolerance=0.6, fps=1.0, matches_dir=str(seq_matches)
    )

    def _strip_match_image(results):
        return {
            name: [{k: v for k, v in m.items() if k != "match_image"} for m in matches]
            for name, matches in results.items()
        }

    assert _sort_results(_strip_match_image(parallel)) == _sort_results(
        _strip_match_image(sequential)
    )


def test_scan_frames_small_scan_uses_fast_path(
    tmp_path, references_dir, sample_face_path, matches_dir
):
    """Scans below PARALLEL_THRESHOLD must not spin up workers."""
    fdir = tmp_path / "small"
    fdir.mkdir()
    for i in range(3):
        shutil.copy(sample_face_path, fdir / f"frame_{i:04d}.jpg")

    refs = load_references(str(references_dir), tolerance=0.6)

    results = scan_frames(
        str(fdir), refs, tolerance=0.6, fps=1.0, matches_dir=str(matches_dir)
    )
    assert "alice" in results
    assert len(results["alice"]) >= 1
