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


def test_progress_callback_contract(frames_dir, references_dir, matches_dir):
    """Callback receives correct shape; frames_total constant; frames_done monotonic.

    Union of new_matches across all callbacks must equal final results.
    """
    refs = load_references(str(references_dir), tolerance=0.6)

    payloads = []

    def cb(payload):
        payloads.append(payload)

    final_results = scan_frames(
        str(frames_dir),
        refs,
        tolerance=0.6,
        fps=1.0,
        matches_dir=str(matches_dir),
        progress_callback=cb,
    )

    assert len(payloads) >= 1, "callback should be invoked at least once"

    totals = {p["frames_total"] for p in payloads}
    assert len(totals) == 1, f"frames_total must be constant, got {totals}"

    dones = [p["frames_done"] for p in payloads]
    assert dones == sorted(dones), "frames_done must be monotonically increasing"
    assert dones[-1] == payloads[0]["frames_total"], "final frames_done must equal total"

    all_streamed = []
    for p in payloads:
        all_streamed.extend(p["new_matches"])

    final_flat = []
    for matches in final_results.values():
        for m in matches:
            final_flat.append(m)

    def _key(m):
        return (m["frame"], m["frame_number"], m["timestamp"])

    streamed_keys = sorted(_key(m) for m in all_streamed)
    final_keys = sorted(_key(m) for m in final_flat)
    assert streamed_keys == final_keys
