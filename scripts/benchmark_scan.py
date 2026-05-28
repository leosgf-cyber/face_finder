#!/usr/bin/env python3
"""Benchmark scan_frames in parallel vs single-process modes.

Usage:
    python scripts/benchmark_scan.py <frames_dir> <references_dir>

Prints wall time and frames/sec for each mode.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner import load_references, scan_frames  # noqa: E402


def run(label, frames_dir, refs, env_value, matches_dir):
    os.environ["FACE_FINDER_PARALLEL"] = env_value
    matches_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    results = scan_frames(
        str(frames_dir), refs, tolerance=0.6, fps=1.0, matches_dir=str(matches_dir)
    )
    elapsed = time.perf_counter() - start
    frame_count = len(list(Path(frames_dir).glob("frame_*.jpg")))
    fps_rate = frame_count / elapsed if elapsed else 0
    total_matches = sum(len(v) for v in results.values())
    print(
        f"{label:>14}: {elapsed:7.2f}s | {fps_rate:6.2f} fps | "
        f"{total_matches} match(es) across {len(results)} person(s)"
    )


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    frames_dir = Path(sys.argv[1])
    refs_dir = Path(sys.argv[2])

    refs = load_references(str(refs_dir))

    out_root = Path("results/_benchmark")
    run("sequential", frames_dir, refs, "0", out_root / "seq")
    run("parallel", frames_dir, refs, "1", out_root / "par")


if __name__ == "__main__":
    main()
