# Parallel Face Scanning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize `scan_frames` and `load_references` across CPU cores using `ProcessPoolExecutor`, stream live progress + matches to the UI during each scan, behind a one-cycle feature flag.

**Architecture:** Workers are persistent within a single scan — initialized once with dlib models, then reused for every frame they handle. Workers do read → detect → encode → crop and return `(encoding, JPEG-encoded crop bytes)`. The main process owns matching, match-file numbering, results dict, and the progress callback. A small-scan fast path keeps short scans (<30 frames) on the existing single-process loop so spawn cost never makes anything slower.

**Tech Stack:** Python 3.12, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing` (spawn context, mandatory on macOS), Flask, dlib, OpenCV, vanilla JS. No new runtime dependencies; pytest is added as dev-only.

**Reference docs:** [Design spec](../specs/2026-05-28-parallel-face-scanning-design.md)

---

## File Map

**Create:**
- `requirements-dev.txt` — pytest + dev deps
- `tests/__init__.py` — empty marker
- `tests/conftest.py` — pytest fixtures (synthetic single-face image, refs dir, frames dir)
- `tests/test_scanner_worker.py` — unit tests for `_worker_init` + `_worker_process_frame`
- `tests/test_scanner_parallel.py` — equivalence + progress-callback tests
- `tests/test_load_references_parallel.py` — equivalence test for parallel reference loading
- `tests/fixtures/.gitkeep` — keep dir, fixture image generated at test runtime
- `scripts/benchmark_scan.py` — wall-clock + frames/sec for both paths

**Modify:**
- `scanner.py` — new `_worker_init`, `_worker_process_frame`; refactor `scan_frames` and `load_references`; remove dead `_detect_and_encode`
- `web/app.py` — add progress callback in `_process_job`; new job-state fields `frames_done`, `frames_total`, `live_matches`
- `web/static/js/app.js` — render frame counter + incremental live matches
- `web/templates/index.html` — add frame counter element
- `requirements.txt` — no changes (kept stdlib-only)
- `.gitignore` — ignore `tests/fixtures/*.jpg` and `tests/fixtures/*.mp4` (allow `.gitkeep`)

---

## Task 1: Test infrastructure setup

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1.1: Add dev requirements file**

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest>=8.0.0
pytest-timeout>=2.2.0
```

- [ ] **Step 1.2: Update .gitignore**

Append to `.gitignore`:

```
# Test fixtures (generated locally, not committed)
tests/fixtures/*.jpg
tests/fixtures/*.png
tests/fixtures/*.mp4
!tests/fixtures/.gitkeep
```

- [ ] **Step 1.3: Create test package marker**

Create `tests/__init__.py` (empty file).

Create `tests/fixtures/.gitkeep` (empty file).

- [ ] **Step 1.4: Create conftest with synthetic-image fixture**

Create `tests/conftest.py`:

```python
"""Pytest fixtures for face_finder tests.

We avoid shipping real face images. Instead, we use the dlib example faces
which the repo already downloads on first run. The fixture image is generated
once per test session by running detection on a synthetic image with a
detected face from dlib's own pipeline.

If dlib detection can't find a face in the synthetic image, the fixture
falls back to downloading a small CC0 face image.
"""

from pathlib import Path
import shutil
import urllib.request

import cv2
import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FACE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
)


@pytest.fixture(scope="session")
def sample_face_path():
    """Path to a single committed-or-downloaded face image with one face.

    Downloaded on first run. ~70KB, CC-licensed via OpenCV samples.
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    target = FIXTURES_DIR / "sample_face.jpg"
    if not target.exists():
        urllib.request.urlretrieve(SAMPLE_FACE_URL, str(target))
    return target


@pytest.fixture
def frames_dir(tmp_path, sample_face_path):
    """A directory of N synthetic 'frames' — all copies of the sample face.

    Default N=8. Override by passing `n` as a parameter via indirect parametrize.
    """
    fdir = tmp_path / "frames"
    fdir.mkdir()
    for i in range(8):
        shutil.copy(sample_face_path, fdir / f"frame_{i:04d}.jpg")
    return fdir


@pytest.fixture
def references_dir(tmp_path, sample_face_path):
    """A references directory with one person ('alice') and one image."""
    rdir = tmp_path / "refs"
    person = rdir / "alice"
    person.mkdir(parents=True)
    shutil.copy(sample_face_path, person / "alice_1.jpg")
    return rdir


@pytest.fixture
def matches_dir(tmp_path):
    mdir = tmp_path / "matches"
    mdir.mkdir()
    return mdir
```

- [ ] **Step 1.5: Verify pytest discovers the test directory**

Run:
```bash
source venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ --collect-only
```

Expected: "no tests ran" (no test files yet) but no import errors. If pytest reports collection errors, fix imports before continuing.

- [ ] **Step 1.6: Commit**

```bash
git add requirements-dev.txt tests/__init__.py tests/conftest.py tests/fixtures/.gitkeep .gitignore
git commit -m "test: scaffold pytest with face-image fixtures"
```

---

## Task 2: Worker initializer + worker function (TDD)

**Files:**
- Modify: `scanner.py` (add `_worker_init` and `_worker_process_frame`)
- Create: `tests/test_scanner_worker.py`

- [ ] **Step 2.1: Write failing test for `_worker_process_frame`**

Create `tests/test_scanner_worker.py`:

```python
"""Tests for the per-frame worker function used by ProcessPoolExecutor."""

import numpy as np

from scanner import _worker_init, _worker_process_frame


def test_worker_init_populates_globals():
    """After init, the worker module has detector/predictor/encoder loaded."""
    import scanner

    scanner._WORKER_DETECTOR = None
    scanner._WORKER_SHAPE_PREDICTOR = None
    scanner._WORKER_FACE_ENCODER = None

    _worker_init()

    assert scanner._WORKER_DETECTOR is not None
    assert scanner._WORKER_SHAPE_PREDICTOR is not None
    assert scanner._WORKER_FACE_ENCODER is not None


def test_worker_process_frame_returns_encoding_and_crop(sample_face_path):
    """Process one image; expect at least one face with 128-d encoding + JPEG bytes."""
    _worker_init()

    frame_index, frame_path, faces = _worker_process_frame(
        42, str(sample_face_path)
    )

    assert frame_index == 42
    assert frame_path == str(sample_face_path)
    assert len(faces) >= 1

    encoding, crop_bytes = faces[0]
    assert isinstance(encoding, np.ndarray)
    assert encoding.shape == (128,)
    assert isinstance(crop_bytes, bytes)
    assert len(crop_bytes) > 100
    # JPEG magic bytes
    assert crop_bytes[:3] == b"\xff\xd8\xff"


def test_worker_process_frame_missing_file_returns_empty(tmp_path):
    """A missing/unreadable frame returns empty faces list, not a crash."""
    _worker_init()

    missing = tmp_path / "does_not_exist.jpg"
    frame_index, frame_path, faces = _worker_process_frame(0, str(missing))

    assert frame_index == 0
    assert faces == []
```

- [ ] **Step 2.2: Run test, verify it fails**

Run:
```bash
pytest tests/test_scanner_worker.py -v
```

Expected: FAIL with `ImportError` or `AttributeError` — neither `_worker_init` nor `_worker_process_frame` exist yet.

- [ ] **Step 2.3: Implement `_worker_init` and `_worker_process_frame`**

In `scanner.py`, add **after** the existing `_get_detector_and_encoder` function (around line 45):

```python
# Worker-process globals — populated by _worker_init() exactly once per worker.
# Process boundary ensures no contention with the main process or other workers.
_WORKER_DETECTOR = None
_WORKER_SHAPE_PREDICTOR = None
_WORKER_FACE_ENCODER = None


def _worker_init():
    """Initializer for ProcessPoolExecutor workers.

    Loads dlib models once per worker process into module globals so subsequent
    frame processing reuses them. Each worker pays ~100MB resident model cost.
    """
    global _WORKER_DETECTOR, _WORKER_SHAPE_PREDICTOR, _WORKER_FACE_ENCODER
    _ensure_models()
    _WORKER_DETECTOR = dlib.get_frontal_face_detector()
    _WORKER_SHAPE_PREDICTOR = dlib.shape_predictor(str(SHAPE_PREDICTOR))
    _WORKER_FACE_ENCODER = dlib.face_recognition_model_v1(str(FACE_REC_MODEL))


def _worker_process_frame(frame_index: int, frame_path: str):
    """Read a frame, detect+encode faces, return (idx, path, [(encoding, jpeg_bytes)]).

    Pure CPU-bound. Called from worker processes. The main process compares
    encodings against the known references — workers don't know about references.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return frame_index, frame_path, []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _WORKER_DETECTOR(small_rgb, 1)

    faces = []
    for face in detected:
        if scale != 1.0:
            orig_face = dlib.rectangle(
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )
            shape = _WORKER_SHAPE_PREDICTOR(rgb, orig_face)
            encoding = np.array(
                _WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape)
            )
            crop = _crop_face(img, orig_face, padding=0.5)
        else:
            shape = _WORKER_SHAPE_PREDICTOR(rgb, face)
            encoding = np.array(
                _WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape)
            )
            crop = _crop_face(img, face, padding=0.5)

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            continue
        faces.append((encoding, buf.tobytes()))

    return frame_index, frame_path, faces
```

- [ ] **Step 2.4: Run test, verify it passes**

Run:
```bash
pytest tests/test_scanner_worker.py -v
```

Expected: all 3 tests PASS. First run may take ~30s if dlib models need to download.

- [ ] **Step 2.5: Commit**

```bash
git add scanner.py tests/test_scanner_worker.py
git commit -m "feat(scanner): add per-frame worker function and pool initializer"
```

---

## Task 3: Refactor `scan_frames` with ProcessPoolExecutor (TDD)

**Files:**
- Modify: `scanner.py` (refactor `scan_frames`)
- Create: `tests/test_scanner_parallel.py`

- [ ] **Step 3.1: Write failing equivalence test**

Create `tests/test_scanner_parallel.py`:

```python
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

    # Two scans must write match files into separate dirs.
    par_matches = matches_dir / "parallel"
    seq_matches = matches_dir / "sequential"
    par_matches.mkdir()
    seq_matches.mkdir()

    monkeypatch.setenv("FACE_FINDER_PARALLEL", "1")
    parallel = scan_frames(
        str(frames_dir), refs, tolerance=0.6, fps=1.0, matches_dir=str(par_matches)
    )

    monkeypatch.setenv("FACE_FINDER_PARALLEL", "0")
    sequential = scan_frames(
        str(frames_dir), refs, tolerance=0.6, fps=1.0, matches_dir=str(seq_matches)
    )

    # Match filenames differ (par_NNNN vs seq_NNNN), so strip that field before compare.
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
    # Only 3 frames, well below the threshold of 30.
    fdir = tmp_path / "small"
    fdir.mkdir()
    for i in range(3):
        shutil.copy(sample_face_path, fdir / f"frame_{i:04d}.jpg")

    refs = load_references(str(references_dir), tolerance=0.6)

    # We can't easily probe internals; just assert it completes and returns results.
    results = scan_frames(
        str(fdir), refs, tolerance=0.6, fps=1.0, matches_dir=str(matches_dir)
    )
    assert "alice" in results
    assert len(results["alice"]) >= 1
```

- [ ] **Step 3.2: Run test, verify it fails or errors out**

Run:
```bash
pytest tests/test_scanner_parallel.py::test_scan_frames_parallel_equals_sequential -v
```

Expected: FAIL — current `scan_frames` ignores `FACE_FINDER_PARALLEL`. Results from both calls will differ only in `match_image` filenames (which we strip), but since the current implementation already produces consistent output, the test may actually pass. To be sure the failure is meaningful, **before implementing**, manually verify the test is exercising the right branch by checking that `FACE_FINDER_PARALLEL=1` doesn't currently take a different code path. If it passes spuriously here, that's OK — Task 3 still proceeds to add the parallel path, and the test serves as a regression guard.

- [ ] **Step 3.3: Refactor `scan_frames` with the parallel path**

In `scanner.py`, near the top of the module (after the model URL constants, ~line 18), add:

```python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Below this many to-process frames, skip the pool — spawn cost would dominate.
PARALLEL_THRESHOLD = 30
```

Replace the body of `scan_frames` (currently lines 237-343) with the following. Keep the function signature compatible but add a new optional `progress_callback` parameter:

```python
def scan_frames(
    frames_dir: str,
    references: dict,
    tolerance: float = 0.6,
    fps: float = 1.0,
    matches_dir: str | None = None,
    progress_callback=None,
) -> dict:
    """Scan a directory of extracted frames for known faces.

    Returns: {name: [match_entry, ...]} keyed by reference person name.

    progress_callback: optional callable invoked from the main thread with
        {"frames_done": int, "frames_total": int, "new_matches": [entry, ...]}
        whenever a frame completes that produced matches, plus every 10 frames
        otherwise, plus on the final frame.
    """
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.jpg"))

    if not frame_files:
        print(f"Nenhum frame encontrado em '{frames_dir}'")
        return {}

    if matches_dir:
        Path(matches_dir).mkdir(parents=True, exist_ok=True)

    all_known_encodings = []
    all_known_names = []
    for name, encodings in references.items():
        for enc in encodings:
            all_known_encodings.append(enc)
            all_known_names.append(name)
    all_known_encodings = np.array(all_known_encodings)

    # Sequential dedup pass — cheap, runs on main.
    prev_frame = None
    skipped = 0
    to_process = []
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(str(frame_file))
        if img is None:
            continue
        if _frames_are_similar(prev_frame, img):
            skipped += 1
            prev_frame = img
            continue
        prev_frame = img
        to_process.append((i, frame_file))

    print(
        f"\nVarrendo {len(to_process)} frames "
        f"({skipped} similares pulados de {len(frame_files)} total)..."
    )

    results = {name: [] for name in references}
    match_counter = [0]  # mutable so the helper can update it

    def _consume_face_result(frame_index, frame_file_path, faces, img_for_crop=None):
        """Compare each detected face against references, record matches.

        Returns the list of match entries discovered in this frame (for the callback).
        img_for_crop is provided only by the single-process fast path; the parallel
        path passes JPEG bytes inline so we never need to re-read the file.
        """
        new_matches = []
        for face_data in faces:
            if isinstance(face_data, tuple) and len(face_data) == 2:
                # Parallel path: (encoding_ndarray, jpeg_bytes)
                encoding, crop_bytes = face_data
                face_rect_for_seq = None
            else:
                # Single-process path: (encoding, face_rect)
                encoding, face_rect_for_seq = face_data
                crop_bytes = None

            distances = np.linalg.norm(all_known_encodings - encoding, axis=1)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] > tolerance:
                continue

            matched_name = all_known_names[best_idx]
            frame_number = frame_index + 1
            timestamp_seconds = frame_number / fps

            entry = {
                "frame": Path(frame_file_path).name,
                "frame_number": frame_number,
                "timestamp": format_timestamp(timestamp_seconds),
                "confidence": round(1 - float(distances[best_idx]), 3),
            }

            if matches_dir:
                match_filename = f"match_{match_counter[0]:04d}.jpg"
                match_path = Path(matches_dir) / match_filename
                if crop_bytes is not None:
                    match_path.write_bytes(crop_bytes)
                else:
                    crop = _crop_face(img_for_crop, face_rect_for_seq, padding=0.5)
                    cv2.imwrite(str(match_path), crop)
                entry["match_image"] = match_filename
                match_counter[0] += 1

            if entry not in results[matched_name]:
                results[matched_name].append(entry)
                new_matches.append(entry)
        return new_matches

    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(to_process) >= PARALLEL_THRESHOLD

    if use_pool:
        max_workers = int(
            os.environ.get(
                "FACE_FINDER_WORKERS", min((os.cpu_count() or 2) - 1, 8)
            )
        )
        max_workers = max(1, max_workers)

        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as executor:
                futures = {
                    executor.submit(_worker_process_frame, i, str(frame_file)): i
                    for i, frame_file in to_process
                }
                frames_done = 0
                frames_since_last_callback = 0
                for future in as_completed(futures):
                    frames_done += 1
                    frames_since_last_callback += 1
                    try:
                        frame_index, frame_file_path, faces = future.result()
                    except Exception as e:
                        print(f"  Aviso: worker falhou em frame: {e}")
                        continue

                    new_matches = _consume_face_result(
                        frame_index, frame_file_path, faces
                    )

                    should_emit = (
                        progress_callback is not None
                        and (
                            bool(new_matches)
                            or frames_since_last_callback >= 10
                            or frames_done == len(to_process)
                        )
                    )
                    if should_emit:
                        progress_callback(
                            {
                                "frames_done": frames_done,
                                "frames_total": len(to_process),
                                "new_matches": new_matches,
                            }
                        )
                        frames_since_last_callback = 0
        except Exception as e:
            print(f"  Aviso: pool falhou ({e}); caindo pra single-process")
            use_pool = False

    if not use_pool:
        detector, shape_predictor, face_encoder = _get_detector_and_encoder()
        frames_done = 0
        frames_since_last_callback = 0
        for idx, (i, frame_file) in enumerate(to_process):
            frames_done += 1
            frames_since_last_callback += 1
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  Processando {idx + 1}/{len(to_process)}...")

            img = cv2.imread(str(frame_file))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            small_rgb, scale = _resize_for_detection(rgb)
            detected = detector(small_rgb, 1)
            face_data_list = []
            for face in detected:
                if scale != 1.0:
                    orig_face = dlib.rectangle(
                        int(face.left() / scale),
                        int(face.top() / scale),
                        int(face.right() / scale),
                        int(face.bottom() / scale),
                    )
                    shape = shape_predictor(rgb, orig_face)
                    encoding = np.array(
                        face_encoder.compute_face_descriptor(rgb, shape)
                    )
                    face_data_list.append((encoding, orig_face))
                else:
                    shape = shape_predictor(rgb, face)
                    encoding = np.array(
                        face_encoder.compute_face_descriptor(rgb, shape)
                    )
                    face_data_list.append((encoding, face))

            new_matches = _consume_face_result(
                i, str(frame_file), face_data_list, img_for_crop=img
            )

            should_emit = (
                progress_callback is not None
                and (
                    bool(new_matches)
                    or frames_since_last_callback >= 10
                    or frames_done == len(to_process)
                )
            )
            if should_emit:
                progress_callback(
                    {
                        "frames_done": frames_done,
                        "frames_total": len(to_process),
                        "new_matches": new_matches,
                    }
                )
                frames_since_last_callback = 0

    # Determinism: sort each name's matches by frame_number (parallel completion order
    # is non-deterministic).
    for name in list(results.keys()):
        results[name].sort(key=lambda m: m["frame_number"])

    results = {name: matches for name, matches in results.items() if matches}
    return results
```

- [ ] **Step 3.4: Run all scanner tests**

Run:
```bash
pytest tests/test_scanner_worker.py tests/test_scanner_parallel.py -v
```

Expected: all PASS. The equivalence test confirms parallel and sequential paths produce identical results (modulo match filenames, which we strip).

- [ ] **Step 3.5: Commit**

```bash
git add scanner.py tests/test_scanner_parallel.py
git commit -m "feat(scanner): parallelize scan_frames via ProcessPoolExecutor"
```

---

## Task 4: Progress callback contract test

**Files:**
- Modify: `tests/test_scanner_parallel.py` (add callback test)

- [ ] **Step 4.1: Append callback test**

Append to `tests/test_scanner_parallel.py`:

```python
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

    # All streamed matches should appear in final_results, and vice versa.
    final_flat = []
    for matches in final_results.values():
        for m in matches:
            final_flat.append(m)

    # Compare ignoring match_image filename (assigned during streaming order).
    def _key(m):
        return (m["frame"], m["frame_number"], m["timestamp"])

    streamed_keys = sorted(_key(m) for m in all_streamed)
    final_keys = sorted(_key(m) for m in final_flat)
    assert streamed_keys == final_keys
```

- [ ] **Step 4.2: Run the new test**

Run:
```bash
pytest tests/test_scanner_parallel.py::test_progress_callback_contract -v
```

Expected: PASS.

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_scanner_parallel.py
git commit -m "test(scanner): verify progress callback contract"
```

---

## Task 5: Wire progress callback into `web/app.py`

**Files:**
- Modify: `web/app.py:603-650` (job init + _process_job)

- [ ] **Step 5.1: Update job initialization to include new fields**

In `web/app.py`, find the job init block (around line 603) and replace it with:

```python
    jobs[job_id] = {
        "status": "processing",
        "progress": "Iniciando...",
        "videos": video_names,
        "results": None,
        "partial_matches": 0,
        "frames_done": 0,
        "frames_total": 0,
        "live_matches": [],
    }
```

- [ ] **Step 5.2: Wire callback into `_process_job`**

In `web/app.py`, find the `scan_frames` call (around line 637) and replace the scan loop with this. The full replacement block starts at `for idx, (video_path, video_name) in enumerate(...)` and ends just before `result_path = RESULTS_DIR / f"{job_id}.json"`:

```python
        for idx, (video_path, video_name) in enumerate(zip(video_paths, video_names)):
            jobs[job_id]["progress"] = (
                f"Extraindo frames: {video_name} ({idx + 1}/{len(video_paths)})..."
            )
            frames_dir = str(RESULTS_DIR / f"frames_{job_id}_{idx}")
            extract_frames(video_path, frames_dir, fps, start, end)

            jobs[job_id]["progress"] = (
                f"Varrendo: {video_name} ({idx + 1}/{len(video_paths)})..."
            )
            # Reset per-video frame counter; live_matches accumulates across videos.
            jobs[job_id]["frames_done"] = 0
            jobs[job_id]["frames_total"] = 0

            def _on_progress(payload, _video_name=video_name):
                jobs[job_id]["frames_done"] = payload["frames_done"]
                jobs[job_id]["frames_total"] = payload["frames_total"]
                for m in payload["new_matches"]:
                    m_with_video = dict(m)
                    m_with_video["video"] = _video_name
                    jobs[job_id]["live_matches"].append(m_with_video)
                jobs[job_id]["partial_matches"] = len(jobs[job_id]["live_matches"])

            results = scan_frames(
                frames_dir,
                refs,
                tolerance,
                fps,
                matches_dir,
                progress_callback=_on_progress,
            )

            for name, matches in results.items():
                for m in matches:
                    m["video"] = video_name
                if name not in all_results:
                    all_results[name] = []
                all_results[name].extend(matches)

            total_matches = sum(len(v) for v in all_results.values())
            people_found = len(all_results)
            jobs[job_id]["partial_matches"] = total_matches
            jobs[job_id]["progress"] = (
                f"Video {idx + 1}/{len(video_paths)} concluido | "
                f"{people_found} pessoa(s), {total_matches} match(es)"
            )

            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass
```

- [ ] **Step 5.3: Restart dev server and sanity-check the API**

The dev server should auto-reload from the `.py` change. Manually verify by hitting the jobs endpoint after kicking off a small scan via the UI:

```bash
curl -s http://localhost:8080/api/jobs/<some-recent-job-id> | python3 -m json.tool | grep -E "frames_done|frames_total|live_matches" | head -5
```

Expected: the new fields are present in the JSON response (possibly with default values if no scan is active).

- [ ] **Step 5.4: Commit**

```bash
git add web/app.py
git commit -m "feat(web): wire scan progress callback and live_matches into job state"
```

---

## Task 6: Frontend — frame counter + live matches

**Files:**
- Modify: `web/templates/index.html`
- Modify: `web/static/js/app.js`

- [ ] **Step 6.1: Add frame counter element to the progress section**

In `web/templates/index.html`, find the element with `id="progressText"` (or the progress display block). Immediately after it, add:

```html
<div id="frameCounter" class="scan-frame-counter" style="display: none; color: var(--text-dim, #888); font-size: 13px; margin-top: 4px;"></div>
<div id="liveMatches" class="scan-live-matches" style="margin-top: 8px;"></div>
```

(If you can't find a `--text-dim` CSS variable, drop the inline color — it'll inherit.)

- [ ] **Step 6.2: Add live-match rendering to `app.js`**

In `web/static/js/app.js`, find the polling block around line 794 (the `fetch("/api/jobs/" + jobId)` call). Replace the inner progress-update logic so it reads:

```javascript
    const res = await fetch("/api/jobs/" + jobId);
    const job = await res.json();
    var progressMsg = job.progress;
    if (job.partial_matches > 0 && job.status === "processing") {
      progressMsg += " | " + job.partial_matches + " match(es) ate agora";
    }
    document.getElementById("progressText").textContent = progressMsg;

    // Frame counter (per-video)
    const frameCounter = document.getElementById("frameCounter");
    if (frameCounter) {
      if (job.frames_total && job.status === "processing") {
        frameCounter.style.display = "block";
        frameCounter.textContent = "Frame " + job.frames_done + " / " + job.frames_total;
      } else {
        frameCounter.style.display = "none";
      }
    }

    // Incremental live matches
    const liveBox = document.getElementById("liveMatches");
    if (liveBox && Array.isArray(job.live_matches)) {
      if (typeof window.__lastSeenMatchIndex === "undefined") {
        window.__lastSeenMatchIndex = 0;
      }
      const slice = job.live_matches.slice(window.__lastSeenMatchIndex);
      for (const m of slice) {
        const row = document.createElement("div");
        row.className = "live-match-row";
        row.style.cssText = "padding: 4px 0; font-size: 13px; color: #ccc;";
        row.textContent =
          (m.video || "") + " · " + (m.timestamp || "") + " · " + (m.frame || "");
        liveBox.appendChild(row);
      }
      window.__lastSeenMatchIndex = job.live_matches.length;
    }
```

- [ ] **Step 6.3: Reset live-match cursor when a new scan starts**

In `app.js`, find where a new scan is kicked off (the function that calls `/api/scan` and gets back a `job_id`). Just before the polling loop starts, add:

```javascript
    window.__lastSeenMatchIndex = 0;
    const liveBox = document.getElementById("liveMatches");
    if (liveBox) liveBox.innerHTML = "";
```

(If the file already has an obvious "starting a new scan" function, place these two lines there. Otherwise add them at the top of the polling function before the first `await fetch`.)

- [ ] **Step 6.4: Manual browser test**

1. Run a real small scan via the UI (a few reference faces, a short video).
2. Verify: frame counter ticks up while the scan runs.
3. Verify: live matches appear in `#liveMatches` as they're found, not all at once at the end.
4. Verify: starting a second scan clears the previous live matches.

- [ ] **Step 6.5: Commit**

```bash
git add web/templates/index.html web/static/js/app.js
git commit -m "feat(web): live frame counter and incremental match rendering"
```

---

## Task 7: Parallel reference loading (TDD)

**Files:**
- Modify: `scanner.py` (refactor `load_references`)
- Create: `tests/test_load_references_parallel.py`

- [ ] **Step 7.1: Write failing equivalence test**

Create `tests/test_load_references_parallel.py`:

```python
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
```

- [ ] **Step 7.2: Run test, verify it fails**

Run:
```bash
pytest tests/test_load_references_parallel.py -v
```

Expected: PASS on first run only if the sequential and parallel paths are already equivalent (the parallel path doesn't exist yet, so both runs use the same code — this test passes trivially for now). Run it after Step 7.3 to confirm the parallel path still produces identical encodings.

- [ ] **Step 7.3: Refactor `load_references` with parallel path**

Near the top of `scanner.py`, add:

```python
REFERENCE_PARALLEL_THRESHOLD = 10
```

Add a helper for worker-side reference encoding (place it next to `_worker_process_frame`):

```python
def _worker_process_reference(args):
    """Process a single reference image. Worker-side helper.

    args: (img_path_str, name)
    returns: (name, encoding_or_None, img_path_str)
    """
    img_path_str, name = args
    img = cv2.imread(img_path_str)
    if img is None:
        return name, None, img_path_str

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _WORKER_DETECTOR(small_rgb, 1)
    if not detected:
        return name, None, img_path_str

    face = detected[0]
    if scale != 1.0:
        orig_face = dlib.rectangle(
            int(face.left() / scale),
            int(face.top() / scale),
            int(face.right() / scale),
            int(face.bottom() / scale),
        )
        shape = _WORKER_SHAPE_PREDICTOR(rgb, orig_face)
    else:
        shape = _WORKER_SHAPE_PREDICTOR(rgb, face)
    encoding = np.array(_WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape))
    return name, encoding, img_path_str
```

Refactor the body of `load_references` (currently lines 114-173). Replace the function with:

```python
def load_references(references_dir: str, tolerance: float = 0.6) -> dict:
    ref_dir = Path(references_dir)
    if not ref_dir.exists():
        print(f"Erro: pasta de referências não encontrada: {references_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    cache_path = ref_dir / ".encodings_cache.pkl"
    ref_files = _get_ref_files(ref_dir, extensions)

    if _cache_is_valid(cache_path, ref_files):
        print("Carregando referências do cache...")
        with open(cache_path, "rb") as f:
            people = pickle.load(f)
        print(f"Referências carregadas (cache): {len(people)} pessoa(s)")
        for name, encs in people.items():
            print(f"  - {name}: {len(encs)} foto(s)")
        return people

    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
    uses_folders = len(subdirs) > 0

    # Build (path, name) work list, preserving order.
    tasks = []
    if uses_folders:
        for person_dir in sorted(subdirs):
            name = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    tasks.append((str(img_path), name))
    else:
        for img_path in sorted(ref_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                name = img_path.stem.rsplit("_", 1)[0]
                tasks.append((str(img_path), name))

    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(tasks) >= REFERENCE_PARALLEL_THRESHOLD

    people = {}

    if use_pool:
        max_workers = int(
            os.environ.get(
                "FACE_FINDER_WORKERS", min((os.cpu_count() or 2) - 1, 8)
            )
        )
        max_workers = max(1, max_workers)

        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as executor:
                # imap-style: preserve input order via list(map(...)).
                ordered_results = list(executor.map(_worker_process_reference, tasks))
        except Exception as e:
            print(f"  Aviso: pool falhou ({e}); caindo pra single-process")
            use_pool = False
        else:
            for name, encoding, img_path_str in ordered_results:
                if encoding is None:
                    print(
                        f"  Aviso: nenhum rosto encontrado em "
                        f"'{Path(img_path_str).name}', pulando."
                    )
                    continue
                people.setdefault(name, []).append(encoding)

    if not use_pool:
        detector, shape_predictor, face_encoder = _get_detector_and_encoder()
        for img_path_str, name in tasks:
            img_path = Path(img_path_str)
            _load_face(
                img_path, name, people, detector, shape_predictor, face_encoder
            )

    if not people:
        print("Erro: nenhum rosto de referência foi carregado.")
        sys.exit(1)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(people, f)
        print("Cache de referências salvo.")
    except Exception as e:
        print(f"Aviso: não foi possível salvar cache: {e}")

    print(f"Referências carregadas: {len(people)} pessoa(s)")
    for name, encs in people.items():
        print(f"  - {name}: {len(encs)} foto(s)")

    return people
```

- [ ] **Step 7.4: Run all tests**

Run:
```bash
pytest tests/ -v
```

Expected: all PASS, including the new `test_load_references_parallel_equals_sequential`.

- [ ] **Step 7.5: Commit**

```bash
git add scanner.py tests/test_load_references_parallel.py
git commit -m "feat(scanner): parallelize load_references for 10+ ref images"
```

---

## Task 8: Remove dead `_detect_and_encode`

**Files:**
- Modify: `scanner.py` (delete unused function)

- [ ] **Step 8.1: Confirm `_detect_and_encode` is unused**

Run:
```bash
grep -rn "_detect_and_encode" /Users/leosgf/face_finder/ --include="*.py"
```

Expected: only the function definition itself in `scanner.py`. If any caller exists, do NOT delete — abort this task and flag.

- [ ] **Step 8.2: Delete the function**

In `scanner.py`, remove the entire `_detect_and_encode` function (the original lines 206-234 in the pre-refactor file). After deletion, run:

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 8.3: Commit**

```bash
git add scanner.py
git commit -m "chore(scanner): remove unused _detect_and_encode scaffold"
```

---

## Task 9: Benchmark script

**Files:**
- Create: `scripts/benchmark_scan.py`

- [ ] **Step 9.1: Create the benchmark script**

Create `scripts/benchmark_scan.py`:

```python
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

# Allow running from repo root.
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
```

- [ ] **Step 9.2: Make it executable**

Run:
```bash
chmod +x scripts/benchmark_scan.py
```

- [ ] **Step 9.3: Smoke test**

Run against any existing frames directory you have lying around. If none, skip this step — the benchmark is documentation, not a regression guard.

- [ ] **Step 9.4: Commit**

```bash
git add scripts/benchmark_scan.py
git commit -m "chore: add scan benchmark script for parallel vs sequential"
```

---

## Task 10: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 10.1: Restart the dev server fresh**

```bash
pkill -f "flask\|web/app.py" 2>/dev/null
./dev.sh
```

- [ ] **Step 10.2: Run a real wedding-length scan**

Via the UI at http://localhost:8080:
1. Upload at least one full-length video (a real job from your archives).
2. Use the existing reference set (or load a folder with 10+ photos to also exercise the parallel reference loader).
3. Start the scan.

- [ ] **Step 10.3: Observe live UI**

Verify all three:
- Frame counter ticks up smoothly (not stuck on the same number for minutes).
- Match rows appear in `#liveMatches` during the scan, not only at the end.
- Final results section still shows the full per-person breakdown as before.

- [ ] **Step 10.4: Verify CPU utilization**

In a separate terminal, while the scan is running:

```bash
top -l 1 -o cpu -n 10 | head -20
```

Expected: several `python` (or `Python`) processes near the top, each consuming significant CPU. Total CPU% across them should be substantially higher than 100% (single-core ceiling).

- [ ] **Step 10.5: Verify fallback works**

Stop the dev server, set `FACE_FINDER_PARALLEL=0`, restart:

```bash
FACE_FINDER_PARALLEL=0 ./dev.sh
```

Run another small scan. Verify it still produces correct results (slower, but functional).

- [ ] **Step 10.6: No commit for this task — it's a validation gate.**

If anything in steps 10.3–10.5 fails, file a follow-up task. Do not proceed to Task 11.

---

## Task 11: Remove feature flag (after one development cycle)

**Files:**
- Modify: `scanner.py` (strip `FACE_FINDER_PARALLEL` env reads)

**Gate:** do not start this task until Task 10 has been performed against a real wedding-length job AND at least 7 days have passed since the parallel path landed (the "one development cycle" buffer).

- [ ] **Step 11.1: Remove the flag from `scan_frames`**

In `scanner.py`, find:

```python
    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(to_process) >= PARALLEL_THRESHOLD
```

Replace with:

```python
    use_pool = len(to_process) >= PARALLEL_THRESHOLD
```

- [ ] **Step 11.2: Remove the flag from `load_references`**

In `scanner.py`, find:

```python
    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(tasks) >= REFERENCE_PARALLEL_THRESHOLD
```

Replace with:

```python
    use_pool = len(tasks) >= REFERENCE_PARALLEL_THRESHOLD
```

- [ ] **Step 11.3: Update the equivalence tests**

The two tests that monkeypatch `FACE_FINDER_PARALLEL=0` (in `tests/test_scanner_parallel.py` and `tests/test_load_references_parallel.py`) need a different way to force the sequential path now that the flag is gone. Easiest: monkeypatch `PARALLEL_THRESHOLD` to a high value.

In `tests/test_scanner_parallel.py`, in `test_scan_frames_parallel_equals_sequential`, replace the two `monkeypatch.setenv(...)` lines with:

```python
    # Force parallel
    monkeypatch.setattr("scanner.PARALLEL_THRESHOLD", 1)
    parallel = scan_frames(...)

    # Force sequential by raising the threshold above the frame count
    monkeypatch.setattr("scanner.PARALLEL_THRESHOLD", 10**6)
    sequential = scan_frames(...)
```

Apply the analogous change in `tests/test_load_references_parallel.py` using `REFERENCE_PARALLEL_THRESHOLD`.

- [ ] **Step 11.4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 11.5: Commit**

```bash
git add scanner.py tests/test_scanner_parallel.py tests/test_load_references_parallel.py
git commit -m "chore: remove FACE_FINDER_PARALLEL feature flag"
```
