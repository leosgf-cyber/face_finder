# Parallel Face Scanning — Design

**Date:** 2026-05-28
**Status:** Approved, ready for implementation planning
**Scope:** `scanner.py` (scan_frames + load_references), `web/app.py`, `web/static/js/app.js`, `web/templates/index.html`

## Goal

Use all CPU cores during face scanning. Today `scan_frames` runs single-threaded, so on an M-series Mac with 8+ cores the scan is at most ~12% of the machine's CPU capacity. Target: process frames in parallel across N worker processes with first-class live progress reporting (frames done + matches streaming into the UI as they're found).

Reference loading is parallelized in the same change because it shares the same worker pool and addresses the same "first-time clustering of a big folder is slow" pain.

## Non-goals

- GPU acceleration.
- Distributed / multi-machine scanning.
- Celery/Redis migration (separate backlog item — happens when this becomes SaaS).
- Smart FPS / scene-cut extraction (separate backlog item).

## Target environment

- macOS on M-series (Apple Silicon), 16GB+ unified memory.
- Python 3.12.
- `multiprocessing` `spawn` context (the only option on macOS).

## Architecture

### Worker pool

- `concurrent.futures.ProcessPoolExecutor(max_workers=N, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)`.
- Default `N = min(os.cpu_count() - 1, 8)`.
  - Capped at 8: M-series has 4–8 performance cores; spilling into efficiency cores tends to thrash more than it speeds up.
  - Override with env var `FACE_FINDER_WORKERS` (integer).
- `_worker_init` loads `detector`, `shape_predictor`, `face_encoder` into module-level globals in the worker process. Loaded once per worker, reused for every frame that worker handles. Each worker holds ~100MB of model data; 8 workers ≈ 800MB total — fine on 16GB+ Macs.

### Work unit

- Main process runs the existing frame-similarity dedup pass sequentially (it's cheap — only thumbnail comparisons) and produces `to_process: list[(frame_index, frame_path)]`.
- Each surviving frame is one work item, submitted via `executor.submit(_worker_process_frame, frame_index, frame_path)`.

### Worker function — `_worker_process_frame`

Pure CPU-bound. Takes `(frame_index, frame_path)`. Returns:

```python
(frame_index: int,
 frame_path: str,
 faces: list[tuple[np.ndarray, bytes]])  # (128-d encoding, JPEG-encoded face crop)
```

Steps:
1. Read image with `cv2.imread`.
2. Convert to RGB and run the existing `_resize_for_detection` path.
3. For each detected face: compute encoding (mapped back to original-resolution rect, matching today's behavior), crop with `_crop_face(padding=0.5)`, JPEG-encode with `cv2.imencode(".jpg", crop)`, append `(encoding, crop_bytes)`.
4. Return the tuple.

Rationale for returning JPEG bytes inline rather than re-reading: avoids a second `imread` in the main process when a match is confirmed. Bytes per face are small (a few KB); even 5 faces × 1000 frames is ~10-20MB of IPC, negligible.

### Main-process responsibilities

- Owns `all_known_encodings`, the match counter, and all I/O for match files and results.
- Iterates `concurrent.futures.as_completed(futures)`. For each completed frame:
  1. For each `(encoding, crop_bytes)` in the result:
     - Compute distances to `all_known_encodings`, find best index.
     - If `distances[best] <= tolerance`: assign sequential `match_counter`, write `crop_bytes` to `matches_dir / f"match_{n:04d}.jpg"`, append entry to `results[name]`.
  2. Update progress state (see below).
- After the loop: re-sort each `results[name]` list by `frame_number` so JSON output is deterministic regardless of which order workers finished.

### Small-scan fast path

If `len(to_process) < 30`, skip the pool entirely and run the existing single-process loop. Threshold chosen to keep short scans (a few seconds today) snappy — spawn cost of N workers (~1-3s × N) would otherwise make small scans slower than they are now.

The constant `PARALLEL_THRESHOLD = 30` lives in `scanner.py`. Not user-configurable; the value is a property of the hardware, not the workload.

### Progress + live matches

`scan_frames` gains an optional parameter:

```python
progress_callback: Callable[[dict], None] | None = None
```

The callback is invoked from the main process (no concurrency on the callback side). Payload shape:

```python
{
    "frames_done": int,
    "frames_total": int,
    "new_matches": [match_entry, ...],  # entries discovered since the last callback
}
```

Throttling:
- Always call when at least one new match was found in the most recently completed frame.
- Otherwise, call every 10 frames.
- Always call on the final frame.

`web/app.py` provides a callback that mutates `jobs[job_id]`:
- Sets `jobs[job_id]["frames_done"]` and `jobs[job_id]["frames_total"]`.
- Appends `new_matches` to `jobs[job_id]["live_matches"]` (a list, append-only for the lifetime of the job).
- Updates `jobs[job_id]["partial_matches"]` count.

Frontend (existing `/api/job/<id>` polling loop) reads the new fields:
- Renders a "frame X / Y" indicator while a video is being scanned.
- Maintains a `last_seen_match_index` cursor; on each poll, renders `live_matches[last_seen_match_index:]` and advances the cursor. No full re-render.

### Reference loading parallelization

`load_references` is refactored to use a `ProcessPoolExecutor` (a separate instance from the scan-time one — created and torn down within `load_references`) when there are 10+ uncached reference images. Cache hit path is unchanged (no pool spun up).

- Each worker receives `(img_path, name)`, runs the same encoding logic as today's `_load_face`, returns `(name, encoding | None)`.
- Main process aggregates results into the `people` dict in the **same order** as `_get_ref_files` returned them, then writes the existing pickle cache.
- The threshold (10) is a constant in `scanner.py`. Smaller folders use the existing sequential path because spawn cost dominates.

If `FACE_FINDER_PARALLEL=0`, sequential loader is used unconditionally.

## Configuration

| Setting | Default | Source | Purpose |
|---|---|---|---|
| Worker count | `min(os.cpu_count() - 1, 8)` | `FACE_FINDER_WORKERS` env var | Override pool size |
| Parallel enabled | `1` | `FACE_FINDER_PARALLEL` env var | Feature flag; `0` reverts to single-process for the whole module |
| Small-scan threshold | `30` frames | constant in `scanner.py` | Don't spawn workers for tiny scans |
| Reference parallel threshold | `10` files | constant in `scanner.py` | Don't spawn workers for tiny ref folders |

## Error handling

- **Worker exception on a single frame:** caught in main via `future.exception()`. Logged with frame path. Frame skipped. Matches today's behavior where unreadable frames are skipped silently.
- **Pool fails to start** (e.g., resource limits, fork-safety regression on a future macOS): logged warning, fall back to single-process path for that scan. The product still works.
- **KeyboardInterrupt:** `with ProcessPoolExecutor(...) as executor:` block ensures workers are torn down. Job state is left in whatever partial state the most recent callback set.
- **Reference encoding errors:** caught per worker result; image is skipped with a warning, same as today's sequential path.

## Rollout

- Feature flag `FACE_FINDER_PARALLEL` (default `1`) wraps the entire new code path in both `scan_frames` and `load_references`. Setting `0` runs the existing single-process code.
- Flag stays for one development cycle so it's possible to roll back without a code change if a regression appears on real wedding-length jobs.
- Flag removed (and the single-process fallback paths for non-small scans deleted) once a real scan has run end-to-end successfully.
- The small-scan fast paths (under 30 frames / 10 reference files) remain even after the flag is removed — they're an intentional optimization, not a fallback.

## Testing

- **Unit:** `_worker_process_frame` tested in isolation against a fixture image with a known face — assert it returns one encoding ndarray of shape `(128,)` and non-empty JPEG bytes.
- **Integration:** small fixture video (a few seconds, 2-3 reference faces). Run `scan_frames` once with `FACE_FINDER_PARALLEL=1` and once with `FACE_FINDER_PARALLEL=0`. Assert the two `results` dicts are equal after sorting each name's matches by `frame_number`. This catches any drift in encoding or thresholding between paths.
- **Reference loading:** same equality test for `load_references` against a folder with 15+ images.
- **Progress callback:** a stub callback records every payload; assert `frames_total` is constant, `frames_done` is monotonic and reaches `frames_total`, and that the union of all `new_matches` equals the final `results`.
- **Benchmark script:** not a test, but a one-shot CLI that runs both paths against the same fixture and prints elapsed wall time and frames/sec. Lives under `scripts/` or as a `--benchmark` flag on `main.py`.

## Files touched

| File | Change |
|---|---|
| `scanner.py` | New `_worker_init`, `_worker_process_frame`, refactor of `scan_frames` and `load_references`. Remove unused `_detect_and_encode` (lines 206-234) — it was scaffolding for exactly this work and is now superseded. |
| `web/app.py` | Wire `progress_callback` into the scan job. New job-state fields: `frames_done`, `frames_total`, `live_matches`. |
| `web/static/js/app.js` | Read new fields. Maintain `last_seen_match_index` cursor and render incremental matches. Render `frames_done/frames_total` while a video is in progress. |
| `web/templates/index.html` | Frame-counter element in the progress section. |

No new dependencies. `concurrent.futures` and `multiprocessing` are stdlib.

## Open questions

None at design-approval time. Anything that emerges during implementation goes to the plan, not back to this spec.
