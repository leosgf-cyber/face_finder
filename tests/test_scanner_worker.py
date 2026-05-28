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
