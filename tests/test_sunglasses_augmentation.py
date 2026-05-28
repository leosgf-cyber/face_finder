"""Tests for synthetic sunglasses augmentation on reference encodings."""

import cv2
import dlib
import numpy as np
import pytest

from scanner import (
    SHAPE_PREDICTOR,
    _augment_reference_encodings,
    _detect_faces,
    _ensure_models,
    _paint_sunglasses,
    _resize_for_detection,
    load_references,
)


def _detect_and_shape(rgb):
    """Helper: detect first face in rgb, return its shape (landmarks)."""
    _ensure_models()
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR))
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _detect_faces(small_rgb, detector)
    assert len(detected) >= 1, "fixture must contain a face"
    face = detected[0]
    if scale != 1.0:
        face = dlib.rectangle(
            int(face.left() / scale),
            int(face.top() / scale),
            int(face.right() / scale),
            int(face.bottom() / scale),
        )
    return shape_predictor(rgb, face)


def test_paint_sunglasses_darkens_eye_region(sample_face_path):
    """The painted band must make the eye region substantially darker."""
    bgr = cv2.imread(str(sample_face_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shape = _detect_and_shape(rgb)

    out = _paint_sunglasses(rgb, shape)

    # Sample inside the eye-bounding-box only (the painted region).
    eye_pts = [shape.part(i) for i in range(36, 48)]
    xs = [p.x for p in eye_pts]
    ys = [p.y for p in eye_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    orig_mean = float(rgb[min_y:max_y, min_x:max_x].mean())
    painted_mean = float(out[min_y:max_y, min_x:max_x].mean())

    assert painted_mean < 30, (
        f"painted eye region mean ({painted_mean}) should be near-black (<30)"
    )
    assert painted_mean < orig_mean - 50
    assert out.shape == rgb.shape


def test_augment_returns_distinct_encoding(sample_face_path):
    """Synthetic encoding must differ from the bare encoding by a noticeable margin."""
    _ensure_models()
    bgr = cv2.imread(str(sample_face_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shape = _detect_and_shape(rgb)
    face_encoder = dlib.face_recognition_model_v1(
        str(__import__("scanner").FACE_REC_MODEL)
    )

    bare = np.array(face_encoder.compute_face_descriptor(rgb, shape))
    augmented = _augment_reference_encodings(rgb, shape, face_encoder)

    assert len(augmented) == 1, "expected exactly one synthetic encoding"
    synthetic = augmented[0]
    assert synthetic.shape == (128,)

    dist = float(np.linalg.norm(bare - synthetic))
    assert dist > 0.2, (
        f"synthetic encoding ({dist:.3f} from bare) should be meaningfully "
        "different — too close suggests the paint didn't take effect"
    )


def test_load_references_includes_augmented_encodings(
    tmp_path, sample_face_path, monkeypatch
):
    """A folder with N reference images should yield 2N encodings (bare + sunglasses)."""
    import shutil

    rdir = tmp_path / "refs"
    alice = rdir / "alice"
    alice.mkdir(parents=True)
    for i in range(3):
        shutil.copy(sample_face_path, alice / f"alice_{i}.jpg")

    # Force sequential path so we exercise _load_face directly
    monkeypatch.setenv("FACE_FINDER_PARALLEL", "0")
    cache = rdir / ".encodings_cache.pkl"
    if cache.exists():
        cache.unlink()

    people = load_references(str(rdir), tolerance=0.6)
    assert "alice" in people
    # 3 source photos × 2 encodings (bare + sunglasses) = 6
    assert len(people["alice"]) == 6
