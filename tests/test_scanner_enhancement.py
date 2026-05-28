"""Tests for low-light detection enhancement (CLAHE + gamma + TTA fallback)."""

import cv2
import numpy as np
import pytest

from scanner import _detect_faces, _enhance_for_detection, _worker_init


def _mean_luminance(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())


def test_enhance_skips_bright_images(sample_face_path):
    """A well-lit image must be returned untouched (no extra cost on bright frames)."""
    bgr = cv2.imread(str(sample_face_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert _mean_luminance(rgb) >= 80, "fixture must be bright for this test to be valid"

    out = _enhance_for_detection(rgb)

    # Same object returned (no copy) when above threshold.
    assert out is rgb


def test_enhance_brightens_dark_images(sample_face_path):
    """A darkened image must come back with higher mean luminance."""
    bgr = cv2.imread(str(sample_face_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dark = (rgb.astype(np.float32) * 0.2).clip(0, 255).astype(np.uint8)
    dark_l = _mean_luminance(dark)
    assert dark_l < 80, "synthetic dark image must be below threshold"

    out = _enhance_for_detection(dark)
    out_l = _mean_luminance(out)

    assert out_l > dark_l, f"enhanced luminance {out_l} not greater than {dark_l}"
    assert out.shape == dark.shape


def test_detect_faces_finds_face_on_dark_image(sample_face_path):
    """_detect_faces should locate the face even after we darken the image significantly."""
    _worker_init()
    from scanner import _WORKER_DETECTOR

    bgr = cv2.imread(str(sample_face_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dark = (rgb.astype(np.float32) * 0.25).clip(0, 255).astype(np.uint8)

    detected = _detect_faces(dark, _WORKER_DETECTOR)

    # The fixture has a clearly visible face; even darkened we should still find it
    # via the enhancement + fallback path.
    assert len(detected) >= 1
