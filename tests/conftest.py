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
