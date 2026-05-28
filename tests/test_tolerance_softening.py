"""Tests for per-person tolerance softening based on reference count."""

import numpy as np
import pytest

from scanner import (
    TOLERANCE_BONUS_FREE_REFS,
    TOLERANCE_BONUS_MAX,
    TOLERANCE_BONUS_PER_REF,
    _compute_per_person_tolerance,
)


def _refs(counts):
    """Build a fake references dict {name: [encoding] * count}."""
    return {name: [np.zeros(128)] * n for name, n in counts.items()}


def test_few_refs_keep_base_tolerance():
    """1-2 references → no bonus, effective tolerance == base."""
    out = _compute_per_person_tolerance(_refs({"alice": 1, "bob": 2}), base_tolerance=0.6)
    assert out["alice"] == pytest.approx(0.6)
    assert out["bob"] == pytest.approx(0.6)


def test_extra_refs_add_proportional_bonus():
    """Each ref beyond TOLERANCE_BONUS_FREE_REFS adds TOLERANCE_BONUS_PER_REF."""
    out = _compute_per_person_tolerance(
        _refs({"alice": 3, "bob": 4, "carol": 5}), base_tolerance=0.6
    )
    assert out["alice"] == pytest.approx(0.6 + 1 * TOLERANCE_BONUS_PER_REF)
    assert out["bob"] == pytest.approx(0.6 + 2 * TOLERANCE_BONUS_PER_REF)
    assert out["carol"] == pytest.approx(0.6 + 3 * TOLERANCE_BONUS_PER_REF)


def test_bonus_is_capped():
    """A person with many references shouldn't get unbounded looseness."""
    out = _compute_per_person_tolerance(_refs({"alice": 50}), base_tolerance=0.6)
    assert out["alice"] == pytest.approx(0.6 + TOLERANCE_BONUS_MAX)


def test_different_base_tolerance_propagates():
    """Bonus stacks on top of whatever base the caller passed in."""
    out = _compute_per_person_tolerance(_refs({"alice": 4}), base_tolerance=0.55)
    assert out["alice"] == pytest.approx(0.55 + 2 * TOLERANCE_BONUS_PER_REF)
