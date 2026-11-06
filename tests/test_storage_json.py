"""Atomic JSON writes preserve the previous document when persistence fails."""

from __future__ import annotations

import json

import pytest

from yeastbench.storage.json import atomic_write_json


def test_atomic_json_failure_preserves_existing_document(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"version": 1})

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr("yeastbench.storage.json.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_json_rejects_non_finite_values_before_replacing(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"version": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"version": 1}
