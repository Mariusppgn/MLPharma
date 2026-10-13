"""Source-release guardrails, independent of any Kaggle files."""

from pathlib import Path

import pytest

from scripts.prepare_public_release import ROOT, SECRET_PATTERNS, prepare_release, source_files


def test_public_release_allowlist_excludes_private_artifacts():
    paths = [path.relative_to(ROOT) for path in source_files()]
    assert Path("LICENSE") in paths
    assert Path("src/deep_pharma/prr_pytorch.py") in paths
    private_roots = {"data", "artifacts", "outputs", "work", "reports"}
    assert all(path.parts[0] not in private_roots for path in paths)
    assert all(path.suffix not in {".csv", ".ipynb", ".pt", ".npz"} for path in paths)
    assert Path("docs/history_pipeline.md") not in paths


def test_public_release_rejects_outside_destination(tmp_path):
    with pytest.raises(ValueError, match="Destination"):
        prepare_release(tmp_path / "release")


def test_secret_patterns_match_credentials_not_placeholder_names():
    assert any(pattern.search("ghp_" + "X" * 30) for pattern in SECRET_PATTERNS)
    fake_url = "https://" + "user:secret@example.com"
    assert any(pattern.search(fake_url) for pattern in SECRET_PATTERNS)
    assert not any(pattern.search('"access_token": ""') for pattern in SECRET_PATTERNS)
