"""Keep pytest's temporary files under work/, including in a fresh checkout."""

from pathlib import Path


def pytest_configure() -> None:
    (Path(__file__).resolve().parents[1] / "work").mkdir(exist_ok=True)
