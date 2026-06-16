"""License metadata validation tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_is_agpl_v3():
    """LICENSE file should contain the AGPL v3 heading."""
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 19 November 2007" in license_text


def test_pyproject_license_metadata_is_agpl():
    """pyproject metadata should reference AGPL-3.0."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'license = { text = "AGPL-3.0-or-later" }' in pyproject
    assert "GNU Affero General Public License v3 or later (AGPLv3+)" in pyproject
