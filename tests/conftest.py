"""Shared fixtures.

Building the full model set takes a couple of seconds, so it happens once per
session and is shared read-only by every test that needs real data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import build, connect

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = str(ROOT / "data" / "sample")


@pytest.fixture(scope="session")
def built():
    """A connection with every model materialised from the committed sample."""
    con = connect()
    result = build(con, SAMPLE)
    yield con, result
    con.close()


@pytest.fixture(scope="session")
def con(built):
    return built[0]


@pytest.fixture(scope="session")
def build_result(built):
    return built[1]
