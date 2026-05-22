"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def init_test_db(tmp_path_factory):
    """Initialize an in-memory test database."""
    from msomi.core.database import init_db
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    init_db(f"sqlite:///{db_path}")
