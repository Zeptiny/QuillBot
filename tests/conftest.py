"""Shared pytest setup: import path, async tests and a throwaway history DB."""

import asyncio
import inspect
import os
import sys

import pytest

# Tests import the bot's modules (cogs, config, api_logger) from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run `async def test_*` functions on a fresh event loop (no plugin needed)."""
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    names = pyfuncitem._fixtureinfo.argnames
    asyncio.run(pyfuncitem.obj(**{name: pyfuncitem.funcargs[name] for name in names}))
    return True


@pytest.fixture
def tmpdb(tmp_path):
    """Path to a fresh SQLite file for HistoryRAG tests."""
    return str(tmp_path / 'history.db')
