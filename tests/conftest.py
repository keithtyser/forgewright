"""Keep the test suite hermetic: point FORGEWRIGHT_HOME at a per-session temp dir so anything
that falls back to the default home (the outcome memory, checkpoints, a default Registry/Ledger)
writes under tmp instead of the user's real ~/.forgewright. Individual tests that set their own
FORGEWRIGHT_HOME via monkeypatch still override this.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("fw_home")
    monkeypatch.setenv("FORGEWRIGHT_HOME", str(home))
    yield
