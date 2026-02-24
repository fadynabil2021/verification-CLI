import shutil
from pathlib import Path

import pytest

from data.generate_dataset import run_verilator


pytestmark = pytest.mark.verilator


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_verilator_runs_fifo_tb():
    rtl = Path("data/raw/fifo.v")
    tb = Path("testbenches/fifo_tb.sv")
    log, err = run_verilator(rtl, tb, "fifo_tb")
    assert err is None
    assert "ASSERT_FAIL" in log or "SCOREBOARD_FAIL" in log
    assert "TB_LOG|" in log
