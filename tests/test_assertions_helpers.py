from pathlib import Path

from testbenches.assertions import emit_svh, render_svh


def test_render_svh_contains_expected_ids_and_tasks() -> None:
    text = render_svh()
    assert "ASSERT_ID_COUNTER_OVERFLOW" in text
    assert "FIFO_COUNTER_OVERFLOW" in text
    assert "task automatic tb_log_scoreboard_mismatch" in text
    assert "task automatic tb_log_counter_overflow" in text


def test_emit_svh_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "assertions.svh"
    emit_svh(out)
    text = out.read_text()
    assert text.startswith("`ifndef VERILOG_AI_ASSERTIONS_SVH")
    assert "TB_LOG|kind=SCOREBOARD_FAIL" in text
