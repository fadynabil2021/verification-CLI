from pathlib import Path

from data.generate_dataset import parse_log_for_label


def _read_fixture(name: str) -> str:
    return (Path("tests/fixtures/logs") / name).read_text()


def test_parse_structured_tb_scoreboard_log() -> None:
    label, explanation, confidence = parse_log_for_label(
        _read_fixture("structured_tb_scoreboard.log")
    )
    assert label == "Data Integrity Error"
    assert "assertion_id=SB_DATA_MISMATCH" in explanation
    assert "signals=" in explanation
    assert confidence >= 0.9


def test_parse_verilator_assertion_id_with_signal_dump_counter() -> None:
    label, explanation, confidence = parse_log_for_label(
        _read_fixture("verilator_assert_id_signal_dump.log")
    )
    assert label == "Off-by-One Error"
    assert "assertion_id=FIFO_COUNTER_OVERFLOW" in explanation
    assert "signals=" in explanation
    assert confidence >= 0.85


def test_parse_verilator_assertion_id_with_signal_dump_handshake() -> None:
    label, explanation, confidence = parse_log_for_label(
        _read_fixture("verilator_assert_id_handshake.log")
    )
    assert label == "Handshake Protocol Violation"
    assert "assertion_id=FIFO_BACKPRESSURE_WRITE" in explanation
    assert confidence >= 0.75
