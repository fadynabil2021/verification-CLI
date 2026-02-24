from __future__ import annotations

import argparse
from pathlib import Path


ASSERT_IDS = {
    "counter_overflow": "FIFO_COUNTER_OVERFLOW",
    "scoreboard_mismatch": "SB_DATA_MISMATCH",
    "scoreboard_unexpected": "SB_UNEXPECTED_DATA",
}


def render_svh() -> str:
    return f"""`ifndef VERILOG_AI_ASSERTIONS_SVH
`define VERILOG_AI_ASSERTIONS_SVH

`define ASSERT_ID_COUNTER_OVERFLOW "{ASSERT_IDS["counter_overflow"]}"
`define ASSERT_ID_SB_DATA_MISMATCH "{ASSERT_IDS["scoreboard_mismatch"]}"
`define ASSERT_ID_SB_UNEXPECTED_DATA "{ASSERT_IDS["scoreboard_unexpected"]}"

task automatic tb_log_scoreboard_mismatch(
    input integer expected,
    input integer actual,
    input longint cycle,
    input integer count,
    input integer wptr,
    input integer rptr,
    input bit valid,
    input bit ready,
    input bit wr_en,
    input bit rd_en
);
    $display(
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=%s|expected=%0d|actual=%0d|cycle=%0d|signals=count=%0d,wptr=%0d,rptr=%0d,valid=%0b,ready=%0b,wr_en=%0b,rd_en=%0b",
        `ASSERT_ID_SB_DATA_MISMATCH,
        expected,
        actual,
        cycle,
        count,
        wptr,
        rptr,
        valid,
        ready,
        wr_en,
        rd_en
    );
endtask

task automatic tb_log_scoreboard_unexpected(
    input integer actual,
    input longint cycle,
    input integer count,
    input integer wptr,
    input integer rptr,
    input bit valid,
    input bit ready,
    input bit wr_en,
    input bit rd_en
);
    $display(
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=%s|actual=%0d|cycle=%0d|signals=count=%0d,wptr=%0d,rptr=%0d,valid=%0b,ready=%0b,wr_en=%0b,rd_en=%0b",
        `ASSERT_ID_SB_UNEXPECTED_DATA,
        actual,
        cycle,
        count,
        wptr,
        rptr,
        valid,
        ready,
        wr_en,
        rd_en
    );
endtask

task automatic tb_log_counter_overflow(
    input integer count,
    input longint cycle,
    input integer wptr,
    input integer rptr,
    input bit full,
    input bit empty,
    input bit valid,
    input bit ready,
    input bit wr_en,
    input bit rd_en
);
    $display(
        "TB_LOG|kind=ASSERT_FAIL|assert_id=%s|count=%0d|cycle=%0d|signals=count=%0d,wptr=%0d,rptr=%0d,full=%0b,empty=%0b,valid=%0b,ready=%0b,wr_en=%0b,rd_en=%0b",
        `ASSERT_ID_COUNTER_OVERFLOW,
        count,
        cycle,
        count,
        wptr,
        rptr,
        full,
        empty,
        valid,
        ready,
        wr_en,
        rd_en
    );
endtask

`endif
"""


def emit_svh(path: Path) -> None:
    path.write_text(render_svh(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SystemVerilog assertion helper include")
    parser.add_argument(
        "--emit-svh",
        default="testbenches/assertions.svh",
        help="Output path for generated assertions include",
    )
    args = parser.parse_args()
    emit_svh(Path(args.emit_svh))
    print(f"Wrote assertion helpers to {args.emit_svh}")


if __name__ == "__main__":
    main()
