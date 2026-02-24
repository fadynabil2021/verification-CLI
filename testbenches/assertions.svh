`ifndef VERILOG_AI_ASSERTIONS_SVH
`define VERILOG_AI_ASSERTIONS_SVH

`define ASSERT_ID_COUNTER_OVERFLOW "FIFO_COUNTER_OVERFLOW"
`define ASSERT_ID_SB_DATA_MISMATCH "SB_DATA_MISMATCH"
`define ASSERT_ID_SB_UNEXPECTED_DATA "SB_UNEXPECTED_DATA"

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
