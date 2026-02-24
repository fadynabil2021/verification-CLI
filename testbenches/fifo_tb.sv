`timescale 1ns/1ps

module fifo_tb;
    // Parameters
    localparam DATA_WIDTH = 8;
    localparam ADDR_WIDTH = 4;
    localparam MAX_DEPTH = 16;

    // Signals
    logic clk, rst_n;
    logic [DATA_WIDTH-1:0] data_in, data_out;
    logic wr_en, rd_en, full, empty;
    logic valid, ready;
`include "assertions.svh"

    // DUT Instantiation
    fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_DEPTH(MAX_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .wr_en(wr_en),
        .rd_en(rd_en),
        .data_out(data_out),
        .full(full),
        .empty(empty),
        .valid(valid),
        .ready(ready)
    );

    // Clock Generation
    initial clk = 0;
    always #5 clk = ~clk;  // 10ns period

    // Determinism: Lock simulation seed
    integer seed;
    initial begin
        seed = 12345;
        $urandom(seed);
    end

    // Counter Boundary Test (Mutation Target 1)
    // DUT contains overflow assertion

    // Basic stimulus
    initial begin
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        valid = 0;
        ready = 1;
        data_in = 0;

        #20;
        rst_n = 1;
        #10;

        // Phase 1: normal writes
        repeat (3) begin
            @(posedge clk);
            valid = 1;
            wr_en = 1;
            data_in = data_in + 1;
        end

        // Phase 2: backpressure (ready low) with writes asserted
        repeat (3) begin
            @(posedge clk);
            ready = 0;
            valid = 1;
            wr_en = 1;
            data_in = data_in + 1;
        end

        // Phase 3: resume ready and write a bit more
        ready = 1;
        repeat (2) begin
            @(posedge clk);
            valid = 1;
            wr_en = 1;
            data_in = data_in + 1;
        end

        @(posedge clk);
        wr_en = 0;
        valid = 0;

        // Read out what we wrote
        repeat (8) begin
            @(posedge clk);
            rd_en = 1;
        end
        @(posedge clk);
        rd_en = 0;
        #50;
        $finish;
    end

    // Scoreboard (Label Enrichment)
    logic [DATA_WIDTH-1:0] expected_data[$];
    always @(posedge clk) begin
        if (valid && ready && wr_en && !full) expected_data.push_back(data_in);
        if (rd_en && !empty && expected_data.size() > 0) begin
            logic [DATA_WIDTH-1:0] exp;
            exp = expected_data.pop_front();
            if (data_out !== exp) begin
                $error("SCOREBOARD_FAIL: Data mismatch expected=%0d actual=%0d at cycle %0t", exp, data_out, $time);
                tb_log_scoreboard_mismatch(
                    exp,
                    data_out,
                    $time,
                    dut.count,
                    dut.wptr,
                    dut.rptr,
                    valid,
                    ready,
                    wr_en,
                    rd_en
                );
            end
        end
        if (rd_en && !empty && expected_data.size() == 0) begin
            $error("SCOREBOARD_FAIL: Unexpected data actual=%0d at cycle %0t", data_out, $time);
            tb_log_scoreboard_unexpected(
                data_out,
                $time,
                dut.count,
                dut.wptr,
                dut.rptr,
                valid,
                ready,
                wr_en,
                rd_en
            );
        end
    end

    // Structured assertion diagnostics to simplify downstream parsing.
    always @(posedge clk) begin
        if (rst_n && (dut.count > MAX_DEPTH)) begin
            tb_log_counter_overflow(
                dut.count,
                $time,
                dut.wptr,
                dut.rptr,
                full,
                empty,
                valid,
                ready,
                wr_en,
                rd_en
            );
        end

    end
endmodule
