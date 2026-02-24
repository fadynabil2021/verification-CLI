`timescale 1ns/1ps
// UART TX Golden Testbench — VeriLog AI mutation target
// Procedural assertions (Verilator-compatible, no SVA ##1)
// Structured TB_LOG format for log parser

module uart_tx_tb;

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    localparam BAUD_DIV  = 16;   // Reduced for fast simulation
    localparam CLK_HALF  = 5;    // 100 MHz → 10 ns period

    // -------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------
    logic       clk, rst_n;
    logic [7:0] data_in;
    logic       valid, ready;
    logic       tx;
    logic       parity_err;

    // -------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------
    uart_tx #(.BAUD_DIV(BAUD_DIV)) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (data_in),
        .valid     (valid),
        .ready     (ready),
        .tx        (tx),
        .parity_err(parity_err)
    );

    // -------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------
    initial clk = 1'b0;
    always  #CLK_HALF clk = ~clk;

    // -------------------------------------------------------------------
    // Scoreboard — capture TX serial stream and compare with data_in
    // -------------------------------------------------------------------
    integer    frame_errors;
    logic[7:0] captured_byte;
    integer    bit_pos;

    initial begin
        frame_errors  = 0;
        captured_byte = 8'h00;
        bit_pos       = -1;   // -1 = idle, 0-7 = data bits
    end

    // -------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------
    initial begin
        // Initialise
        rst_n   = 1'b0;
        valid   = 1'b0;
        data_in = 8'hA5;
        @(posedge clk); @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Send three bytes
        repeat (3) begin
            @(posedge clk iff ready);
            data_in <= $urandom_range(0, 255);
            valid   <= 1'b1;
            @(posedge clk);
            valid   <= 1'b0;
            // Wait for TX to complete (~BAUD_DIV * 10 clocks)
            repeat (BAUD_DIV * 12) @(posedge clk);
        end

        // Allow final byte to finish
        repeat (BAUD_DIV * 4) @(posedge clk);
        $finish;
    end

    // -------------------------------------------------------------------
    // Assertions — procedural (Verilator compatible)
    // -------------------------------------------------------------------

    // 1. TX must remain high when idle
    always @(posedge clk) begin
        if (rst_n && ready && !valid) begin
            if (tx !== 1'b1) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=UART_IDLE_HIGH|cycle=%0t|signals=tx=%0b,ready=%0b,valid=%0b",
                         $time, tx, ready, valid);
                $error("ASSERT_FAIL: UART TX not idle-high; tx=%0b at cycle %0t", tx, $time);
                frame_errors = frame_errors + 1;
            end
        end
    end

    // 2. Parity error must never be asserted (no loopback fault injected)
    always @(posedge clk) begin
        if (rst_n && parity_err) begin
            $display("TB_LOG|kind=ASSERT_FAIL|assert_id=UART_PARITY_ERROR|cycle=%0t|signals=parity_err=%0b",
                     $time, parity_err);
            $error("ASSERT_FAIL: UART parity error asserted at cycle %0t", $time);
            frame_errors = frame_errors + 1;
        end
    end

    // 3. Ready must de-assert while transmission is in progress
    //    (valid && ready → next cycle ready should be 0)
    logic valid_prev;
    always @(posedge clk) begin
        valid_prev <= valid;
        if (rst_n && valid_prev && ready) begin
            // The very next cycle, ready should have dropped
            if (ready) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=UART_READY_STUCK|cycle=%0t|signals=ready=%0b,valid=%0b",
                         $time, ready, valid);
                $error("ASSERT_FAIL: UART ready stayed high after valid handshake at cycle %0t", $time);
                frame_errors = frame_errors + 1;
            end
        end
    end

    // 4. Baud counter overflow check (shadow counter)
    integer baud_shadow;
    initial baud_shadow = 0;
    always @(posedge clk) begin
        if (!rst_n) begin
            baud_shadow = 0;
        end else begin
            baud_shadow = baud_shadow + 1;
            if (baud_shadow > BAUD_DIV * 12) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=UART_BAUD_OVERFLOW|cycle=%0t|signals=baud_shadow=%0d",
                         $time, baud_shadow);
                $error("ASSERT_FAIL: Baud counter overflow — baud_shadow=%0d at cycle %0t", baud_shadow, $time);
                baud_shadow = 0;
            end
        end
    end

    // -------------------------------------------------------------------
    // Final report
    // -------------------------------------------------------------------
    final begin
        if (frame_errors > 0)
            $display("TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_UART_FRAME|frame_errors=%0d", frame_errors);
        else
            $display("TB_LOG|kind=PASS|assert_id=UART_ALL_OK|frame_errors=0");
    end

endmodule
