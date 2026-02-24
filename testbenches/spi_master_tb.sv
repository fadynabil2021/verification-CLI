`timescale 1ns/1ps
// SPI Master Golden Testbench — VeriLog AI mutation target
// Procedural assertions (Verilator-compatible, no SVA ##1)
// Structured TB_LOG format for log parser

module spi_master_tb;

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    localparam CLK_DIV    = 4;
    localparam DATA_WIDTH = 8;
    localparam CLK_HALF   = 5;      // 100 MHz

    // -------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------
    logic                  clk, rst_n;
    logic [DATA_WIDTH-1:0] tx_data;
    logic                  start;
    logic [DATA_WIDTH-1:0] rx_data;
    logic                  done;
    logic                  sclk, mosi, miso, cs_n;

    // -------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------
    spi_master #(
        .CLK_DIV   (CLK_DIV),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .tx_data(tx_data),
        .start  (start),
        .rx_data(rx_data),
        .done   (done),
        .sclk   (sclk),
        .mosi   (mosi),
        .miso   (miso),
        .cs_n   (cs_n)
    );

    // -------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------
    initial clk  = 1'b0;
    always  #CLK_HALF clk = ~clk;

    // Loopback MISO = MOSI (slave echoes master)
    assign miso = mosi;

    // -------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] expected_rx[$];
    integer                scoreboard_errors;
    initial scoreboard_errors = 0;

    // Capture each expected echoed byte
    always @(posedge clk) begin
        if (start && cs_n)        // transaction launches
            expected_rx.push_back(tx_data);
    end

    // Verify received data on done
    always @(posedge done) begin
        if (expected_rx.size() > 0) begin
            automatic logic [DATA_WIDTH-1:0] exp = expected_rx.pop_front();
            if (rx_data !== exp) begin
                $display("TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=%0h|actual=%0h|cycle=%0t",
                         exp, rx_data, $time);
                $error("SCOREBOARD_FAIL: SPI MISO mismatch expected=0x%0h actual=0x%0h at cycle %0t",
                       exp, rx_data, $time);
                scoreboard_errors = scoreboard_errors + 1;
            end
        end
    end

    // -------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------
    initial begin
        rst_n   = 1'b0;
        start   = 1'b0;
        tx_data = 8'h00;
        @(posedge clk); @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Transfer 4 bytes
        repeat (4) begin
            tx_data <= $urandom_range(0, 255);
            start   <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            // Wait for done
            @(posedge done);
            @(posedge clk);
        end

        repeat (8) @(posedge clk);
        $finish;
    end

    // -------------------------------------------------------------------
    // Assertions
    // -------------------------------------------------------------------

    // 1. CS_N must be asserted (low) during transfer
    always @(posedge clk) begin
        if (rst_n && !done && !start && sclk) begin
            if (cs_n) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_CS_DEASSERTED_DURING_TRANSFER|cycle=%0t|signals=cs_n=%0b,sclk=%0b",
                         $time, cs_n, sclk);
                $error("ASSERT_FAIL: SPI CS_N deasserted during active transfer at cycle %0t", $time);
            end
        end
    end

    // 2. SCLK must be low when CS_N is high (idle)
    always @(posedge clk) begin
        if (rst_n && cs_n && sclk) begin
            $display("TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_SCLK_ACTIVE_WHEN_IDLE|cycle=%0t|signals=cs_n=%0b,sclk=%0b",
                     $time, cs_n, sclk);
            $error("ASSERT_FAIL: SPI SCLK active while CS_N high at cycle %0t", $time);
        end
    end

    // 3. Bit counter overflow: done must arrive within DATA_WIDTH*CLK_DIV*2 cycles
    integer xfer_watchdog;
    logic   xfer_active;
    initial begin
        xfer_watchdog = 0;
        xfer_active   = 1'b0;
    end
    always @(posedge clk) begin
        if (!cs_n) begin
            xfer_active   = 1'b1;
            xfer_watchdog = xfer_watchdog + 1;
            if (xfer_watchdog > DATA_WIDTH * CLK_DIV * 4 + 8) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_BIT_COUNTER_OVERFLOW|cycle=%0t|signals=bit_watchdog=%0d",
                         $time, xfer_watchdog);
                $error("ASSERT_FAIL: SPI transfer watchdog expired — bit counter may have overflowed at cycle %0t", $time);
                xfer_watchdog = 0;
            end
        end else begin
            if (xfer_active) xfer_watchdog = 0;
            xfer_active = 1'b0;
        end
    end

    // -------------------------------------------------------------------
    // Final report
    // -------------------------------------------------------------------
    final begin
        if (scoreboard_errors > 0)
            $display("TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_FRAME|total_errors=%0d", scoreboard_errors);
        else
            $display("TB_LOG|kind=PASS|assert_id=SPI_ALL_OK");
    end

endmodule
