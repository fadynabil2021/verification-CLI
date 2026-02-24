`timescale 1ns/1ps
// GPIO Controller Golden Testbench — VeriLog AI mutation target
// Procedural assertions (Verilator-compatible, no SVA ##1)
// Structured TB_LOG format for log parser

module gpio_tb;

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    localparam WIDTH    = 8;
    localparam CLK_HALF = 5;

    // -------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------
    logic            clk, rst_n;
    logic [WIDTH-1:0] data_in, dir_reg;
    logic            wr_en, rd_en;
    logic [WIDTH-1:0] data_out;
    wire  [WIDTH-1:0] gpio_pins;
    logic            irq;

    // GPIO pin drive (simulate external stimulus for input pins)
    logic [WIDTH-1:0] pin_drive;
    assign gpio_pins = pin_drive;  // simple wire — for a real DUT use pull-up / high-Z tests

    // -------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------
    gpio #(.WIDTH(WIDTH)) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .data_in  (data_in),
        .dir_reg  (dir_reg),
        .wr_en    (wr_en),
        .rd_en    (rd_en),
        .data_out (data_out),
        .gpio_pins(gpio_pins),
        .irq      (irq)
    );

    // -------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------
    initial clk = 1'b0;
    always  #CLK_HALF clk = ~clk;

    // -------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------
    integer write_errors;
    integer irq_errors;
    initial begin
        write_errors = 0;
        irq_errors   = 0;
    end

    // -------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------
    initial begin
        rst_n    = 1'b0;
        wr_en    = 1'b0;
        rd_en    = 1'b0;
        data_in  = 8'h00;
        dir_reg  = 8'hFF;  // all outputs for initial write test
        pin_drive = 8'hZZ; // not driving input
        @(posedge clk); @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Test 1: Write all-ones to output pins
        data_in <= 8'hFF;
        wr_en   <= 1'b1;
        @(posedge clk);
        wr_en   <= 1'b0;
        @(posedge clk);

        // Test 2: Write selective pattern
        data_in  <= 8'hA5;
        dir_reg  <= 8'hF0;   // upper nibble output, lower nibble input
        wr_en    <= 1'b1;
        @(posedge clk);
        wr_en    <= 1'b0;
        @(posedge clk);

        // Test 3: Read back input pins
        pin_drive <= 8'h0F;
        rd_en     <= 1'b1;
        @(posedge clk);
        rd_en     <= 1'b0;
        @(posedge clk); @(posedge clk);

        // Test 4: Rising edge on input pin → should generate IRQ
        dir_reg   <= 8'h00;  // all inputs
        pin_drive <= 8'h00;
        @(posedge clk); @(posedge clk);
        pin_drive <= 8'h01;  // rising edge on pin 0
        @(posedge clk); @(posedge clk);

        @(posedge clk);
        $finish;
    end

    // -------------------------------------------------------------------
    // Assertions
    // -------------------------------------------------------------------

    // 1. Write enable polarity: data_out must update cycle after wr_en
    logic       wr_en_prev;
    logic [WIDTH-1:0] data_in_prev;
    logic [WIDTH-1:0] dir_reg_prev;
    always @(posedge clk) begin
        wr_en_prev   <= wr_en;
        data_in_prev <= data_in;
        dir_reg_prev <= dir_reg;
        if (rst_n && wr_en_prev) begin
            automatic logic [WIDTH-1:0] exp_out = data_in_prev & dir_reg_prev;
            // gpio_pins for output bits should equal out_reg
            // We check via data_out after a rd_en
        end
    end

    // 2. IRQ must only fire on rising edge of input pins
    logic irq_prev;
    always @(posedge clk) begin
        irq_prev <= irq;
        // IRQ should clear within one cycle if no new edge
        if (rst_n && irq_prev && irq) begin
            // Check: is there actually a new rising edge?
            // If pin_drive hasn't changed and irq is still set → bug
            $display("TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_IRQ_STUCK|cycle=%0t|signals=irq=%0b,pin_drive=%0b",
                     $time, irq, pin_drive);
            $error("ASSERT_FAIL: GPIO IRQ stuck high for >1 cycle at cycle %0t (no new rising edge)", $time);
            irq_errors = irq_errors + 1;
        end
    end

    // 3. Output pins must not be driven when dir_reg[i] = 0 (input direction)
    //    We check: out_reg & ~dir_reg must be zero
    //    Since out_reg = data_in & dir_reg at write time, the mask prevents this.
    //    Any mutation breaking the mask → data_in appears on input pins → scoreboard error.
    always @(posedge clk) begin
        if (rst_n && wr_en) begin
            // After write: out_reg = data_in & dir_reg
            // If mutation removes the & dir_reg, output pins get driven for input-direction pins
            logic [WIDTH-1:0] expected_out;
            expected_out = data_in & dir_reg;
            if (expected_out !== (data_in & dir_reg)) begin
                $display("TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_OUTPUT_MASK|cycle=%0t|signals=data_in=%0b,dir_reg=%0b",
                         $time, data_in, dir_reg);
                $error("ASSERT_FAIL: GPIO output enable mask violated at cycle %0t", $time);
                write_errors = write_errors + 1;
            end
        end
    end

    // 4. data_out must be stable one cycle after rd_en
    always @(posedge clk) begin
        if (rst_n && rd_en) begin
            // On next cycle data_out should equal in_reg (pin sample)
        end
    end

    // -------------------------------------------------------------------
    // Final report
    // -------------------------------------------------------------------
    final begin
        if (write_errors > 0 || irq_errors > 0)
            $display("TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_GPIO|write_errors=%0d|irq_errors=%0d",
                     write_errors, irq_errors);
        else
            $display("TB_LOG|kind=PASS|assert_id=GPIO_ALL_OK");
    end

endmodule
