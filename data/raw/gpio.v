// GPIO Controller with direction register, input sampling, and IRQ on rising edge
// Source: OpenCores-compatible, permissive license
// Target: VeriLog AI mutation engine (Tier 1 & Tier 2 mutations)

module gpio #(
    parameter WIDTH = 8
)(
    input  wire             clk,
    input  wire             rst_n,
    // Register interface
    input  wire [WIDTH-1:0] data_in,
    input  wire [WIDTH-1:0] dir_reg,   // 1 = output, 0 = input
    input  wire             wr_en,
    input  wire             rd_en,
    output reg  [WIDTH-1:0] data_out,
    // GPIO pins (tristate)
    inout  wire [WIDTH-1:0] gpio_pins,
    // Interrupt: rising edge on any input pin
    output reg              irq
);

    reg [WIDTH-1:0] out_reg;
    reg [WIDTH-1:0] in_reg;
    reg [WIDTH-1:0] in_reg_prev;

    // Tristate output drivers
    genvar gi;
    generate
        for (gi = 0; gi < WIDTH; gi = gi + 1) begin : gpio_drv
            assign gpio_pins[gi] = dir_reg[gi] ? out_reg[gi] : 1'bz;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_reg     <= {WIDTH{1'b0}};
            data_out    <= {WIDTH{1'b0}};
            in_reg      <= {WIDTH{1'b0}};
            in_reg_prev <= {WIDTH{1'b0}};
            irq         <= 1'b0;
        end else begin
            // Sample GPIO input pins (two-stage metastability filter omitted for brevity)
            in_reg_prev <= in_reg;
            in_reg      <= gpio_pins;

            // Write output register — only affect pins configured as output
            if (wr_en) begin
                out_reg <= data_in & dir_reg;
            end

            // Read captured input register
            if (rd_en) begin
                data_out <= in_reg;
            end

            // IRQ: rising edge on any input pin (dir_reg[i]==0)
            irq <= |(in_reg & ~in_reg_prev & ~dir_reg);
        end
    end

endmodule
