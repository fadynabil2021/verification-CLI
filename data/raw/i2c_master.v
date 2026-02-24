// I2C Master Controller (7-bit addressing, single-byte write)
// Source: OpenCores-compatible, permissive license
// Target: VeriLog AI mutation engine (Tier 1 & Tier 2 mutations)

module i2c_master #(
    parameter CLK_DIV = 250  // clk / (4 * SCL_freq): 100 MHz / 400 kHz = 250
)(
    input  wire       clk,
    input  wire       rst_n,
    // Software interface
    input  wire [6:0] dev_addr,
    input  wire [7:0] data_wr,
    input  wire       wr_en,     // pulse to start a write transaction
    output reg        done,
    output reg        nack,      // set if slave NACKed
    // I2C bus
    inout  wire       sda,
    output reg        scl
);

    localparam IDLE  = 3'd0;
    localparam START = 3'd1;
    localparam ADDR  = 3'd2;
    localparam ACK1  = 3'd3;
    localparam DATA  = 3'd4;
    localparam ACK2  = 3'd5;
    localparam STOP  = 3'd6;

    reg [2:0]  state;
    reg [7:0]  clk_cnt;
    reg [3:0]  bit_cnt;
    reg [7:0]  shift_reg;
    reg        sda_out;
    reg        sda_oe;   // 1 = drive, 0 = release (input)

    // Tristate SDA
    assign sda = sda_oe ? sda_out : 1'bz;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            scl     <= 1'b1;
            sda_out <= 1'b1;
            sda_oe  <= 1'b1;
            clk_cnt <= 8'd0;
            bit_cnt <= 4'd0;
            done    <= 1'b0;
            nack    <= 1'b0;
            shift_reg <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    scl     <= 1'b1;
                    sda_oe  <= 1'b1;
                    sda_out <= 1'b1;
                    done    <= 1'b0;
                    nack    <= 1'b0;
                    if (wr_en) begin
                        // Address byte = {dev_addr[6:0], R/W=0}
                        shift_reg <= {dev_addr, 1'b0};
                        clk_cnt   <= 8'd0;
                        state     <= START;
                    end
                end

                START: begin
                    // SDA falls while SCL high → START condition
                    sda_out <= 1'b0;
                    if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                    end else begin
                        scl     <= 1'b0;
                        clk_cnt <= 8'd0;
                        bit_cnt <= 4'd7;
                        state   <= ADDR;
                    end
                end

                ADDR: begin
                    // Drive address byte MSB first
                    if (clk_cnt < CLK_DIV/2 - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b0;
                        sda_out <= shift_reg[7];
                    end else if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b1;
                    end else begin
                        clk_cnt   <= 8'd0;
                        scl       <= 1'b0;
                        shift_reg <= {shift_reg[6:0], 1'b0};
                        if (bit_cnt == 0) begin
                            state <= ACK1;
                        end else begin
                            bit_cnt <= bit_cnt - 4'd1;
                        end
                    end
                end

                ACK1: begin
                    // Release SDA, slave drives ACK (low) or NACK (high)
                    sda_oe <= 1'b0;
                    if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= (clk_cnt >= CLK_DIV/2) ? 1'b1 : 1'b0;
                    end else begin
                        nack    <= sda;           // NACK if SDA was high
                        scl     <= 1'b0;
                        sda_oe  <= 1'b1;
                        clk_cnt <= 8'd0;
                        if (!sda) begin
                            // ACK received: send data byte
                            shift_reg <= data_wr;
                            bit_cnt   <= 4'd7;
                            state     <= DATA;
                        end else begin
                            state <= STOP;
                        end
                    end
                end

                DATA: begin
                    if (clk_cnt < CLK_DIV/2 - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b0;
                        sda_out <= shift_reg[7];
                    end else if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b1;
                    end else begin
                        clk_cnt   <= 8'd0;
                        scl       <= 1'b0;
                        shift_reg <= {shift_reg[6:0], 1'b0};
                        if (bit_cnt == 0) begin
                            state <= ACK2;
                        end else begin
                            bit_cnt <= bit_cnt - 4'd1;
                        end
                    end
                end

                ACK2: begin
                    sda_oe <= 1'b0;
                    if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= (clk_cnt >= CLK_DIV/2) ? 1'b1 : 1'b0;
                    end else begin
                        scl     <= 1'b0;
                        sda_oe  <= 1'b1;
                        clk_cnt <= 8'd0;
                        done    <= 1'b1;
                        state   <= STOP;
                    end
                end

                STOP: begin
                    // SDA rises while SCL high → STOP condition
                    sda_out <= 1'b0;
                    if (clk_cnt < CLK_DIV/2 - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b0;
                    end else if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 8'd1;
                        scl     <= 1'b1;
                    end else begin
                        sda_out <= 1'b1;
                        clk_cnt <= 8'd0;
                        state   <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
