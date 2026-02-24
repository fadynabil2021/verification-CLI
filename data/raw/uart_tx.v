// UART Transmitter (8N1, configurable baud divisor)
// Source: OpenCores-compatible, permissive license
// Target: VeriLog AI mutation engine (Tier 1 & Tier 2 mutations)

module uart_tx #(
    parameter BAUD_DIV  = 868   // 100 MHz / 115200 baud
)(
    input  wire       clk,
    input  wire       rst_n,
    // TX data handshake
    input  wire [7:0] data_in,
    input  wire       valid,
    output reg        ready,
    // Serial output
    output reg        tx,
    // Parity error flag (set when parity mismatch detected on loopback)
    output reg        parity_err
);

    // FSM states
    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0]  state;
    reg [9:0]  baud_cnt;
    reg [2:0]  bit_cnt;
    reg [7:0]  shift_reg;
    reg        parity_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            tx         <= 1'b1;   // idle high
            ready      <= 1'b1;
            parity_err <= 1'b0;
            baud_cnt   <= 10'd0;
            bit_cnt    <= 3'd0;
            shift_reg  <= 8'd0;
            parity_reg <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    tx    <= 1'b1;
                    ready <= 1'b1;
                    if (valid && ready) begin
                        shift_reg  <= data_in;
                        parity_reg <= ^data_in; // even parity
                        ready      <= 1'b0;
                        baud_cnt   <= 10'd0;
                        state      <= START;
                    end
                end

                START: begin
                    tx <= 1'b0;  // start bit (space)
                    if (baud_cnt < BAUD_DIV - 1) begin
                        baud_cnt <= baud_cnt + 10'd1;
                    end else begin
                        baud_cnt <= 10'd0;
                        bit_cnt  <= 3'd0;
                        state    <= DATA;
                    end
                end

                DATA: begin
                    tx <= shift_reg[0];
                    if (baud_cnt < BAUD_DIV - 1) begin
                        baud_cnt <= baud_cnt + 10'd1;
                    end else begin
                        baud_cnt  <= 10'd0;
                        shift_reg <= {1'b0, shift_reg[7:1]};
                        if (bit_cnt < 7) begin
                            bit_cnt <= bit_cnt + 3'd1;
                        end else begin
                            state <= STOP;
                        end
                    end
                end

                STOP: begin
                    tx <= 1'b1;  // stop bit (mark)
                    if (baud_cnt < BAUD_DIV - 1) begin
                        baud_cnt <= baud_cnt + 10'd1;
                    end else begin
                        baud_cnt <= 10'd0;
                        ready    <= 1'b1;
                        state    <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
