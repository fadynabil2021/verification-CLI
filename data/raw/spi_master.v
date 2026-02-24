// SPI Master Controller (Mode 0: CPOL=0, CPHA=0)
// Source: OpenCores-compatible, permissive license
// Target: VeriLog AI mutation engine (Tier 1 & Tier 2 mutations)

module spi_master #(
    parameter CLK_DIV    = 4,   // sclk = clk / (2 * CLK_DIV)
    parameter DATA_WIDTH = 8
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // Control interface
    input  wire [DATA_WIDTH-1:0] tx_data,
    input  wire                  start,
    output reg  [DATA_WIDTH-1:0] rx_data,
    output reg                   done,
    // SPI bus
    output reg                   sclk,
    output reg                   mosi,
    input  wire                  miso,
    output reg                   cs_n
);

    localparam IDLE  = 2'd0;
    localparam TRANS = 2'd1;
    localparam DONE  = 2'd2;

    reg [1:0]              state;
    reg [3:0]              bit_cnt;
    reg [3:0]              clk_cnt;
    reg [DATA_WIDTH-1:0]   shift_tx;
    reg [DATA_WIDTH-1:0]   shift_rx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            sclk     <= 1'b0;
            cs_n     <= 1'b1;
            mosi     <= 1'b0;
            done     <= 1'b0;
            clk_cnt  <= 4'd0;
            bit_cnt  <= 4'd0;
            rx_data  <= {DATA_WIDTH{1'b0}};
            shift_tx <= {DATA_WIDTH{1'b0}};
            shift_rx <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    sclk <= 1'b0;
                    cs_n <= 1'b1;
                    done <= 1'b0;
                    if (start) begin
                        shift_tx <= tx_data;
                        cs_n     <= 1'b0;
                        bit_cnt  <= DATA_WIDTH - 1;
                        clk_cnt  <= 4'd0;
                        state    <= TRANS;
                    end
                end

                TRANS: begin
                    if (clk_cnt < CLK_DIV - 1) begin
                        clk_cnt <= clk_cnt + 4'd1;
                    end else begin
                        clk_cnt <= 4'd0;
                        sclk    <= ~sclk;

                        if (!sclk) begin
                            // Rising edge (CPOL=0): sample MISO
                            shift_rx <= {shift_rx[DATA_WIDTH-2:0], miso};
                            if (bit_cnt == 0) begin
                                state   <= DONE;
                                rx_data <= {shift_rx[DATA_WIDTH-2:0], miso};
                            end else begin
                                bit_cnt <= bit_cnt - 4'd1;
                            end
                        end else begin
                            // Falling edge: drive MOSI
                            mosi     <= shift_tx[DATA_WIDTH-1];
                            shift_tx <= {shift_tx[DATA_WIDTH-2:0], 1'b0};
                        end
                    end
                end

                DONE: begin
                    cs_n  <= 1'b1;
                    sclk  <= 1'b0;
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
