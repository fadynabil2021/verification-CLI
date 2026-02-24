`timescale 1ns/1ps

module fifo #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 4,
    parameter MAX_DEPTH = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [DATA_WIDTH-1:0] data_in,
    input  wire                  wr_en,
    input  wire                  rd_en,
    output reg  [DATA_WIDTH-1:0] data_out,
    output reg                   full,
    output reg                   empty,
    input  wire                  valid,
    input  wire                  ready
);

    reg [DATA_WIDTH-1:0] mem [0:MAX_DEPTH-1];
    reg [ADDR_WIDTH-1:0] wptr, rptr;
    reg [ADDR_WIDTH:0]   count;

    wire write_fire = (valid && ready && wr_en && allow_write);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= 0;
            rptr <= 0;
            count <= 0;
            full <= 0;
            empty <= 1;
            data_out <= 0;
        end else begin
            if (write_fire) begin
                mem[wptr] <= data_in;
                wptr <= wptr + 1;
                count <= count + 1;
            end

            if (rd_en && !empty) begin
                data_out <= mem[rptr];
                rptr <= rptr + 1;
                count <= count - 1;
            end

            full <= (count >= MAX_DEPTH);
            empty <= (count == 0);
        end
    end

    // Boundary check used by mutation engine
    wire allow_write = (count <= MAX_DEPTH);

    // Simple overflow check (for simulation logs)
    always @(posedge clk) begin
        if (count > MAX_DEPTH) begin
            $error("ASSERT_FAIL: Counter overflow count=%0d at cycle %0t", count, $time);
        end
        if (write_fire && !ready) begin
            $error("ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle %0t", $time);
        end
    end

endmodule
