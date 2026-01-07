`timescale 1ns / 1ps

import axi4stream_vip_pkg::*;
import design_1_axi4stream_vip_I_0_pkg::*;
import design_1_axi4stream_vip_Q_0_pkg::*;
import design_1_axi4stream_vip_fifo_demod_bit_0_pkg::*;

module tb_design_1;

    // Clock and Reset
    logic clk = 0;
    logic resetn = 0;
    
    // Test parameters
    parameter CLK_PERIOD = 10; // 10ns = 100 MHz
    parameter NUM_SAMPLES = 1000;
    parameter real FREQ = 0.01; // Normalized frequency
    
    // File handle
    integer output_file;
    
    // VIP Agents
    design_1_axi4stream_vip_I_0_mst_t axi_stream_i_agent;
    design_1_axi4stream_vip_Q_0_mst_t axi_stream_q_agent;
    design_1_axi4stream_vip_fifo_demod_bit_0_slv_t axi_stream_out_agent;
    
    // AXI transaction types
    axi4stream_transaction axis_i_trans;
    axi4stream_transaction axis_q_trans;
    axi4stream_transaction axis_out_trans;
    axi4stream_ready_gen axis_out_ready_gen;
    
    // DUT instance
    design_1_wrapper dut (
        .clock(clk),
        .reset(~resetn)
    );
    
    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Initialize VIP agents
    initial begin
        // Create AXI Stream I master agent
        axi_stream_i_agent = new("axi_stream_i_agent", dut.design_1_i.axi4stream_vip_I.inst.IF);
        
        // Create AXI Stream Q master agent
        axi_stream_q_agent = new("axi_stream_q_agent", dut.design_1_i.axi4stream_vip_Q.inst.IF);
        
        // Create AXI Stream output slave agent
        axi_stream_out_agent = new("axi_stream_out_agent", dut.design_1_i.axi4stream_vip_fifo_demod_bit.inst.IF);
        
        // DON'T start agents yet - wait for reset to complete
        
        $display("[%0t] VIP agents created (not started yet)", $time);
    end
    

    
    // AXI Stream I channel sender task
    task automatic send_i_stream();
        logic [15:0] sine_val;
        logic [1:0] keep_beat;
        real sine_real;
        int i;
        int signed sine_int;
        
        begin
            $display("[%0t] Starting I stream transmission (sine wave)...", $time);
            
            keep_beat = 2'b11; // Both bytes valid
            
            for (i = 0; i < NUM_SAMPLES; i++) begin
                // Generate sine wave
                sine_real = $sin(2.0 * 3.14159265359 * FREQ * real'(i));
                sine_int = int'($rtoi(sine_real * 32767.0));
                sine_val = logic'(sine_int[15:0]);
                
                // Create AXI Stream transaction
                axis_i_trans = axi_stream_i_agent.driver.create_transaction($sformatf("i_trans_%0d", i));
                axis_i_trans.set_data_beat(sine_val);
                axis_i_trans.set_last(i == (NUM_SAMPLES-1));
                axis_i_trans.set_keep_beat(keep_beat);
                
                // Send transaction
                axi_stream_i_agent.driver.send(axis_i_trans);
            end
            
            $display("[%0t] I stream transmission complete", $time);
        end
    endtask
    
    // AXI Stream Q channel sender task
    task automatic send_q_stream();
        logic [15:0] cosine_val;
        logic [1:0] keep_beat;
        real cosine_real;
        int i;
        int signed cosine_int;
        
        begin
            $display("[%0t] Starting Q stream transmission (cosine wave)...", $time);
            
            keep_beat = 2'b11; // Both bytes valid
            
            for (i = 0; i < NUM_SAMPLES; i++) begin
                // Generate cosine wave
                cosine_real = $cos(2.0 * 3.14159265359 * FREQ * real'(i));
                cosine_int = int'($rtoi(cosine_real * 32767.0));
                cosine_val = logic'(cosine_int[15:0]);
                
                // Create AXI Stream transaction
                axis_q_trans = axi_stream_q_agent.driver.create_transaction($sformatf("q_trans_%0d", i));
                axis_q_trans.set_data_beat(cosine_val);
                axis_q_trans.set_last(i == (NUM_SAMPLES-1));
                axis_q_trans.set_keep_beat(keep_beat);
                
                // Send transaction
                axi_stream_q_agent.driver.send(axis_q_trans);
            end
            
            $display("[%0t] Q stream transmission complete", $time);
        end
    endtask
    
    // AXI Stream output monitor task
    task automatic monitor_output_stream();
        logic [31:0] tdata;
        logic tlast;
        integer sample_count = 0;
        axi4stream_monitor_transaction mon_trans;
        
        begin
            // Open output file
            output_file = $fopen("output_stream.txt", "w");
            if (output_file == 0) begin
                $error("Failed to open output file!");
                $finish;
            end
            
            $display("[%0t] Starting output stream monitoring...", $time);
            
            fork
                begin
                    while (sample_count < NUM_SAMPLES) begin
                        // Get transaction from monitor
                        axi_stream_out_agent.monitor.item_collected_port.get(mon_trans);
                        
                        // Extract data
                        tdata = mon_trans.get_data_beat();
                        tlast = mon_trans.get_last();
                        
                        // Write to file
                        $fwrite(output_file, "%0d: 0x%08h (dec: %0d) %s\n", 
                                sample_count, tdata, $signed(tdata), tlast ? "LAST" : "");
                        
                        $display("[%0t] Output[%0d]: 0x%08h %s", 
                                 $time, sample_count, tdata, tlast ? "(LAST)" : "");
                        
                        sample_count++;
                        
                        if (tlast) break;
                    end
                    
                    $fclose(output_file);
                    $display("[%0t] Output monitoring complete. %0d samples saved to output_stream.txt", 
                             $time, sample_count);
                end
            join_none
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("========================================");
        $display("=== Starting AXI I/Q Stream Testbench ===");
        $display("========================================");
        
        // Wait for agents to be created
        wait(axi_stream_i_agent != null);
        wait(axi_stream_q_agent != null);
        wait(axi_stream_out_agent != null);
        
        $display("[%0t] Agents created", $time);
        
        // Hold reset (active low, so resetn=0 means reset asserted)
        resetn = 0;
        repeat(20) @(posedge clk);
        
        $display("[%0t] Releasing reset...", $time);
        
        // Release reset
        resetn = 1;
        
        // Wait for the actual signal change and stabilization
        #1; // Small delay to ensure signal has changed
        repeat(50) @(posedge clk);
        
        $display("[%0t] Reset released and stabilized, starting VIP agents", $time);
        
        // NOW start the agents after reset
        axi_stream_i_agent.start_master();
        axi_stream_q_agent.start_master();
        axi_stream_out_agent.start_slave();
        
        $display("[%0t] VIP agents started", $time);
        
        // Configure output stream to always be ready
        axis_out_ready_gen = axi_stream_out_agent.driver.create_ready("axis_out_ready");
        axis_out_ready_gen.set_ready_policy(XIL_AXI4STREAM_READY_GEN_SINGLE);
        axis_out_ready_gen.set_low_time(0);
        axis_out_ready_gen.set_high_time(1);
        axi_stream_out_agent.driver.send_tready(axis_out_ready_gen);
        
        $display("[%0t] Output ready signal configured", $time);
        
        repeat(10) @(posedge clk);
        
        // Start output monitoring
        monitor_output_stream();
        
        repeat(10) @(posedge clk);
        
        // Start I and Q stream transmission in parallel
        $display("[%0t] Starting I/Q stream transmission...", $time);
        fork
            send_i_stream();
            send_q_stream();
        join
        
        $display("[%0t] All input streams sent", $time);
        
        // Wait for output processing to complete
        repeat(NUM_SAMPLES + 200) @(posedge clk);
        
        $display("========================================");
        $display("=== Testbench Complete ===");
        $display("========================================");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * NUM_SAMPLES * 20);
        $error("Testbench timeout!");
        $finish;
    end

endmodule