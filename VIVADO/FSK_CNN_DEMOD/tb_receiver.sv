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
    parameter NUM_TEST_BITS = 100;
    parameter SAMPLES_PER_SYMBOL = 8;
    parameter real PI = 3.14159265359;
    
    // File handle
    integer output_file;
    
    // Test statistics
    integer correct_count = 0;
    integer total_bits = 0;
    
    // VIP Agents
    design_1_axi4stream_vip_I_0_mst_t axi_stream_i_agent;
    design_1_axi4stream_vip_Q_0_mst_t axi_stream_q_agent;
    design_1_axi4stream_vip_fifo_demod_bit_0_slv_t axi_stream_out_agent;
    
    // AXI transaction types
    axi4stream_transaction axis_i_trans;
    axi4stream_transaction axis_q_trans;
    axi4stream_ready_gen axis_out_ready_gen;
    
    // DUT instance
    design_1_wrapper dut (
        .clock(clk),
        .reset(resetn)
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
        
        $display("[%0t] VIP agents created", $time);
        
        // Wait for reset to be released
        wait(resetn == 0); // Wait for reset assertion
        wait(resetn == 1); // Wait for reset deassertion
        repeat(50) @(posedge clk); // Stabilization delay
        
        $display("[%0t] Starting VIP agents after reset...", $time);
        
        // Start the agents
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
        
        $display("[%0t] Output ready configured", $time);
    end
    
    // Task to send a single FSK symbol (I and Q channels in parallel)
    task automatic send_fsk_symbol(
        input integer test_bit,
        input integer test_num
    );
        real I_samples[SAMPLES_PER_SYMBOL];
        real Q_samples[SAMPLES_PER_SYMBOL];
        logic signed [15:0] I_fixed[SAMPLES_PER_SYMBOL];
        logic signed [15:0] Q_fixed[SAMPLES_PER_SYMBOL];
        logic [1:0] keep_beat;
        real phase_shift;
        integer i;
        
        begin
            // Calculate phase shift based on bit value
            // Matches Python/C++: phase_shift = 2*pi*(1 if bit==1 else -1)/SAMPLES_PER_SYMBOL
            if (test_bit == 1) begin
                phase_shift = 2.0 * PI * 1.0 / SAMPLES_PER_SYMBOL;  // 2*pi/8 for bit 1
            end else begin
                phase_shift = 2.0 * PI * (-1.0) / SAMPLES_PER_SYMBOL;  // -2*pi/8 for bit 0
            end
            
            $display("[%0t] Test #%0d - Bit = %0d", $time, test_num + 1, test_bit);
            $display("-------------------------------------------------");
            $display("Generated I/Q samples:");
            
            // Generate I and Q samples
            keep_beat = 2'b11; // Both bytes valid
            
            for (i = 0; i < SAMPLES_PER_SYMBOL; i++) begin
                I_samples[i] = $cos(phase_shift * i);
                Q_samples[i] = $sin(phase_shift * i);
                
                // Scale float to 16-bit fixed point (multiply by 2^7 = 128 for Q8.7)
                I_fixed[i] = $rtoi(I_samples[i] * 128.0);
                Q_fixed[i] = $rtoi(Q_samples[i] * 128.0);
                
                $display("  t=%0d: I=%f (0x%04h), Q=%f (0x%04h)", 
                         i, I_samples[i], I_fixed[i], Q_samples[i], Q_fixed[i]);
            end
            
            // Send I and Q samples in parallel
            fork
                begin : send_i
                    for (int j = 0; j < SAMPLES_PER_SYMBOL; j++) begin
                        axis_i_trans = axi_stream_i_agent.driver.create_transaction(
                            $sformatf("i_trans_test%0d_sample%0d", test_num, j)
                        );
                        axis_i_trans.set_data_beat(I_fixed[j]);
                        axis_i_trans.set_last(j == (SAMPLES_PER_SYMBOL - 1));
                        axis_i_trans.set_keep_beat(keep_beat);
                        axi_stream_i_agent.driver.send(axis_i_trans);
                    end
                end
                
                begin : send_q
                    for (int j = 0; j < SAMPLES_PER_SYMBOL; j++) begin
                        axis_q_trans = axi_stream_q_agent.driver.create_transaction(
                            $sformatf("q_trans_test%0d_sample%0d", test_num, j)
                        );
                        axis_q_trans.set_data_beat(Q_fixed[j]);
                        axis_q_trans.set_last(j == (SAMPLES_PER_SYMBOL - 1));
                        axis_q_trans.set_keep_beat(keep_beat);
                        axi_stream_q_agent.driver.send(axis_q_trans);
                    end
                end
            join
            
            $display("[%0t] FSK symbol transmission complete", $time);
        end
    endtask
    
    // Task to receive and check predicted bit
    task automatic check_predicted_bit(
        input integer expected_bit,
        input integer test_num
    );
        axi4stream_monitor_transaction mon_trans;
        logic [31:0] tdata;
        logic tlast;
        integer predicted_bit;
        
        begin
            // Wait for output transaction
            axi_stream_out_agent.monitor.item_collected_port.get(mon_trans);
            
            // Extract data
            tdata = mon_trans.get_data_beat();
            tlast = mon_trans.get_last();
            predicted_bit = tdata[0];
            
            // Check prediction
            $display("Actual bit:    %0d", expected_bit);
            $display("Predicted bit: %0d (tlast=%0d, tdata=0x%08h)", 
                     predicted_bit, tlast, tdata);
            
            if (predicted_bit == expected_bit) begin
                $display("Result: PASS");
                correct_count++;
            end else begin
                $display("Result: FAIL");
            end
            
            $display("");
            total_bits++;
        end
    endtask
    
    // Main test sequence
    initial begin
        integer test_bit;
        integer test_num;
        //integer seed = 42; // Fixed seed for reproducibility
        
        $display("=================================================");
        $display("FSK CNN DEMODULATOR - SystemVerilog TESTBENCH");
        $display("Testing with exact Python/C++ FSK generation");
        $display("=================================================");
        $display("");
        
        // Open output file
        output_file = $fopen("output_results.txt", "w");
        if (output_file == 0) begin
            $error("Failed to open output file!");
            $finish;
        end
        
        $fwrite(output_file, "FSK Demodulator Test Results\n");
        $fwrite(output_file, "=================================================\n\n");
        
        // Hold reset (active low)
        resetn = 0;
        $display("[%0t] Reset asserted", $time);
        repeat(20) @(posedge clk);
        
        // Release reset
        resetn = 1;
        $display("[%0t] Reset released", $time);
        
        // Wait for agents to be fully started
        wait(axi_stream_i_agent != null);
        wait(axi_stream_q_agent != null);
        wait(axi_stream_out_agent != null);
        
        // Extra wait for initialization
        repeat(100) @(posedge clk);
        
        $display("[%0t] Starting FSK demodulation tests...", $time);
        $display("");
        
        // Test NUM_TEST_BITS random bits
        for (test_num = 0; test_num < NUM_TEST_BITS; test_num++) begin
            // Generate random bit (0 or 1) with fixed seed
            //test_bit = $urandom(seed) % 2;
            test_bit = $urandom % 2;
           
            // Send FSK symbol
            send_fsk_symbol(test_bit, test_num);
            
            // Allow processing time
            repeat(20) @(posedge clk);
            
            // Check predicted output
            fork
                check_predicted_bit(test_bit, test_num);
            join
            
            // Log to file
            $fwrite(output_file, "Test #%0d: Expected=%0d, Result=%s\n", 
                    test_num + 1, test_bit, 
                    (total_bits > 0 && correct_count == total_bits) ? "PASS" : "FAIL");
            
            // Small delay between tests
            repeat(10) @(posedge clk);
        end
        
        // Print summary
        $display("=================================================");
        $display("TEST SUMMARY");
        $display("=================================================");
        $display("Total bits tested: %0d", total_bits);
        $display("Correct predictions: %0d", correct_count);
        $display("Incorrect predictions: %0d", total_bits - correct_count);
        $display("Accuracy: %0.1f%%", (100.0 * correct_count / total_bits));
        $display("=================================================");
        
        if (correct_count == total_bits) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED");
        end
        $display("=================================================");
        
        // Write summary to file
        $fwrite(output_file, "\n=================================================\n");
        $fwrite(output_file, "SUMMARY\n");
        $fwrite(output_file, "=================================================\n");
        $fwrite(output_file, "Total: %0d, Correct: %0d, Accuracy: %0.1f%%\n", 
                total_bits, correct_count, (100.0 * correct_count / total_bits));
        $fwrite(output_file, "Status: %s\n", 
                (correct_count == total_bits) ? "ALL PASSED" : "SOME FAILED");
        
        $fclose(output_file);
        $display("Results saved to output_results.txt");
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * NUM_TEST_BITS * SAMPLES_PER_SYMBOL * 100);
        $error("Testbench timeout!");
        $finish;
    end

endmodule