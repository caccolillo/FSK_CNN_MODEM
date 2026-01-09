import numpy as np
import vart
import xir
import os
import time
import random

# FSK Parameters (Matching your training script)
F_SAMPLING = 8e6
F_DEVIATION = 1e6
SAMPLES_PER_SYMBOL = 8

def generate_iq_symbol(bit):
    """
    Generates I/Q samples for a single bit.
    Returns shape (2, 8) -> Row 0: I, Row 1: Q
    """
    t = np.arange(SAMPLES_PER_SYMBOL)
    # Bit 0: -1MHz shift, Bit 1: +1MHz shift
    phase_shift = 2 * np.pi * (F_DEVIATION if bit == 1 else -F_DEVIATION) / F_SAMPLING
    
    I = np.cos(phase_shift * t)
    Q = np.sin(phase_shift * t)
    
    # Stack into (2, 8)
    symbol = np.stack([I, Q], axis=0).astype(np.float32)
    return symbol

def get_child_subgraph_dpu(graph):
    """Helper to find the DPU subgraph in the xmodel"""
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.get_children()
    dpu_subgraphs = [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    return dpu_subgraphs[0]

def run_fsk_test_bench(model_path, num_bits=10):
    # 1. Load the Graph and Create the DPU Runner
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return

    graph = xir.Graph.deserialize(model_path)
    subgraph = get_child_subgraph_dpu(graph)
    runner = vart.Runner.create_runner(subgraph, "run")

    # 2. Setup Input/Output Buffers
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    # Shapes on Ultra96 DPU are usually:
    # Input:  [1, 8, 2] (Batch, Length, Channels)
    # Output: [1, 2]    (Batch, Classes)
    in_shape = tuple(input_tensors[0].dims)
    out_shape = tuple(output_tensors[0].dims)

    input_data = [np.empty(in_shape, dtype=np.float32, order="C")]
    output_data = [np.empty(out_shape, dtype=np.float32, order="C")]

    # 3. Prepare Test Data
    test_bits = [random.randint(0, 1) for _ in range(num_bits)]
    results = []
    latencies = []

    print(f"\n{'='*60}")
    print(f" DPU FSK TEST BENCH - ULTRA96 V2")
    print(f"{'='*60}")
    print(f"{'Bit #':<8} {'Target':<10} {'DPU Output (0, 1)':<20} {'Prediction':<12} {'Status'}")
    print(f"{'-'*60}")

    # 4. Run Inference Loop
    for i, bit in enumerate(test_bits):
        # Generate signal (2, 8)
        symbol = generate_iq_symbol(bit)
        
        # FIX: Transpose (2, 8) to (8, 2) to match DPU hardware layout
        input_data[0][0] = symbol.transpose(1, 0)

        # Time the DPU execution
        start = time.perf_counter()
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)
        end = time.perf_counter()
        
        # Calculate result
        raw_scores = output_data[0][0]
        prediction = np.argmax(raw_scores)
        status = "✓ PASS" if prediction == bit else "✗ FAIL"
        
        results.append(prediction)
        latencies.append((end - start) * 1_000_000) # Microseconds
        
        # Print row
        scores_str = f"({raw_scores[0]:.2f}, {raw_scores[1]:.2f})"
        print(f"{i:<8} {bit:<10} {scores_str:<20} {prediction:<12} {status}")

    # 5. Print Stats
    accuracy = (sum(1 for i in range(num_bits) if results[i] == test_bits[i]) / num_bits) * 100
    avg_latency = sum(latencies) / num_bits
    
    print(f"{'-'*60}")
    print(f" STATS SUMMARY")
    print(f"{'-'*60}")
    print(f" Total Bits Processed: {num_bits}")
    print(f" Accuracy:             {accuracy:.2f}%")
    print(f" Average Latency:      {avg_latency:.2f} us")
    print(f" Theoretical Speed:    {1000000/avg_latency:.2f} bits/sec")
    print(f"{'='*60}\n")

    del runner

if __name__ == "__main__":
    # Ensure this matches the filename you moved to the board
    MODEL_FILE = "fsk_cnn_u96v2.xmodel"
    run_fsk_test_bench(MODEL_FILE, num_bits=10)
