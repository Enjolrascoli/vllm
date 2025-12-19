import subprocess
import argparse
import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Sweep RPS test")
    parser.add_argument("--rps-list", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0], help="List of RPS values to test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--output-dir", type=str, default="results_final", help="Output directory")
    parser.add_argument("--input-data", type=str, default="ShareGPT_V3_unfiltered_cleaned_split.json", help="Input data for trace generation")
    parser.add_argument("--mem-util", type=float, default=0.90, help="server memory utilization")
    parser.add_argument("--num-trace", type=int, default=300, help="number of trace entries to use")
    parser.add_argument("--max-concurrency-list", nargs="+", type=int, default=[5, 10, 15, 20], help="maximum number of concurrent requests")
    parser.add_argument("--follow-timestamp", action='store_true', help="whether to follow timestamp")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("trace", exist_ok=True)
    
    ports = [40001, 40002]

    # Get absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sharegpt_script = os.path.join(base_dir, "sharegpt_to_belady.py")
    replay_script = os.path.join(base_dir, "replay_trace_stat.py")
    process_script = os.path.join(base_dir, "process_stats.py")

    result_dir = os.path.join(base_dir, args.output_dir)
    trace_dir = os.path.join(base_dir, "trace")
    
    # Check if input data exists
    if not os.path.exists(args.input_data):
        # Try to find it in the base_dir
        potential_path = os.path.join(base_dir, args.input_data)
        if os.path.exists(potential_path):
            args.input_data = potential_path
        else:
            print(f"Warning: Input data {args.input_data} not found.")

    items = args.rps_list if args.follow_timestamp else args.max_concurrency_list
    for item in items:
        if args.follow_timestamp:
            print(f"\n=== Testing RPS: {item} ===")
            rps = item
            load_string = f"rps_{item}"
        else:
            print(f"\n=== Testing Max Concurrency: {item} ===")
            rps = 5.0
            load_string = f"concurrency_{item}"
        trace_file = os.path.join(trace_dir, f"trace_rps_{rps}.jsonl")
        output_subdir = os.path.join(result_dir, f"{load_string}_memutil_{args.mem_util}_{args.num_trace}_trace")
        os.makedirs(output_subdir, exist_ok=True)
        
        # 1. Generate trace
        cmd_gen = (
            f"python3 {sharegpt_script} "
            f"--input {args.input_data} "
            f"--output {trace_file} "
            f"--request-rate {rps} "
            f"--model {args.model} "
        )
        run_command(cmd_gen)

        # 2. Replay trace
        processes = []
        for port in ports:
            cmd_replay = (
                f"python3 {replay_script} "
                f"--trace-file {trace_file} "
                f"--port {port} "
                f"--model {args.model} "
                f"--max-tokens 4096 "
                f"--output-dir {output_subdir} "
                f"--output-suffix {"belady" if port == 40002 else "lru"} "
                f"{'--follow-timestamp' if args.follow_timestamp else ''} "
                f"--num-trace {args.num_trace} "
                f"--max-concurrent-requests {item if not args.follow_timestamp else 1000} "
            )
            print(f"Starting on port {port}: {cmd_replay}")
            p = subprocess.Popen(cmd_replay, shell=True)
            processes.append(p)
        
        for p in processes:
            p.wait()

        # 3. Process results
        print(f"\n=== Processing Results ===")
        cmd_process = (
            f"python3 {process_script} "
            f"--dir {output_subdir} "
            f"--output-prefix {load_string}_memutil_{args.mem_util} "
        )
        run_command(cmd_process)

if __name__ == "__main__":
    main()
