import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

def read_data(file_path):
    """Reads data from CSV or JSONL file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.jsonl') or file_path.endswith('.json'):
        # Check if it's a summary JSON (single object) or JSONL (lines)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict): # Summary file
                return data
            else:
                return pd.read_json(file_path, lines=True)
        except:
             return pd.read_json(file_path, lines=True)
    else:
        # Try CSV as fallback
        try:
            return pd.read_csv(file_path)
        except:
            raise ValueError(f"Unsupported file format: {file_path}")

def plot_cdf(data_dict, output_file="ttft_cdf_comparison.png"):
    """Plots CDF of TTFT for multiple datasets."""
    plt.figure(figsize=(10, 6))
    
    for label, df in data_dict.items():
        if not isinstance(df, pd.DataFrame): continue # Skip summary dicts

        if 'ttft_ms' not in df.columns:
            print(f"Warning: 'ttft_ms' column not found in data for {label}. Skipping.")
            continue
            
        ttft_values = df['ttft_ms'].dropna().values
        if len(ttft_values) == 0:
            print(f"Warning: No valid TTFT data for {label}. Skipping.")
            continue

        sorted_ttft = np.sort(ttft_values)
        yvals = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
        
        # Use step plot or markers
        plt.plot(sorted_ttft, yvals, marker='.', linestyle='-', label=f"{label} (n={len(sorted_ttft)})", markersize=3, alpha=0.7)

    plt.xlabel('Time to First Token (TTFT) [ms]')
    plt.ylabel('CDF')
    plt.title('CDF of Time to First Token (TTFT) Comparison')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--dir", help="Directory containing result files to process")
    parser.add_argument("--csv-files", nargs='+', help="List of result CSV files for CDF")
    parser.add_argument("--json-files", nargs='+', help="List of summary JSON files for bar charts")
    parser.add_argument("--labels", nargs='+', help="List of labels for the input files")
    parser.add_argument("--output-prefix", default="benchmark_plot", help="Prefix for output images")
    
    args = parser.parse_args()
    
    csv_data = {}
    summary_data = {}
    
    # Collect all files and assign labels
    all_files = []
    
    if args.dir:
        # Automatically find JSON and CSV files in the directory
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found.")
            return
            
        # Find all summary JSON files
        json_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.startswith("summary_") and f.endswith(".json")]
        all_files.extend([(f, 'json') for f in json_files])
        
        # Set output prefix to be inside the directory
        args.output_prefix = os.path.join(args.dir, args.output_prefix)
        print(f"Processing results in {args.dir}")
    
    if args.json_files: all_files.extend([(f, 'json') for f in args.json_files])
    if args.csv_files: all_files.extend([(f, 'csv') for f in args.csv_files])
    
    if args.labels:
        if len(args.labels) != len(all_files):
             print("Error: Number of labels must match total number of files (csv + json).")
             # Fallback to filenames
             labels = [os.path.splitext(os.path.basename(f[0]))[0].rsplit('_', 1)[-1] for f in all_files]
        else:
            labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(f[0]))[0].rsplit('_', 1)[-1] for f in all_files]    # Load data

    for i, (file_path, ftype) in enumerate(all_files):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
        
        try:
            data = read_data(file_path)
            label = labels[i]
            
            if ftype == 'csv':
                csv_data[label] = data
            elif ftype == 'json':
                # If user passed a label, use it. Otherwise try to get instance name from JSON
                if not args.labels and isinstance(data, dict) and 'instance' in data:
                     label = data['instance']
                summary_data[label] = data
                if args.csv_files is None:  # If only JSON files provided, add to CSV data for plotting
                    csv_data[label] = read_data(str(Path(file_path).with_name(data['stats_file'])))
            print(f"Loaded data from {file_path} as '{label}'")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Plot TTFT CDF
    if csv_data:
        plot_cdf(csv_data, f"{args.output_prefix}_ttft_cdf.png")
        
    # Calculate throughput
    for label, summary in summary_data.items():
        try:
            data = csv_data.get(label)
            total_input_tokens = data["input_num_tokens"].sum()
            total_output_tokens = data["output_num_tokens"].sum()
            total_tokens = total_input_tokens + total_output_tokens
            duration_str = summary.get('duration', '1s')
            total_time_sec = float(duration_str.rstrip('s'))
            throughput = total_tokens / total_time_sec
            summary['total_throughput'] = throughput
            #print(f"Throughput for {label}: {throughput:.2f} tokens/sec")

            if "num_cached_tokens" in data.columns:
                total_cached_tokens = data["num_cached_tokens"].sum()
                reuse_ratio = total_cached_tokens / total_input_tokens if total_input_tokens > 0 else 0.0
                total_provided_input_tokens = data["provided_input_num_tokens"].sum()
                provided_reuse_ratio = total_cached_tokens / total_provided_input_tokens if total_provided_input_tokens > 0 else 0.0
                print(f"Cache Reuse Ratio for {label}: \33[34m{reuse_ratio:.4f}\33[0m")
                print(f"Provided Cache Reuse Ratio for {label}: \33[34m{provided_reuse_ratio:.4f}\33[0m")
                print(f"total_cached_tokens: \33[34m{total_cached_tokens}\33[0m, total_input_tokens: \33[34m{total_input_tokens}\33[0m, total_provided_input_tokens: \33[34m{total_provided_input_tokens}\33[0m")
        except Exception as e:
            print(f"Error calculating throughput for {label}: {e}")
        

if __name__ == "__main__":
    main()
