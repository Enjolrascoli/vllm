import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_data(file_path):
    """Reads data from CSV or JSONL file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.jsonl') or file_path.endswith('.json'):
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
    parser = argparse.ArgumentParser(description="Plot CDF of TTFT from benchmark result files.")
    parser.add_argument("--files", nargs='+', required=True, help="List of result files (CSV or JSONL)")
    parser.add_argument("--labels", nargs='+', help="List of labels for the files. If not provided, filenames will be used.")
    parser.add_argument("--output", default="ttft_cdf_comparison.png", help="Output image filename")
    
    args = parser.parse_args()
    
    if args.labels and len(args.files) != len(args.labels):
        print("Error: Number of labels must match number of files.")
        return

    data_dict = {}
    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
            
        try:
            df = read_data(file_path)
            label = args.labels[i] if args.labels else os.path.basename(file_path)
            data_dict[label] = df
            print(f"Loaded {len(df)} records from {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not data_dict:
        print("No data loaded. Exiting.")
        return

    plot_cdf(data_dict, args.output)

if __name__ == "__main__":
    main()
