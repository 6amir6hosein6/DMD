import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_lsar_and_loop_overhead():
    # List of files with corresponding batch and worker values
    configs = [
        ('runtime-size-sorted/execution_times_b256_w8.csv', 256, 8, "Name_Sorted"),
        ('runtime-size-sorted/execution_times_b256_w8_sorted.csv', 256, 8, "Size_Sorted"),
    ]
    
    # Read all CSV files into dataframes
    dfs = []
    for file_path, batch, worker, is_sorted in configs:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Columns to compare
    columns = ['lsar_score_torchB', 'loop_overhead', 'calculate_score_torchB']
    
    # Compute means for each column and configuration
    means = {col: [] for col in columns}
    for df in dfs:
        for col in columns:
            mean_value = df[col].mean()
            means[col].append(mean_value)
    
    # X positions for configurations
    x_positions = np.arange(len(configs))
    config_labels = [f'b{b}_w{w}_{s}' for _, b, w, s in configs]
    
    # Create the plot with a single y-axis (auto-scaling)
    plt.figure(figsize=(10, 6))
    
    # Define colors
    colors = ["#FF0000", "#1900FF", "#00FF0D"]
    
    # Plot each column as a single line
    plt.plot(x_positions, means['lsar_score_torchB'], color=colors[0], marker='o', linestyle='-', linewidth=2, label='lsar_score_torchB')
    plt.plot(x_positions, means['loop_overhead'], color=colors[1], marker='o', linestyle='-', linewidth=2, label='loop_overhead')
    plt.plot(x_positions, means['calculate_score_torchB'], color=colors[2], marker='o', linestyle='-', linewidth=2, label='calculate_score_torchB')
    
    # Set x-ticks and labels
    plt.xticks(x_positions, config_labels)
    plt.xlabel('Batch-Worker Configuration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparison of lsar_score_torchB, loop_overhead, and calculate_score_torchB')
    plt.legend(title='Metrics')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('runtime-size-sorted/comparison_plot.png', bbox_inches='tight')
    plt.close()

# Run the function
if __name__ == "__main__":
    compare_lsar_and_loop_overhead()