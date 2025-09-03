import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_execution_times():
    # List of files with corresponding batch and worker values
    configs = [
        ('runtime/execution_times_b64_w8.csv', 64, 8),
        ('runtime/execution_times_b128_w8.csv', 128, 8),
        ('runtime/execution_times_b256_w8.csv', 256, 8),
        ('runtime/execution_times_b512_w8.csv', 512, 8),
    ]
    
    # Read all CSV files into dataframes
    dfs = []
    for file_path, batch, worker in configs:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Get columns from the first dataframe
    columns = dfs[0].columns
    
    # Define a color palette for consistency
    colors = ["#FF0000", "#1900FF", "#00FF0D", "#FBFF00"]  # Distinct colors for dark/light themes
    
    # For each column, create a separate normalized dot plot
    for col in columns:
        plt.figure(figsize=(10, 6))
        
        # Collect all values for this column across all files
        all_values = np.concatenate([df[col].values for df in dfs])
        max_value = all_values.max()  # Maximum value for normalization
        
        # Plot data from each file
        for i, (df, (_, batch, worker)) in enumerate(zip(dfs, configs)):
            y_values = df[col].values / max_value  # Normalize by max value
            x_values = np.full(len(y_values), i) + np.random.uniform(-0.2, 0.2, len(y_values))
            plt.scatter(x_values, y_values, label=f'b{batch}_w{worker}', alpha=0.7, color=colors[i])
            
            # Add dashed line for the mean
            mean_value = np.mean(y_values)
            plt.axhline(y=mean_value, color=colors[i], linestyle='--', linewidth=1, alpha=0.7)
        
        # Set x-ticks to batch-worker combinations
        config_labels = [f'b{b}_w{w}' for _, b, w in configs]
        plt.xticks(range(len(configs)), config_labels)
        
        # Set title, labels, and legend
        plt.title(col)
        plt.xlabel('Batch-Worker Configuration')
        plt.ylabel('Normalized Execution Time (0 to 1)')
        plt.legend(title='Tests')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot with sanitized column name
        safe_col_name = col.replace('/', '_').replace('\\', '_')
        plt.savefig(f'runtime/{safe_col_name}_plot.png', bbox_inches='tight')
        plt.close()

# Run the function
if __name__ == "__main__":
    visualize_execution_times()