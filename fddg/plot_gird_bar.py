import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the JSON file
with open('grid_search_results/all_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame with all hyperparameters
df_rows = []
for result in results:
    row = {
        'Algorithm': result['algorithm'],
        'Metric': 'Accuracy',
        'Value': result['metrics']['accuracy']
    }
    # Add all hyperparameters
    for param, value in result['hparams'].items():
        row[param] = value
    df_rows.append(row)

    # Add other metrics
    for metric in ['demographic_parity', 'equalized_odds', 'combined_score']:
        row = row.copy()
        row['Metric'] = metric.replace('_', ' ').title()
        row['Value'] = result['metrics'][metric]
        df_rows.append(row)

df = pd.DataFrame(df_rows)

# Create separate plots for each algorithm and metric
algorithms = df['Algorithm'].unique()
metrics = df['Metric'].unique()

plt.figure(figsize=(20, 5 * len(algorithms)))

plot_idx = 1
for algorithm in algorithms:
    for metric in metrics:
        plt.subplot(len(algorithms), len(metrics), plot_idx)

        # Filter data for current algorithm and metric
        mask = (df['Algorithm'] == algorithm) & (df['Metric'] == metric)
        data = df[mask].copy()

        # Create x-axis label combining all hyperparameters
        data['Hyperparameters'] = data.apply(lambda row: '\n'.join([
            f"lr={row['lr']}",
            f"bs={row['batch_size']}",
            f"wd={row['weight_decay']}"
        ] + (
            [f"λ={row['irm_lambda']}", f"anneal={row['irm_penalty_anneal_iters']}"] if algorithm == 'IRM'
            else [f"η={row['groupdro_eta']}"] if algorithm == 'GroupDRO'
            else []
        )), axis=1)

        # Create bar plot
        sns.barplot(data=data, x='Hyperparameters', y='Value')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{algorithm} - {metric}')
        plt.ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 for better visualization

        plot_idx += 1

plt.tight_layout()
plt.savefig('grid_search_results/grid_visualization.png', dpi=300, bbox_inches='tight')