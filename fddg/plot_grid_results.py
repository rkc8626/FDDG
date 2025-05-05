import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set style for better visualization
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#E5E5E5'
})

# Read the JSON file
with open('grid_search_results/all_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
plot_data = []
for result in results:
    # Create hyperparameter string with better formatting
    base_params = [
        f"lr={result['hparams']['lr']}",
        f"bs={result['hparams']['batch_size']}",
        f"wd={result['hparams']['weight_decay']}"
    ]

    if result['algorithm'] == 'IRM':
        base_params.extend([
            f"λ={result['hparams']['irm_lambda']}",
            f"anneal={result['hparams']['irm_penalty_anneal_iters']}"
        ])
    elif result['algorithm'] == 'GroupDRO':
        base_params.append(f"η={result['hparams']['groupdro_eta']}")

    # Join with newlines and add extra spacing for readability
    hparams = '\n'.join(base_params)

    for metric in ['accuracy', 'demographic_parity', 'equalized_odds', 'combined_score']:
        plot_data.append({
            'Algorithm': result['algorithm'],
            'Hyperparameters': hparams,
            'Metric': metric.replace('_', ' ').title(),
            'Value': result['metrics'][metric]
        })

df = pd.DataFrame(plot_data)

# Create figure with increased width
plt.figure(figsize=(30, 16))

# Create a subplot for each metric
metrics = df['Metric'].unique()
for i, metric in enumerate(metrics, 1):
    ax = plt.subplot(2, 2, i)

    # Create pivot table for heatmap
    pivot_data = df[df['Metric'] == metric].pivot(
        index='Algorithm',
        columns='Hyperparameters',
        values='Value'
    )

    # Create heatmap with refined parameters
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={
            'label': metric,
            'pad': 0.01,
            'aspect': 40
        },
        square=True,
        annot_kws={
            'size': 9,
            'weight': 'bold'
        },
        linewidths=1,
        linecolor='white'
    )

    # Enhance title and labels
    plt.title(f'{metric}', pad=20, fontweight='bold')
    plt.xlabel('Hyperparameters', labelpad=15)
    plt.ylabel('Algorithm', labelpad=15)

    # Adjust tick labels with more space
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0, va='center')

    # Add box around the plot
    ax.set_frame_on(True)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.axhline(y=pivot_data.shape[0], color='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.axvline(x=pivot_data.shape[1], color='black', linewidth=1.5)

    # Adjust bottom margin to accommodate labels
    plt.subplots_adjust(bottom=0.2)

# Adjust layout with more space for labels
plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=4.0, rect=[0, 0.05, 1, 0.95])

# Save with enhanced quality
plt.savefig(
    'grid_search_results/grid_visualization.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close()