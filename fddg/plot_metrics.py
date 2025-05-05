import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('FDDG/fddg/train_output/tensorboard_results.csv')

# Get unique metrics (excluding environment and in/out indicators)
base_metrics = ['accuracy', 'max_difference', 'demographic_parity', 'equalized_odds', 'auc_score']
environments = ['env0', 'env1', 'env2']
distributions = ['in', 'out']

# Create subplots for each metric
fig, axes = plt.subplots(len(base_metrics), 1, figsize=(15, 5*len(base_metrics)))
fig.suptitle('Training Metrics Over Time', fontsize=16, y=0.95)

# Plot each metric
for idx, metric in enumerate(base_metrics):
    ax = axes[idx]

    # Plot for each environment and distribution
    for env in environments:
        for dist in distributions:
            metric_name = f"{env}_{dist}_{metric}"
            if metric_name in df['metric'].unique():
                data = df[df['metric'] == metric_name]
                ax.plot(data['step'], data['value'],
                       label=f"{env} ({dist})",
                       marker='o', markersize=2)

    ax.set_title(f'{metric.replace("_", " ").title()}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('training_metrics.png', bbox_inches='tight', dpi=300)
plt.close()