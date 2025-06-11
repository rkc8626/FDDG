import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import matplotlib.ticker as mticker

def create_boxplot(df):
    plt.figure(figsize=(12, 6))
    accuracy_cols = ['env0_in_acc', 'env0_out_acc', 'env1_in_acc', 'env1_out_acc']
    df_melted = pd.melt(df, id_vars=['model'], value_vars=accuracy_cols,
                        var_name='Metric', value_name='Accuracy')
    sns.boxplot(x='model', y='Accuracy', hue='Metric', data=df_melted)
    plt.title('Distribution of Accuracy Metrics by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_boxplot.png')
    plt.close()

def create_heatmap(df):
    best_configs = df.groupby('model').apply(lambda x: x.loc[x['env1_out_acc'].idxmax()])
    out_metrics = ['env0_out_acc', 'env1_out_acc', 'env0_out_md', 'env1_out_md',
                   'env0_out_dp', 'env1_out_dp', 'env0_out_eo', 'env1_out_eo',
                   'env0_out_auc', 'env1_out_auc']
    plt.figure(figsize=(12, 8))
    sns.heatmap(best_configs[out_metrics], annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Best Configuration Performance by Model (Out-of-Distribution Metrics)')
    plt.tight_layout()
    plt.savefig('best_configs_heatmap.png')
    plt.close()

def create_scatter(df):
    plt.figure(figsize=(15, 12))

    # Define the pairs of metrics to plot
    metric_pairs = [
        ('env0_in_acc', 'env0_in_dp', 'Environment 0 In-Distribution'),
        ('env0_out_acc', 'env0_out_dp', 'Environment 0 Out-of-Distribution'),
        ('env1_in_acc', 'env1_in_dp', 'Environment 1 In-Distribution'),
        ('env1_out_acc', 'env1_out_dp', 'Environment 1 Out-of-Distribution')
    ]

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Use a more extensive and distinct color palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9896',  # light red
        '#98df8a',  # light green
        '#ffbb78',  # light orange
        '#aec7e8',  # light blue
        '#c5b0d5'   # light purple
    ]

    for idx, (acc_col, dp_col, title) in enumerate(metric_pairs):
        ax = axes[idx]
        for i, model in enumerate(df['model'].unique()):
            model_data = df[df['model'] == model]
            # Swap x and y axes, use distinct colors
            ax.scatter(model_data[dp_col], model_data[acc_col],
                      label=model, alpha=0.7, s=100, color=colors[i % len(colors)])

        ax.set_xlabel('Demographic Parity')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('Fairness vs Accuracy Trade-off by Environment and Split', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('accuracy_fairness_tradeoff.png', bbox_inches='tight')
    plt.close()

def create_lineplot(df):
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['lr'], model_data['env1_out_acc'],
                 marker='o', label=model)
    plt.xlabel('Learning Rate')
    plt.ylabel('Out-of-Distribution Accuracy')
    plt.title('Impact of Learning Rate on Model Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_rate_impact.png')
    plt.close()

def create_barplot(df):
    best_model = df.loc[df['env1_out_acc'].idxmax()]
    metrics = ['env0_in_acc', 'env0_out_acc', 'env1_in_acc', 'env1_out_acc']
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, best_model[metrics])
    plt.title(f'Best Model Performance ({best_model["model"]})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('best_model_performance.png')
    plt.close()

def create_violinplot(df):
    plt.figure(figsize=(12, 6))
    fairness_cols = ['env0_in_dp', 'env0_out_dp', 'env1_in_dp', 'env1_out_dp']
    df_melted = pd.melt(df, id_vars=['model'], value_vars=fairness_cols,
                        var_name='Metric', value_name='Fairness')
    sns.violinplot(x='model', y='Fairness', hue='Metric', data=df_melted)
    plt.title('Distribution of Fairness Metrics by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fairness_violin.png')
    plt.close()

def create_lineplot_by_bs(df):
    unique_bs = sorted(df['bs'].unique())
    n_bs = len(unique_bs)
    fig, axes = plt.subplots(1, n_bs, figsize=(7 * n_bs, 6), sharey=True)
    if n_bs == 1:
        axes = [axes]
    for i, bs in enumerate(unique_bs):
        ax = axes[i]
        bs_data = df[df['bs'] == bs]
        for model in bs_data['model'].unique():
            model_data = bs_data[bs_data['model'] == model]
            ax.plot(model_data['lr'], model_data['env1_out_acc'], marker='o', label=model)
        ax.set_title(f'Batch Size: {bs}')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Out-of-Distribution Accuracy')
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
    plt.suptitle('Impact of Learning Rate on Model Performance by Batch Size')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('learning_rate_impact_by_bs.png')
    plt.close()

def create_average_scatter(df):
    # Create two separate figures for in and out distribution
    fig_in = plt.figure(figsize=(12, 10))
    fig_out = plt.figure(figsize=(12, 10))

    # Calculate averages for each model
    model_averages = df.groupby('model').agg({
        'env0_in_acc': 'mean',
        'env0_out_acc': 'mean',
        'env1_in_acc': 'mean',
        'env1_out_acc': 'mean',
        'env0_in_dp': 'mean',
        'env0_out_dp': 'mean',
        'env1_in_dp': 'mean',
        'env1_out_dp': 'mean'
    })

    # Calculate average across environments
    model_averages['avg_in_acc'] = (model_averages['env0_in_acc'] + model_averages['env1_in_acc']) / 2
    model_averages['avg_out_acc'] = (model_averages['env0_out_acc'] + model_averages['env1_out_acc']) / 2
    model_averages['avg_in_dp'] = (model_averages['env0_in_dp'] + model_averages['env1_in_dp']) / 2
    model_averages['avg_out_dp'] = (model_averages['env0_out_dp'] + model_averages['env1_out_dp']) / 2

    # Use the same color palette as the other scatter plot
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9896',  # light red
        '#98df8a',  # light green
        '#ffbb78',  # light orange
        '#aec7e8',  # light blue
        '#c5b0d5'   # light purple
    ]

    # Plot in-distribution average
    plt.figure(fig_in.number)
    for i, (model, row) in enumerate(model_averages.iterrows()):
        plt.scatter(row['avg_in_dp'], row['avg_in_acc'],
                   color=colors[i % len(colors)],
                   s=150, alpha=0.7)

    plt.xlabel('Average Demographic Parity')
    plt.ylabel('Average Accuracy')
    plt.title('In-Distribution: Average Accuracy vs Fairness Trade-off by Model\n(Averaged across Environments)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add model color legend
    model_handles = []
    for i, model in enumerate(model_averages.index):
        model_handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)],
                                      label=model, linestyle='-'))
    plt.legend(handles=model_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('average_tradeoff_in_avg.png', bbox_inches='tight')
    plt.close(fig_in)

    # Plot out-of-distribution average
    plt.figure(fig_out.number)
    for i, (model, row) in enumerate(model_averages.iterrows()):
        plt.scatter(row['avg_out_dp'], row['avg_out_acc'],
                   color=colors[i % len(colors)],
                   s=150, alpha=0.7)

    plt.xlabel('Average Demographic Parity')
    plt.ylabel('Average Accuracy')
    plt.title('Out-of-Distribution: Average Accuracy vs Fairness Trade-off by Model\n(Averaged across Environments)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add model color legend
    model_handles = []
    for i, model in enumerate(model_averages.index):
        model_handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)],
                                      label=model, linestyle='-'))
    plt.legend(handles=model_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('average_tradeoff_out_avg.png', bbox_inches='tight')
    plt.close(fig_out)

def create_average_with_std(df):
    plt.figure(figsize=(10, 8))

    # Calculate means and standard deviations for each model
    stats = df.groupby('model').agg({
        'env0_in_acc': ['mean', 'std'],
        'env0_out_acc': ['mean', 'std'],
        'env1_in_acc': ['mean', 'std'],
        'env1_out_acc': ['mean', 'std'],
        'env0_in_dp': ['mean', 'std'],
        'env0_out_dp': ['mean', 'std'],
        'env1_in_dp': ['mean', 'std'],
        'env1_out_dp': ['mean', 'std']
    })

    # Calculate combined averages for in and out distributions
    stats['in_acc_mean'] = (stats[('env0_in_acc', 'mean')] + stats[('env1_in_acc', 'mean')]) / 2
    stats['out_acc_mean'] = (stats[('env0_out_acc', 'mean')] + stats[('env1_out_acc', 'mean')]) / 2
    stats['in_dp_mean'] = (stats[('env0_in_dp', 'mean')] + stats[('env1_in_dp', 'mean')]) / 2
    stats['out_dp_mean'] = (stats[('env0_out_dp', 'mean')] + stats[('env1_out_dp', 'mean')]) / 2

    # Calculate combined standard deviations
    stats['in_acc_std'] = np.sqrt((stats[('env0_in_acc', 'std')]**2 + stats[('env1_in_acc', 'std')]**2) / 2)
    stats['out_acc_std'] = np.sqrt((stats[('env0_out_acc', 'std')]**2 + stats[('env1_out_acc', 'std')]**2) / 2)
    stats['in_dp_std'] = np.sqrt((stats[('env0_in_dp', 'std')]**2 + stats[('env1_in_dp', 'std')]**2) / 2)
    stats['out_dp_std'] = np.sqrt((stats[('env0_out_dp', 'std')]**2 + stats[('env1_out_dp', 'std')]**2) / 2)

    # Use the same color palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9896',  # light red
        '#98df8a',  # light green
        '#ffbb78',  # light orange
        '#aec7e8',  # light blue
        '#c5b0d5'   # light purple
    ]

    # Create the plot
    plt.figure(figsize=(10, 8))

    for i, model in enumerate(stats.index):
        # In-distribution (averaged across environments)
        plt.errorbar(stats.loc[model, 'in_dp_mean'],
                    stats.loc[model, 'in_acc_mean'],
                    xerr=stats.loc[model, 'in_dp_std'],
                    yerr=stats.loc[model, 'in_acc_std'],
                    fmt='o', color=colors[i % len(colors)],
                    label=f'{model} (In)', alpha=0.7, capsize=5, markersize=10)

    # for i, model in enumerate(stats.index):
    #     # Out-of-distribution (averaged across environments)
    #     plt.errorbar(stats.loc[model, 'out_dp_mean'],
    #                 stats.loc[model, 'out_acc_mean'],
    #                 xerr=stats.loc[model, 'out_dp_std'],
    #                 yerr=stats.loc[model, 'out_acc_std'],
    #                 fmt='s', color=colors[i % len(colors)],
    #                 label=f'{model} (Out)', alpha=0.7, capsize=5, markersize=10)

    plt.ylim(0, 1)
    plt.xlabel('Demographic Parity')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy vs Fairness Trade-off\n(In Distribution Averages Across Environments)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create custom legend for markers (in/out distribution)
    marker_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', label='In-distribution', linestyle='None', markersize=10),
        plt.Line2D([0], [0], marker='s', color='gray', label='Out-of-distribution', linestyle='None', markersize=10)
    ]

    # Create custom legend for models (colors)
    model_handles = []
    for i, model in enumerate(stats.index):
        model_handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)],
                                      label=model, linestyle='-'))

    # Add legends
    plt.legend(handles=marker_handles, loc='lower left')
    plt.legend(handles=model_handles, bbox_to_anchor=(1.02, 0.5), loc='center left')

    plt.tight_layout()
    plt.savefig('average_with_std.png', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for experiment results')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--boxplot', action='store_true', help='Generate accuracy boxplot')
    parser.add_argument('--heatmap', action='store_true', help='Generate best configurations heatmap')
    parser.add_argument('--scatter', action='store_true', help='Generate accuracy vs fairness scatter plot')
    parser.add_argument('--lineplot', action='store_true', help='Generate learning rate impact line plot')
    parser.add_argument('--barplot', action='store_true', help='Generate best model performance bar plot')
    parser.add_argument('--violinplot', action='store_true', help='Generate fairness metrics violin plot')
    parser.add_argument('--lineplot_by_bs', action='store_true', help='Generate learning rate impact line plot faceted by batch size')
    parser.add_argument('--average_scatter', action='store_true', help='Generate average accuracy vs fairness trade-off plot')
    parser.add_argument('--average_with_std', action='store_true', help='Generate average accuracy vs fairness trade-off plot with standard deviation')

    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv('summary.csv')

    # Set the style
    plt.style.use('seaborn')
    sns.set_palette("husl")

    # Generate selected visualizations
    if args.all or args.boxplot:
        create_boxplot(df)
    if args.all or args.heatmap:
        create_heatmap(df)
    if args.all or args.scatter:
        create_scatter(df)
    if args.all or args.lineplot:
        create_lineplot(df)
    if args.all or args.barplot:
        create_barplot(df)
    if args.all or args.violinplot:
        create_violinplot(df)
    if args.all or args.lineplot_by_bs:
        create_lineplot_by_bs(df)
    if args.all or args.average_scatter:
        create_average_scatter(df)
    if args.all or args.average_with_std:
        create_average_with_std(df)

    print("Selected visualizations have been generated and saved as PNG files.")

if __name__ == "__main__":
    main()