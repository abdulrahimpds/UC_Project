"""
analyze test set results and generate visualizations.
performs statistical comparisons between models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse


def load_results(csv_path):
    """load results csv and organize by model type."""
    df = pd.read_csv(csv_path)
    
    # extract model type and run number
    df['model_type'] = df['model_name'].str.rsplit('_', n=1).str[0]
    df['run'] = df['model_name'].str.rsplit('_', n=1).str[1]
    
    return df


def perform_paired_ttest(df, model1, model2, metric='macro_iou'):
    """
    perform paired t-test between two models across runs.
    
    args:
        df: results dataframe
        model1, model2: model names (e.g., 'TSViT', 'TSViT-ST')
        metric: metric to compare
        
    returns:
        t_statistic, p_value, mean_diff, ci_lower, ci_upper
    """
    model1_values = df[df['model_type'] == model1][metric].values
    model2_values = df[df['model_type'] == model2][metric].values
    
    # ensure we have 3 runs for each
    assert len(model1_values) == 3, f"expected 3 runs for {model1}, got {len(model1_values)}"
    assert len(model2_values) == 3, f"expected 3 runs for {model2}, got {len(model2_values)}"
    
    # paired t-test
    t_stat, p_value = stats.ttest_rel(model1_values, model2_values)
    
    # mean difference
    diff = model1_values - model2_values
    mean_diff = np.mean(diff)
    
    # 95% confidence interval
    ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
    
    return t_stat, p_value, mean_diff, ci[0], ci[1]


def plot_model_comparison(df, output_dir):
    """create bar chart comparing model performance."""
    plt.figure(figsize=(12, 6))
    
    metrics = ['macro_iou', 'macro_f1', 'macro_accuracy', 'macro_precision', 'macro_recall']
    metric_labels = ['IoU', 'F1', 'Accuracy', 'Precision', 'Recall']
    
    # compute mean and std for each model
    summary = df.groupby('model_type')[metrics].agg(['mean', 'std'])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    models = ['TSViT', 'TSViT-ST', 'UNet3D']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        means = [summary.loc[model, (metric, 'mean')] for metric in metrics]
        stds = [summary.loc[model, (metric, 'std')] for metric in metrics]
        plt.bar(x + i*width, means, width, yerr=stds, label=model, 
                color=color, capsize=5, alpha=0.8)
    
    plt.xlabel('metric', fontsize=12)
    plt.ylabel('score', fontsize=12)
    plt.title('model performance comparison on spacenet7 test set', fontsize=14, fontweight='bold')
    plt.xticks(x + width, metric_labels)
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved plot: {output_path}")
    plt.close()


def plot_perclass_iou(df, output_dir):
    """create bar chart for per-class iou."""
    plt.figure(figsize=(10, 6))
    
    class_metrics = ['class_0_iou', 'class_1_iou']
    class_labels = ['background', 'building']
    
    summary = df.groupby('model_type')[class_metrics].agg(['mean', 'std'])
    
    x = np.arange(len(class_labels))
    width = 0.25
    
    models = ['TSViT', 'TSViT-ST', 'UNet3D']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        means = [summary.loc[model, (metric, 'mean')] for metric in class_metrics]
        stds = [summary.loc[model, (metric, 'std')] for metric in class_metrics]
        plt.bar(x + i*width, means, width, yerr=stds, label=model,
                color=color, capsize=5, alpha=0.8)
    
    plt.xlabel('class', fontsize=12)
    plt.ylabel('iou', fontsize=12)
    plt.title('per-class iou on spacenet7 test set', fontsize=14, fontweight='bold')
    plt.xticks(x + width, class_labels)
    plt.legend(loc='upper right', fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'perclass_iou.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved plot: {output_path}")
    plt.close()


def plot_run_variability(df, output_dir):
    """create box plot showing variability across runs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['macro_iou', 'macro_f1', 'macro_accuracy']
    titles = ['IoU', 'F1 Score', 'Accuracy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        data_to_plot = [df[df['model_type'] == model][metric].values 
                       for model in ['TSViT', 'TSViT-ST', 'UNet3D']]
        
        bp = ax.boxplot(data_to_plot, labels=['TSViT', 'TSViT-ST', 'UNet3D'],
                       patch_artist=True, showmeans=True)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('performance variability across 3 runs', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'run_variability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved plot: {output_path}")
    plt.close()


def generate_summary_table(df, output_dir):
    """generate latex-formatted summary table."""
    metrics = ['macro_iou', 'macro_f1', 'macro_accuracy', 'macro_precision', 'macro_recall']
    metric_names = ['IoU', 'F1', 'Accuracy', 'Precision', 'Recall']
    
    summary = df.groupby('model_type')[metrics].agg(['mean', 'std'])
    
    # format as mean ± std
    table_data = []
    for model in ['TSViT', 'TSViT-ST', 'UNet3D']:
        row = [model]
        for metric in metrics:
            mean = summary.loc[model, (metric, 'mean')]
            std = summary.loc[model, (metric, 'std')]
            row.append(f"{mean:.4f} ± {std:.4f}")
        table_data.append(row)
    
    # create dataframe
    result_df = pd.DataFrame(table_data, columns=['Model'] + metric_names)
    
    # save as csv
    csv_path = Path(output_dir) / 'summary_table.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"saved summary table: {csv_path}")
    
    # also save as latex
    latex_path = Path(output_dir) / 'summary_table.tex'
    with open(latex_path, 'w') as f:
        f.write(result_df.to_latex(index=False, escape=False))
    print(f"saved latex table: {latex_path}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description='analyze test results and generate visualizations')
    parser.add_argument('--input', default='results/test_metrics.csv',
                       help='path to test metrics csv')
    parser.add_argument('--output_dir', default='results/analysis',
                       help='directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load results
    print(f"\nloading results from {args.input}")
    df = load_results(args.input)
    print(f"loaded {len(df)} model results")
    print(f"\nmodel types: {df['model_type'].unique()}")
    
    # generate summary statistics
    print("\n" + "="*80)
    print("summary statistics (mean ± std across 3 runs)")
    print("="*80)
    summary_df = generate_summary_table(df, output_dir)
    print(summary_df.to_string(index=False))
    
    # perform statistical tests
    print("\n" + "="*80)
    print("statistical analysis: tsvit vs tsvit-st (paired t-test)")
    print("="*80)
    
    metrics_to_test = [
        ('macro_iou', 'IoU'),
        ('macro_f1', 'F1'),
        ('macro_accuracy', 'Accuracy'),
        ('macro_precision', 'Precision'),
        ('macro_recall', 'Recall')
    ]
    
    test_results = []
    for metric_key, metric_name in metrics_to_test:
        t_stat, p_value, mean_diff, ci_lower, ci_upper = perform_paired_ttest(
            df, 'TSViT', 'TSViT-ST', metric_key
        )
        
        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"\n{metric_name}:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f} {significant}")
        print(f"  mean difference (TSViT - TSViT-ST): {mean_diff:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        test_results.append({
            'Metric': metric_name,
            't-statistic': f"{t_stat:.4f}",
            'p-value': f"{p_value:.6f}",
            'Mean Diff': f"{mean_diff:.4f}",
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            'Significant': significant
        })
    
    # save test results
    test_df = pd.DataFrame(test_results)
    test_csv = output_dir / 'statistical_tests.csv'
    test_df.to_csv(test_csv, index=False)
    print(f"\nsaved statistical test results: {test_csv}")
    
    # generate visualizations
    print("\n" + "="*80)
    print("generating visualizations")
    print("="*80)
    
    plot_model_comparison(df, output_dir)
    plot_perclass_iou(df, output_dir)
    plot_run_variability(df, output_dir)
    
    print("\nanalysis complete!")
    print(f"all outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()