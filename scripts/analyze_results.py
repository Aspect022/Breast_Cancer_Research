"""
Results Analysis Script for Breast Cancer Classification Project.

Generates:
- Statistical significance testing (p-values, effect sizes)
- Comparison tables
- Performance rankings
- Ablation study summaries
- Visualization plots

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --output-dir outputs
    python scripts/analyze_results.py --metric Test_AUC
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.statistics import (
    compute_all_pairwise_tests,
    generate_significance_table,
    format_p_value,
    bootstrap_confidence_interval,
)
from src.utils.interpretability import visualize_gate_distribution


def load_comparison_results(output_dir: str) -> pd.DataFrame:
    """Load comparison CSV from outputs directory."""
    comparison_path = os.path.join(output_dir, 'comparison_binary.csv')
    
    if not os.path.exists(comparison_path):
        raise FileNotFoundError(f"Comparison file not found: {comparison_path}")
    
    df = pd.read_csv(comparison_path)
    print(f"✓ Loaded results for {len(df)} models")
    
    return df


def generate_results_summary(
    results_df: pd.DataFrame,
    metric: str = 'Mean_Acc',
    output_dir: str = 'outputs',
) -> str:
    """
    Generate comprehensive results summary.
    
    Args:
        results_df: Results DataFrame.
        metric: Primary metric for ranking.
        output_dir: Output directory.
    
    Returns:
        Summary text.
    """
    output = []
    
    output.append("=" * 100)
    output.append("BREAST CANCER CLASSIFICATION - RESULTS SUMMARY")
    output.append("=" * 100)
    output.append("")
    
    # Rank models by primary metric
    if metric in results_df.columns:
        ranked = results_df.sort_values(metric, ascending=False).reset_index(drop=True)
        ranked['Rank'] = range(1, len(ranked) + 1)
        
        output.append("MODEL RANKINGS")
        output.append("-" * 100)
        output.append(f"{'Rank':<6} {'Model':<30} {metric:<12} {'Std':<10} {'Params':<12} {'Time (s)':<10}")
        output.append("-" * 100)
        
        for _, row in ranked.iterrows():
            std_col = metric.replace('Mean_', 'Std_')
            std_val = row.get(std_col, 'N/A')
            output.append(
                f"{row['Rank']:<6} {row['Model']:<30} {row[metric]:<12.4f} {std_val:<10.4f} "
                f"{row.get('Total_Params', 'N/A'):<12} {row.get('Total_Time_s', 'N/A'):<10.1f}"
            )
        
        output.append("")
        
        # Save rankings
        rankings_path = os.path.join(output_dir, 'model_rankings.csv')
        ranked.to_csv(rankings_path, index=False)
        print(f"✓ Saved rankings to {rankings_path}")
    
    # Statistical significance
    output.append("")
    output.append(generate_significance_table(results_df, metric))
    
    # Bootstrap confidence intervals
    output.append("")
    output.append("=" * 100)
    output.append("BOOTSTRAP CONFIDENCE INTERVALS (95%%)")
    output.append("=" * 100)
    
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        
        # Get fold-level data if available
        if 'Test_Acc' in model_data.columns:
            scores = model_data['Test_Acc'].values
            mean, ci_lower, ci_upper = bootstrap_confidence_interval(scores, n_bootstrap=1000)
            output.append(f"{model:35s}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return "\n".join(output)


def generate_ablation_summary(
    results_df: pd.DataFrame,
    output_dir: str = 'outputs',
) -> str:
    """
    Generate ablation study summary.
    
    Groups models by architecture type and compares variants.
    """
    output = []
    
    output.append("")
    output.append("=" * 100)
    output.append("ABLATION STUDY SUMMARY")
    output.append("=" * 100)
    
    # Swin Transformer ablation
    swin_models = [c for c in results_df['Model'].unique() if 'Swin' in c]
    if swin_models:
        output.append("")
        output.append("Swin Transformer Variants:")
        output.append("-" * 60)
        swin_data = results_df[results_df['Model'].isin(swin_models)]
        output.append(swin_data[['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Total_Time_s']].to_string(index=False))
    
    # ConvNeXt ablation
    convnext_models = [c for c in results_df['Model'].unique() if 'ConvNeXt' in c]
    if convnext_models:
        output.append("")
        output.append("ConvNeXt Variants:")
        output.append("-" * 60)
        convnext_data = results_df[results_df['Model'].isin(convnext_models)]
        output.append(convnext_data[['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Total_Time_s']].to_string(index=False))
    
    # DeiT ablation
    deit_models = [c for c in results_df['Model'].unique() if 'DeiT' in c]
    if deit_models:
        output.append("")
        output.append("DeiT Variants:")
        output.append("-" * 60)
        deit_data = results_df[results_df['Model'].isin(deit_models)]
        output.append(deit_data[['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Total_Time_s']].to_string(index=False))
    
    # Quantum rotation ablation
    qenn_models = [c for c in results_df['Model'].unique() if 'QENN' in c]
    if qenn_models:
        output.append("")
        output.append("QENN Rotation Gate Ablation:")
        output.append("-" * 60)
        qenn_data = results_df[results_df['Model'].isin(qenn_models)]
        output.append(qenn_data[['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Total_Time_s']].to_string(index=False))
    
    # Fusion models
    fusion_models = [c for c in results_df['Model'].unique() if 'Fusion' in c or 'DualBranch' in c]
    if fusion_models:
        output.append("")
        output.append("Fusion Models:")
        output.append("-" * 60)
        fusion_data = results_df[results_df['Model'].isin(fusion_models)]
        output.append(fusion_data[['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Total_Time_s']].to_string(index=False))
    
    return "\n".join(output)


def create_comparison_plots(
    results_df: pd.DataFrame,
    output_dir: str = 'outputs',
):
    """Create comparison bar charts and radar plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to top models for readability
    top_models = results_df.nlargest(10, 'Mean_Acc') if 'Mean_Acc' in results_df.columns else results_df
    
    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'Mean_Acc' in top_models.columns:
        models = top_models['Model'].tolist()
        accs = top_models['Mean_Acc'].tolist()
        stds = top_models.get('Std_Acc', [0] * len(models)).tolist()
        
        bars = ax.bar(range(len(models)), accs, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison (Top 10)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        acc_plot_path = os.path.join(output_dir, 'accuracy_comparison.png')
        plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved accuracy comparison plot to {acc_plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze breast cancer classification results')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory with results')
    parser.add_argument('--metric', type=str, default='Mean_Acc',
                        help='Primary metric for ranking')
    parser.add_argument('--save-summary', action='store_true',
                        help='Save summary to file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print("BREAST CANCER CLASSIFICATION - RESULTS ANALYSIS")
    print("=" * 100)
    print()
    
    # Load results
    try:
        results_df = load_comparison_results(args.output_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nMake sure to run the pipeline first: python run_pipeline.py")
        return
    
    # Generate summary
    summary = generate_results_summary(results_df, args.metric, args.output_dir)
    print(summary)
    
    # Generate ablation summary
    ablation_summary = generate_ablation_summary(results_df, args.output_dir)
    print(ablation_summary)
    
    # Create plots
    create_comparison_plots(results_df, args.output_dir)
    
    # Save full summary
    if args.save_summary:
        summary_path = os.path.join(args.output_dir, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
            f.write("\n")
            f.write(ablation_summary)
        print(f"\n✓ Saved full summary to {summary_path}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nKey outputs:")
    print(f"  - Model rankings: {args.output_dir}/model_rankings.csv")
    print(f"  - Comparison plot: {args.output_dir}/accuracy_comparison.png")
    if args.save_summary:
        print(f"  - Full summary: {args.output_dir}/results_summary.txt")


if __name__ == '__main__':
    main()
