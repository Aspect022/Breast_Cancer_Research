"""
Statistical Significance Testing for Model Comparison.

Provides statistical tests for comparing model performance:
- Paired t-tests
- Wilcoxon signed-rank tests
- Confidence intervals
- Effect size calculations (Cohen's d)
- McNemar's test for paired proportions
- Bootstrap resampling

Reference: implementation_plan.md §4 - Statistical Significance Testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    model_a: str
    model_b: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    notes: Optional[str] = None


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Paired t-test for comparing two models.
    
    Args:
        scores_a: Performance scores for model A (n_folds,).
        scores_b: Performance scores for model B (n_folds,).
        alpha: Significance level.
    
    Returns:
        StatisticalTestResult with test statistics.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Calculate effect size (Cohen's d)
    diff = scores_a - scores_b
    cohen_d = np.mean(diff) / (np.std(diff) + 1e-8)
    
    # Confidence interval for mean difference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = mean_diff - t_crit * std_diff / np.sqrt(n)
    ci_upper = mean_diff + t_crit * std_diff / np.sqrt(n)
    
    return StatisticalTestResult(
        test_name="Paired t-test",
        model_a="Model A",
        model_b="Model B",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=cohen_d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        notes=f"Cohen's d: {cohen_d:.3f}",
    )


def wilcoxon_signed_rank_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        scores_a: Performance scores for model A.
        scores_b: Performance scores for model B.
        alpha: Significance level.
    
    Returns:
        StatisticalTestResult with test statistics.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    
    # Effect size (rank-biserial correlation)
    n = len(scores_a)
    r = stat / (n * (n + 1) / 2)
    r = 2 * r - 1  # Convert to [-1, 1] range
    
    return StatisticalTestResult(
        test_name="Wilcoxon signed-rank",
        model_a="Model A",
        model_b="Model B",
        statistic=stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=r,
        notes=f"Rank-biserial r: {r:.3f}",
    )


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    correction: bool = True,
) -> StatisticalTestResult:
    """
    McNemar's test for paired nominal data (classification predictions).
    
    Tests whether two classifiers have different error rates.
    
    Args:
        predictions_a: Predictions from model A (n_samples,).
        predictions_b: Predictions from model B (n_samples,).
        labels: True labels (n_samples,).
        alpha: Significance level.
        correction: Use Edwards' continuity correction.
    
    Returns:
        StatisticalTestResult with test statistics.
    """
    # Build contingency table
    # a: both correct, b: A correct B wrong
    # c: A wrong B correct, d: both wrong
    
    both_correct = np.sum((predictions_a == labels) & (predictions_b == labels))
    a_correct = np.sum((predictions_a == labels) & (predictions_b != labels))
    b_correct = np.sum((predictions_a != labels) & (predictions_b == labels))
    both_wrong = np.sum((predictions_a != labels) & (predictions_b != labels))
    
    # McNemar's test statistic
    b, c = a_correct, b_correct
    
    if correction:
        # Edwards' continuity correction
        stat = (np.abs(b - c) - 1) ** 2 / (b + c + 1e-8)
    else:
        stat = (b - c) ** 2 / (b + c + 1e-8)
    
    p_value = stats.chi2.sf(stat, 1)
    
    # Effect size (odds ratio)
    odds_ratio = (b + 0.5) / (c + 0.5)
    
    return StatisticalTestResult(
        test_name="McNemar's test",
        model_a="Model A",
        model_b="Model B",
        statistic=stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=odds_ratio,
        notes=f"Odds ratio: {odds_ratio:.3f}, Contingency: [[{both_correct}, {a_correct}], [{b_correct}, {both_wrong}]]",
    )


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    statistic: str = 'mean',
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.
    
    Args:
        scores: Performance scores.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level.
        statistic: Statistic to compute ('mean', 'median', 'std').
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n = len(scores)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample = scores[indices]
        
        if statistic == 'mean':
            bootstrap_stats.append(np.mean(sample))
        elif statistic == 'median':
            bootstrap_stats.append(np.median(sample))
        elif statistic == 'std':
            bootstrap_stats.append(np.std(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Point estimate
    if statistic == 'mean':
        point_est = np.mean(scores)
    elif statistic == 'median':
        point_est = np.median(scores)
    else:
        point_est = np.std(scores)
    
    # Confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_est, ci_lower, ci_upper


def compute_all_pairwise_tests(
    results_df: pd.DataFrame,
    metric: str = 'Test_Acc',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute all pairwise statistical tests between models.
    
    Args:
        results_df: DataFrame with columns ['Model', 'Fold', metric].
        metric: Metric column to test.
        alpha: Significance level.
    
    Returns:
        DataFrame with all pairwise test results.
    """
    models = results_df['Model'].unique()
    test_results = []
    
    for i, model_a in enumerate(models):
        for model_b in models[i+1:]:
            # Get scores for each model across folds
            scores_a = results_df[results_df['Model'] == model_a][metric].values
            scores_b = results_df[results_df['Model'] == model_b][metric].values
            
            if len(scores_a) != len(scores_b):
                continue
            
            # Paired t-test
            t_result = paired_t_test(scores_a, scores_b, alpha)
            t_result.model_a = model_a
            t_result.model_b = model_b
            test_results.append({
                'Model_A': model_a,
                'Model_B': model_b,
                'Test': 'Paired t-test',
                'Statistic': t_result.statistic,
                'P-value': t_result.p_value,
                'Significant': t_result.significant,
                'Effect_Size': t_result.effect_size,
                'CI_Lower': t_result.ci_lower,
                'CI_Upper': t_result.ci_upper,
            })
            
            # Wilcoxon test
            w_result = wilcoxon_signed_rank_test(scores_a, scores_b, alpha)
            w_result.model_a = model_a
            w_result.model_b = model_b
            test_results.append({
                'Model_A': model_a,
                'Model_B': model_b,
                'Test': 'Wilcoxon',
                'Statistic': w_result.statistic,
                'P-value': w_result.p_value,
                'Significant': w_result.significant,
                'Effect_Size': w_result.effect_size,
            })
    
    return pd.DataFrame(test_results)


def format_p_value(p_value: float) -> str:
    """Format p-value for reporting."""
    if p_value < 0.001:
        return "p < 0.001***"
    elif p_value < 0.01:
        return f"p = {p_value:.3f}**"
    elif p_value < 0.05:
        return f"p = {p_value:.3f}*"
    elif p_value < 0.1:
        return f"p = {p_value:.3f}†"
    else:
        return f"p = {p_value:.3f}"


def interpret_effect_size(effect_size: float, test_type: str = 'cohens_d') -> str:
    """
    Interpret effect size magnitude.
    
    Args:
        effect_size: Effect size value.
        test_type: Type of effect size ('cohens_d', 'r', 'odds_ratio').
    
    Returns:
        Interpretation string.
    """
    if test_type == 'cohens_d':
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    elif test_type == 'r':
        abs_r = abs(effect_size)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    elif test_type == 'odds_ratio':
        if effect_size < 1:
            effect_size = 1 / effect_size
        if effect_size < 1.5:
            return "negligible"
        elif effect_size < 3:
            return "small"
        elif effect_size < 5:
            return "medium"
        else:
            return "large"
    
    return "unknown"


def generate_significance_table(
    results_df: pd.DataFrame,
    metric: str = 'Test_Acc',
    alpha: float = 0.05,
) -> str:
    """
    Generate a formatted significance table for paper/report.
    
    Args:
        results_df: Results DataFrame.
        metric: Metric to test.
        alpha: Significance level.
    
    Returns:
        Formatted table string.
    """
    tests = compute_all_pairwise_tests(results_df, metric, alpha)
    
    # Format output
    output = []
    output.append("=" * 100)
    output.append("STATISTICAL SIGNIFICANCE TESTING")
    output.append(f"Metric: {metric} | Alpha: {alpha}")
    output.append("=" * 100)
    output.append("")
    
    for _, row in tests.iterrows():
        p_str = format_p_value(row['P-value'])
        effect_str = ""
        if row['Effect_Size'] is not None:
            if 'Wilcoxon' in row['Test']:
                effect_str = f" (r = {row['Effect_Size']:.3f}, {interpret_effect_size(row['Effect_Size'], 'r')})"
            else:
                effect_str = f" (d = {row['Effect_Size']:.3f}, {interpret_effect_size(row['Effect_Size'], 'cohens_d')})"
        
        sig_marker = "✓" if row['Significant'] else "✗"
        output.append(f"{row['Model_A']:25s} vs {row['Model_B']:25s} | {row['Test']:20s} | {p_str:15s} | {sig_marker} {effect_str}")
    
    output.append("")
    output.append("Legend: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1")
    output.append("=" * 100)
    
    return "\n".join(output)
