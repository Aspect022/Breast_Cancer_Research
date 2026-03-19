"""
Utility modules for breast cancer classification project.
"""

from .metrics import (
    compute_metrics,
    print_medical_metrics,
    get_convergence_epoch,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    count_parameters,
    compute_flops,
    measure_inference_time,
    get_gpu_memory_peak,
    count_attention_params,
    count_quantum_params,
    save_results_csv,
    save_epoch_log,
    build_comparison_row,
)

from .statistics import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    mcnemar_test,
    bootstrap_confidence_interval,
    compute_all_pairwise_tests,
    generate_significance_table,
    format_p_value,
    interpret_effect_size,
    StatisticalTestResult,
)

from .interpretability import (
    GradCAM,
    AttentionVisualizer,
    compute_saliency_map,
    visualize_gate_distribution,
    tensor_to_image,
    save_interpretability_results,
)

__all__ = [
    # Metrics
    'compute_metrics',
    'print_medical_metrics',
    'get_convergence_epoch',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'count_parameters',
    'compute_flops',
    'measure_inference_time',
    'get_gpu_memory_peak',
    'count_attention_params',
    'count_quantum_params',
    'save_results_csv',
    'save_epoch_log',
    'build_comparison_row',
    
    # Statistics
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'mcnemar_test',
    'bootstrap_confidence_interval',
    'compute_all_pairwise_tests',
    'generate_significance_table',
    'format_p_value',
    'interpret_effect_size',
    'StatisticalTestResult',
    
    # Interpretability
    'GradCAM',
    'AttentionVisualizer',
    'compute_saliency_map',
    'visualize_gate_distribution',
    'tensor_to_image',
    'save_interpretability_results',
]
