"""工具模块。"""
import logging

logger = logging.getLogger(__name__)
from .logging_utils import (
    setup_logging,
    setup_exception_logging,
    setup_json_logging,
    get_logger,
    add_file_handler,
    LoggingContext,
    log_execution_time,
    log_function_call,
    setup_logging_from_config,
    JsonFormatter
)

from .config_utils import (
    load_config,
    save_config,
    merge_configs,
    validate_config,
    create_default_config,
    update_config_with_args,
    get_config_value,
    set_config_value,
    print_config,
    ConfigError
)

from .file_utils import (
    ensure_directory,
    find_files,
    copy_files,
    remove_files,
    get_file_size,
    get_file_hash,
    read_text_file,
    write_text_file,
    read_json_file,
    write_json_file,
    read_yaml_file,
    write_yaml_file,
    read_pickle_file,
    write_pickle_file,
    backup_file,
    get_file_info,
    FileUtils
)

from .time_utils import (
    parse_timestamp,
    format_timedelta,
    Timer,
    timeit,
    timeit_decorator,
    get_current_time,
    get_timestamp,
    calculate_time_window,
    time_since,
    TimeSeriesGenerator,
    estimate_remaining_time
)

from .active_region_utils import (
    parse_active_region_info,
    build_region_key,
    attach_active_region_columns,
    find_unknown_active_region_events,
    format_unknown_active_region_summary,
    is_unknown_active_region,
    raise_if_unknown_active_regions,
)

from .metrics_calculation import (
    calculate_metrics,
    ConfusionMatrix,
    ROC_AUC_Calculator,
    PrecisionRecallCalculator,
    RegressionMetrics,
    BoundingBoxMetrics,
    MetricsTracker
)

try:
    from .visualization import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_precision_recall_curves,
        plot_training_history,
        plot_feature_importance,
        visualize_attention,
        plot_predictions_distribution,
        plot_calibration_curve,
        plot_uncertainty_distribution,
        plot_multimodal_features,
        create_dashboard,
        plot_solar_flare_prediction,
        save_figure,
    )
except Exception as exc:
    logger.warning("导入 utils.visualization 失败: %s", exc)
    plot_confusion_matrix = None
    plot_roc_curves = None
    plot_precision_recall_curves = None
    plot_training_history = None
    plot_feature_importance = None
    visualize_attention = None
    plot_predictions_distribution = None
    plot_calibration_curve = None
    plot_uncertainty_distribution = None
    plot_multimodal_features = None
    create_dashboard = None
    plot_solar_flare_prediction = None
    save_figure = None

__all__ = [
    # logging_utils
    'setup_logging',
    'setup_exception_logging',
    'setup_json_logging',
    'get_logger',
    'add_file_handler',
    'LoggingContext',
    'log_execution_time',
    'log_function_call',
    'setup_logging_from_config',
    'JsonFormatter',

    # config_utils
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config',
    'create_default_config',
    'update_config_with_args',
    'get_config_value',
    'set_config_value',
    'print_config',
    'ConfigError',

    # file_utils
    'ensure_directory',
    'find_files',
    'copy_files',
    'remove_files',
    'get_file_size',
    'get_file_hash',
    'read_text_file',
    'write_text_file',
    'read_json_file',
    'write_json_file',
    'read_yaml_file',
    'write_yaml_file',
    'read_pickle_file',
    'write_pickle_file',
    'backup_file',
    'get_file_info',
    'FileUtils',

    # time_utils
    'parse_timestamp',
    'format_timedelta',
    'Timer',
    'timeit',
    'timeit_decorator',
    'get_current_time',
    'get_timestamp',
    'calculate_time_window',
    'time_since',
    'TimeSeriesGenerator',
    'estimate_remaining_time',

    # metrics_calculation
    'calculate_metrics',
    'ConfusionMatrix',
    'ROC_AUC_Calculator',
    'PrecisionRecallCalculator',
    'RegressionMetrics',
    'BoundingBoxMetrics',
    'MetricsTracker',

    # visualization
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_training_history',
    'plot_feature_importance',
    'visualize_attention',
    'plot_predictions_distribution',
    'plot_calibration_curve',
    'plot_uncertainty_distribution',
    'plot_multimodal_features',
    'create_dashboard',
    'plot_solar_flare_prediction',
    'save_figure',

    # active_region_utils
    'parse_active_region_info',
    'build_region_key',
    'attach_active_region_columns',
    'find_unknown_active_region_events',
    'format_unknown_active_region_summary',
    'is_unknown_active_region',
    'raise_if_unknown_active_regions',
]
