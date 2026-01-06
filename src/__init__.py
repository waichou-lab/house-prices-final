"""
House Prices EDA Project - Source Modules
"""

from .data_loader import load_dataset, get_data_info
from .data_cleaner import check_missing_data, remove_high_missing_columns
from .visualization import plot_price_distribution, plot_correlation_analysis
from .feature_engineering import create_new_features, apply_log_transform
from .utils import setup_environment, detect_outliers_iqr
from .pdf_report import create_pdf_report

__version__ = "1.0.0"
__author__ = "Your Name"

print("Source modules loaded successfully")