import os
import logging
import traceback
import pandas as pd
import numpy as np
from functools import wraps
import time
import json

# Configure logging
def setup_logging(log_dir="./logs", level=logging.INFO):
    """
    Set up logging configuration for the application
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"supply_chain_analytics_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger
    logger = logging.getLogger('supply_chain_analytics')
    logger.info(f"Logging configured. Log file: {log_file}")
    
    return logger

# Create a default logger
logger = setup_logging()

def safe_operation(default_return=None, log_error=True):
    """
    Decorator for safely executing operations with error handling
    
    Args:
        default_return: Value to return if operation fails
        log_error: Whether to log the error (default: True)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                return default_return
        return wrapper
    return decorator

def validate_dataframe(df, required_columns=None, numeric_columns=None, date_columns=None):
    """
    Validate a DataFrame for required columns and data types
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of columns that must be present
        numeric_columns: List of columns that should be numeric
        date_columns: List of columns that should be dates
        
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid DataFrame object")
        return False
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col])
                except:
                    logger.warning(f"Column '{col}' contains non-numeric values")
                    return False
    
    # Check date columns
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col])
                except:
                    logger.warning(f"Column '{col}' contains invalid date values")
                    return False
    
    return True

def clean_dataframe(df, numeric_columns=None, date_columns=None, categorical_columns=None):
    """
    Clean a DataFrame by handling missing values and converting data types
    
    Args:
        df: Pandas DataFrame to clean
        numeric_columns: List of columns that should be numeric
        date_columns: List of columns that should be dates
        categorical_columns: List of columns that should be categorical
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid DataFrame object")
        return df
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                # Convert to numeric, coercing errors to NaN
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Replace NaN with column median
                median_value = cleaned_df[col].median()
                cleaned_df[col].fillna(median_value, inplace=True)
    
    # Handle date columns
    if date_columns:
        for col in date_columns:
            if col in cleaned_df.columns:
                # Convert to datetime, coercing errors to NaT
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    # Handle categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in cleaned_df.columns:
                # Convert to categorical
                cleaned_df[col] = cleaned_df[col].astype('category')
    
    return cleaned_df

def log_analysis_progress(step, status="start", details=None):
    """
    Log the progress of the analysis pipeline
    
    Args:
        step: Name of the current step
        status: 'start', 'complete', or 'error'
        details: Additional details about the step
    """
    message = f"Analysis step '{step}' - {status}"
    if details:
        message += f": {details}"
    
    if status == "start":
        logger.info(message)
    elif status == "complete":
        logger.info(message)
    elif status == "error":
        logger.error(message)
    else:
        logger.debug(message)

def log_performance_metrics(metrics):
    """
    Log performance metrics for tracking
    
    Args:
        metrics: Dictionary of performance metrics
    """
    try:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        
        # Ensure the logs directory exists
        os.makedirs("./logs", exist_ok=True)
        
        # Append to the metrics log file
        with open("./logs/performance_metrics.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Error logging performance metrics: {str(e)}")