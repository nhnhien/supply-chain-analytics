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

def validate_time_series(ts_data, date_column, value_column, min_periods=3):
    """
    Validate a time series DataFrame for forecasting
    
    Args:
        ts_data: DataFrame containing time series data
        date_column: Name of column containing dates
        value_column: Name of column containing values
        min_periods: Minimum number of periods required
        
    Returns:
        tuple: (is_valid, message)
    """
    if ts_data is None or not isinstance(ts_data, pd.DataFrame):
        return False, "Invalid DataFrame object"
    
    # Check for required columns
    if date_column not in ts_data.columns:
        return False, f"Missing date column: {date_column}"
    
    if value_column not in ts_data.columns:
        return False, f"Missing value column: {value_column}"
    
    # Check if there are enough data points
    if len(ts_data) < min_periods:
        return False, f"Insufficient data: {len(ts_data)} rows available, {min_periods} required"
    
    # Check if dates are unique
    if ts_data[date_column].duplicated().any():
        return False, f"Duplicate dates found in {date_column}"
    
    # Check for missing values
    if ts_data[value_column].isna().any():
        return False, f"Missing values found in {value_column}"
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(ts_data[date_column]):
        try:
            ts_data[date_column] = pd.to_datetime(ts_data[date_column])
        except:
            return False, f"Invalid date format in {date_column}"
    
    # Check if values are numeric
    if not pd.api.types.is_numeric_dtype(ts_data[value_column]):
        return False, f"Non-numeric values found in {value_column}"
    
    # Check if time series is regular (same interval between points)
    try:
        ts_data_sorted = ts_data.sort_values(date_column)
        date_diffs = ts_data_sorted[date_column].diff().dropna().dt.days
        
        # Allow for small variations in interval (e.g., different month lengths)
        interval_range = date_diffs.max() - date_diffs.min()
        if interval_range > 5:  # More than 5 days variation
            return False, "Irregular time intervals detected"
    except:
        return False, "Error checking time intervals"
    
    return True, "Time series is valid for forecasting"

class DataValidator:
    """
    Class for validating supply chain data
    """
    
    @staticmethod
    def validate_orders(orders_df):
        """
        Validate orders DataFrame
        
        Args:
            orders_df: Orders DataFrame
            
        Returns:
            bool: True if valid
        """
        required_cols = ['order_id', 'customer_id', 'order_purchase_timestamp']
        numeric_cols = []
        date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                   'order_delivered_timestamp', 'order_estimated_delivery_date']
        
        return validate_dataframe(orders_df, required_cols, numeric_cols, date_cols)
    
    @staticmethod
    def validate_order_items(order_items_df):
        """
        Validate order items DataFrame
        
        Args:
            order_items_df: Order items DataFrame
            
        Returns:
            bool: True if valid
        """
        required_cols = ['order_id', 'product_id']
        numeric_cols = ['price', 'shipping_charges']
        
        return validate_dataframe(order_items_df, required_cols, numeric_cols)
    
    @staticmethod
    def validate_products(products_df):
        """
        Validate products DataFrame
        
        Args:
            products_df: Products DataFrame
            
        Returns:
            bool: True if valid
        """
        required_cols = ['product_id']
        numeric_cols = ['product_weight_g', 'product_length_cm', 
                       'product_height_cm', 'product_width_cm']
        
        return validate_dataframe(products_df, required_cols, numeric_cols)
    
    @staticmethod
    def validate_customers(customers_df):
        """
        Validate customers DataFrame
        
        Args:
            customers_df: Customers DataFrame
            
        Returns:
            bool: True if valid
        """
        required_cols = ['customer_id']
        
        return validate_dataframe(customers_df, required_cols)
    
    @staticmethod
    def check_referential_integrity(orders_df, order_items_df, products_df, customers_df):
        """
        Check referential integrity between tables
        
        Args:
            orders_df: Orders DataFrame
            order_items_df: Order items DataFrame
            products_df: Products DataFrame
            customers_df: Customers DataFrame
            
        Returns:
            dict: Integrity check results
        """
        results = {
            'valid': True,
            'issues': []
        }
        
        # Check if all order_id in order_items exist in orders
        if 'order_id' in order_items_df.columns and 'order_id' in orders_df.columns:
            order_items_order_ids = set(order_items_df['order_id'].unique())
            orders_order_ids = set(orders_df['order_id'].unique())
            
            missing_orders = order_items_order_ids - orders_order_ids
            if missing_orders:
                results['valid'] = False
                results['issues'].append(
                    f"Found {len(missing_orders)} order_ids in order_items that don't exist in orders"
                )
        
        # Check if all product_id in order_items exist in products
        if 'product_id' in order_items_df.columns and 'product_id' in products_df.columns:
            order_items_product_ids = set(order_items_df['product_id'].unique())
            products_product_ids = set(products_df['product_id'].unique())
            
            missing_products = order_items_product_ids - products_product_ids
            if missing_products:
                results['valid'] = False
                results['issues'].append(
                    f"Found {len(missing_products)} product_ids in order_items that don't exist in products"
                )
        
        # Check if all customer_id in orders exist in customers
        if 'customer_id' in orders_df.columns and 'customer_id' in customers_df.columns:
            orders_customer_ids = set(orders_df['customer_id'].unique())
            customers_customer_ids = set(customers_df['customer_id'].unique())
            
            missing_customers = orders_customer_ids - customers_customer_ids
            if missing_customers:
                results['valid'] = False
                results['issues'].append(
                    f"Found {len(missing_customers)} customer_ids in orders that don't exist in customers"
                )
        
        return results


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