import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

def detect_and_handle_anomalies(df, column, window_size=5, z_threshold=3.0):
    """
    Detect and handle anomalies in time series data using rolling Z-score method
    
    Args:
        df: DataFrame containing time series data
        column: Column name to check for anomalies
        window_size: Window size for rolling statistics
        z_threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomalies handled
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Calculate rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window_size, center=True).mean()
    rolling_std = df[column].rolling(window=window_size, center=True).std()
    
    # Replace NaN values in rolling stats
    rolling_mean = rolling_mean.fillna(df[column].mean())
    rolling_std = rolling_std.fillna(df[column].std())
    
    # Make sure std is not zero to avoid division by zero
    rolling_std = rolling_std.replace(0, df[column].std())
    if rolling_std.iloc[0] == 0:
        rolling_std = rolling_std.replace(0, 1)
    
    # Calculate z-scores
    z_scores = (df[column] - rolling_mean) / rolling_std
    
    # Identify anomalies
    anomalies = (z_scores.abs() > z_threshold)
    anomaly_indices = df.index[anomalies]
    
    if len(anomaly_indices) > 0:
        print(f"Detected {len(anomaly_indices)} anomalies in {column}")
        
        # Replace anomalies with the median of neighboring non-anomalous values
        for idx in anomaly_indices:
            # Define window around the anomaly
            left_bound = max(0, df.index.get_loc(idx) - window_size)
            right_bound = min(len(df), df.index.get_loc(idx) + window_size + 1)
            window = df.iloc[left_bound:right_bound]
            
            # Get non-anomalous values from the window
            non_anomalous = window[~anomalies.iloc[left_bound:right_bound]]
            
            if len(non_anomalous) > 0:
                # Replace with median of non-anomalous values
                result.loc[idx, column] = non_anomalous[column].median()
            else:
                # If all window values are anomalous, use global median
                result.loc[idx, column] = df[column].median()
    
    return result

def infer_date_format(date_str):
    """
    Infer date format from a string by trying common formats
    
    Args:
        date_str: Date string to infer format from
        
    Returns:
        Inferred format string or None if no format can be inferred
    """
    formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
        '%m-%d-%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%m-%d-%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f'
    ]
    
    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue
    
    return None

def smart_date_conversion(date_column):
    """
    Convert a date column to datetime with automatic format detection
    
    Args:
        date_column: Series containing date strings
        
    Returns:
        Series with converted datetime values
    """
    # Filter out NaN values for format detection
    valid_dates = date_column.dropna()
    
    if len(valid_dates) == 0:
        return pd.Series([None] * len(date_column))
    
    # Try to infer format from the first valid date
    sample_date = valid_dates.iloc[0]
    inferred_format = infer_date_format(str(sample_date))
    
    if inferred_format:
        try:
            return pd.to_datetime(date_column, format=inferred_format, errors='coerce')
        except:
            # Fall back to pandas default parsing
            return pd.to_datetime(date_column, errors='coerce')
    else:
        # Let pandas try to infer the format
        return pd.to_datetime(date_column, errors='coerce')

def fill_missing_dates(df, date_column='date', freq='MS', fill_method='linear'):
    """
    Fill in missing dates in time series data
    
    Args:
        df: DataFrame containing time series data
        date_column: Name of date column
        freq: Frequency of time series ('MS' for month start, 'D' for daily, etc.)
        fill_method: Method for filling in missing values ('linear', 'ffill', 'bfill', 'mean')
        
    Returns:
        DataFrame with missing dates filled in
    """
    # Make sure the date column is a datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Set date as index
    df_indexed = df.set_index(date_column)
    
    # Create a complete date range
    date_range = pd.date_range(start=df_indexed.index.min(), end=df_indexed.index.max(), freq=freq)
    
    # Reindex the DataFrame to include all dates
    df_filled = df_indexed.reindex(date_range)
    
    # Fill in missing values
    if fill_method == 'linear':
        df_filled = df_filled.interpolate(method='linear')
    elif fill_method == 'ffill':
        df_filled = df_filled.ffill()
    elif fill_method == 'bfill':
        df_filled = df_filled.bfill()
    elif fill_method == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    
    # Reset index to convert date back to a column
    df_filled = df_filled.reset_index()
    df_filled = df_filled.rename(columns={'index': date_column})
    
    return df_filled

def apply_seasonal_adjustment(df, column, period=12, multiplicative=False):
    """
    Apply seasonal adjustment to time series data
    
    Args:
        df: DataFrame containing time series data
        column: Column to apply seasonal adjustment to
        period: Seasonality period (12 for monthly, 4 for quarterly, etc.)
        multiplicative: Whether to use multiplicative model (otherwise additive)
        
    Returns:
        DataFrame with seasonal adjustment factors and adjusted values
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Create a copy of the DataFrame
    result = df.copy()
    
    # Ensure we have enough data for decomposition
    if len(df) < period * 2:
        print(f"Not enough data for seasonal decomposition (need at least {period * 2} periods)")
        return result
    
    # Set up the model
    model = 'multiplicative' if multiplicative else 'additive'
    
    try:
        # Apply seasonal decomposition
        decomposition = seasonal_decompose(
            df[column], model=model, period=period, extrapolate_trend='freq'
        )
        
        # Extract components
        result['trend'] = decomposition.trend
        result['seasonal'] = decomposition.seasonal
        result['residual'] = decomposition.resid
        
        # Create seasonally adjusted column
        if multiplicative:
            result[f'{column}_adjusted'] = df[column] / decomposition.seasonal
        else:
            result[f'{column}_adjusted'] = df[column] - decomposition.seasonal
        
        print(f"Applied {model} seasonal adjustment with period {period}")
        
        # Create a seasonal factors DataFrame for future adjustments
        seasonal_factors = pd.DataFrame({'seasonal_factor': decomposition.seasonal})
        seasonal_patterns = seasonal_factors.groupby(seasonal_factors.index % period).mean()
        
        return result, seasonal_patterns
        
    except Exception as e:
        print(f"Error during seasonal decomposition: {e}")
        return result, None
def calculate_delivery_days(orders, supply_chain=None):
    if 'delivery_days' not in orders.columns:
        orders['delivery_days'] = np.nan

    # Convert timestamps
    for c in ['order_purchase_timestamp','order_approved_at','order_delivered_timestamp','order_estimated_delivery_date']:
        if c in orders: orders[c] = pd.to_datetime(orders[c], errors='coerce')

    # Compute delivery_days from actual or estimated dates
    orders['delivery_days'] = (
        (orders['order_delivered_timestamp'].fillna(orders['order_estimated_delivery_date']) -
         orders['order_purchase_timestamp'])
        .dt.days.clip(lower=1)
    )

    # Impute missing using category median if supply_chain provided
    if supply_chain is not None and orders['delivery_days'].isna().any():
        medians = (supply_chain.dropna(subset=['delivery_days'])
                   .groupby('product_category_name')['delivery_days']
                   .median())
        orders['delivery_days'] = orders.apply(
            lambda r: medians.get(r.product_category_name, np.nan) 
                      if pd.isna(r.delivery_days) else r.delivery_days, axis=1
        )

    # Final fallback to global median
    global_median = orders['delivery_days'].median()
    orders['delivery_days'] = orders['delivery_days'].fillna(global_median).clip(1,30)

    # Impute missing delivered timestamp
    missing = orders['order_delivered_timestamp'].isna() & orders['order_purchase_timestamp'].notna()
    orders.loc[missing, 'order_delivered_timestamp'] = (
        orders.loc[missing, 'order_purchase_timestamp'] + 
        pd.to_timedelta(orders.loc[missing, 'delivery_days'], unit='D')
    )

    return orders

def preprocess_order_data(orders_df, output_dir=None, apply_seasonal=True):
    """
    Preprocess order data to handle missing values and calculate delivery metrics
    with improved handling of missing timestamp columns and data quality.
    
    Args:
        orders_df: DataFrame containing order data
        output_dir: Directory to save processed data (optional)
        apply_seasonal: Whether to apply seasonal adjustment to orders
        
    Returns:
        Processed DataFrame with enhanced data quality
    """
    print("Preprocessing order data...")
    
    # Define required timestamp columns
    timestamp_columns = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_timestamp', 
        'order_estimated_delivery_date'
    ]
    
    # Identify missing timestamp columns
    missing_columns = [col for col in timestamp_columns if col not in orders_df.columns]
    
    if missing_columns:
        print(f"Note: Missing timestamp columns: {', '.join(missing_columns)}")
        print("Creating synthetic timestamps based on available data...")
        
        # Create synthetic order_purchase_timestamp if missing
        if 'order_purchase_timestamp' not in orders_df.columns:
            if 'order_year' in orders_df.columns and 'order_month' in orders_df.columns:
                orders_df['order_purchase_timestamp'] = pd.to_datetime(
                    orders_df['order_year'].astype(str) + '-' + 
                    orders_df['order_month'].astype(str).str.zfill(2) + '-15'  # Middle of month
                )
                print("Created synthetic order_purchase_timestamp from order_year and order_month")
            else:
                # Use the current date as fallback minus 1 year
                default_start = (datetime.now().replace(year=datetime.now().year-1)).strftime("%Y-%m-%d")
                orders_df['order_purchase_timestamp'] = pd.date_range(
                    start=default_start, periods=len(orders_df), freq='D'
                )
                print(f"Warning: Created default order_purchase_timestamp starting from {default_start}")
    
    # Convert existing timestamps to datetime with smart format detection
    for col in timestamp_columns:
        if col in orders_df.columns:
            if orders_df[col].dtype == 'object':  # String dates
                orders_df[col] = smart_date_conversion(orders_df[col])
            else:
                orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
    
    # Create synthetic timestamps for missing columns using business logic
    if 'order_approved_at' not in orders_df.columns and 'order_purchase_timestamp' in orders_df.columns:
        # Approval is ~1 day after purchase plus random hours
        orders_df['order_approved_at'] = orders_df['order_purchase_timestamp'] + pd.Timedelta(days=1)
        # Add some randomness to approval times
        random_hours = np.random.randint(0, 24, size=len(orders_df))
        orders_df['order_approved_at'] += pd.Series([pd.Timedelta(hours=h) for h in random_hours])
        print("Created synthetic order_approved_at based on purchase timestamp")
    
    # For missing order_delivered_timestamp: delivery takes 3-7 days after approval or 4-8 days after purchase
    if 'order_delivered_timestamp' not in orders_df.columns:
        if 'order_approved_at' in orders_df.columns:
            delivery_days = np.random.randint(3, 8, size=len(orders_df))
            orders_df['order_delivered_timestamp'] = orders_df['order_approved_at'] + pd.Series(
                [pd.Timedelta(days=d) for d in delivery_days]
            )
            print("Created synthetic order_delivered_timestamp based on approval timestamp")
        elif 'order_purchase_timestamp' in orders_df.columns:
            delivery_days = np.random.randint(4, 9, size=len(orders_df))
            orders_df['order_delivered_timestamp'] = orders_df['order_purchase_timestamp'] + pd.Series(
                [pd.Timedelta(days=d) for d in delivery_days]
            )
            print("Created synthetic order_delivered_timestamp based on purchase timestamp")
    
    # For missing order_estimated_delivery_date: estimated delivery is 5-10 days after purchase
    if 'order_estimated_delivery_date' not in orders_df.columns and 'order_purchase_timestamp' in orders_df.columns:
        est_days = np.random.randint(5, 11, size=len(orders_df))
        orders_df['order_estimated_delivery_date'] = orders_df['order_purchase_timestamp'] + pd.Series(
            [pd.Timedelta(days=d) for d in est_days]
        )
        print("Created synthetic order_estimated_delivery_date based on purchase timestamp")
    
    # Calculate processing time (in days) from purchase to approval
    if 'order_purchase_timestamp' in orders_df.columns and 'order_approved_at' in orders_df.columns:
        orders_df['processing_time'] = (
            orders_df['order_approved_at'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)
        orders_df.loc[orders_df['processing_time'] < 0, 'processing_time'] = np.nan
    else:
        orders_df['processing_time'] = np.nan
        print("Warning: Missing timestamp columns for processing time calculation")
    
    # Calculate actual delivery days
    if 'order_purchase_timestamp' in orders_df.columns and 'order_delivered_timestamp' in orders_df.columns:
        orders_df['actual_delivery_days'] = (
            orders_df['order_delivered_timestamp'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)
        orders_df.loc[orders_df['actual_delivery_days'] < 0, 'actual_delivery_days'] = np.nan
    else:
        orders_df['actual_delivery_days'] = np.nan
        print("Warning: Missing timestamp columns for actual delivery time calculation")
    
    # Calculate estimated delivery days
    if 'order_purchase_timestamp' in orders_df.columns and 'order_estimated_delivery_date' in orders_df.columns:
        orders_df['estimated_delivery_days'] = (
            orders_df['order_estimated_delivery_date'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)
        orders_df.loc[orders_df['estimated_delivery_days'] < 0, 'estimated_delivery_days'] = np.nan
    else:
        orders_df['estimated_delivery_days'] = np.nan
        print("Warning: Missing timestamp columns for estimated delivery time calculation")
    
    # Detect and handle anomalies in processing and delivery times
    if 'processing_time' in orders_df.columns:
        orders_df = detect_and_handle_anomalies(orders_df, 'processing_time')
    
    if 'actual_delivery_days' in orders_df.columns:
        orders_df = detect_and_handle_anomalies(orders_df, 'actual_delivery_days')
    
    # Determine on-time delivery: delivered on or before estimated date
    if 'actual_delivery_days' in orders_df.columns and 'estimated_delivery_days' in orders_df.columns:
        mask = (~orders_df['actual_delivery_days'].isna() & ~orders_df['estimated_delivery_days'].isna())
        orders_df.loc[mask, 'on_time_delivery'] = (
            orders_df.loc[mask, 'actual_delivery_days'] <= orders_df.loc[mask, 'estimated_delivery_days']
        ).astype(int)
    else:
        # Use a synthetic on-time delivery flag with a realistic distribution
        on_time_ratio = 0.85  # 85% on-time rate is typical for e-commerce
        random_vals = np.random.random(len(orders_df))
        orders_df['on_time_delivery'] = (random_vals <= on_time_ratio).astype(int)
        print("Created synthetic on_time_delivery with 85% on-time rate")
    
    # Fill missing processing times with median by product category if available
    if 'processing_time' in orders_df.columns and 'product_category_name' in orders_df.columns:
        category_medians = orders_df.groupby('product_category_name')['processing_time'].median()
        for category, median in category_medians.items():
            mask = (orders_df['product_category_name'] == category) & (orders_df['processing_time'].isna())
            orders_df.loc[mask, 'processing_time'] = median
    
    # Fill any remaining missing processing times with overall median
    if 'processing_time' in orders_df.columns:
        median_pt = orders_df['processing_time'].median()
        orders_df['processing_time'] = orders_df['processing_time'].fillna(median_pt if pd.notna(median_pt) else 1.0)
    
    # Use actual delivery days as primary delivery_days; fallback to estimated
    if 'actual_delivery_days' in orders_df.columns:
        orders_df['delivery_days'] = orders_df['actual_delivery_days']
    
    if 'estimated_delivery_days' in orders_df.columns:
        mask = orders_df['delivery_days'].isna()
        orders_df.loc[mask, 'delivery_days'] = orders_df.loc[mask, 'estimated_delivery_days']
    
    # Fill any remaining NaN values in delivery_days
    median_delivery = orders_df['delivery_days'].median() if 'delivery_days' in orders_df.columns else None
    if pd.isna(median_delivery):
        median_delivery = 7.0  # Reasonable default
    
    if 'delivery_days' not in orders_df.columns:
        orders_df['delivery_days'] = median_delivery
    else:
        orders_df['delivery_days'] = orders_df['delivery_days'].fillna(median_delivery)
    
    # Winsorize extreme delivery days (cap at reasonable values)
    q99 = orders_df['delivery_days'].quantile(0.99)
    orders_df['delivery_days'] = orders_df['delivery_days'].clip(upper=min(q99, 30))
    
    # Extract year and month from order_purchase_timestamp if available
    if 'order_purchase_timestamp' in orders_df.columns:
        orders_df['order_year'] = orders_df['order_purchase_timestamp'].dt.year
        orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.month
    elif 'order_year' not in orders_df.columns or 'order_month' not in orders_df.columns:
        current_year = datetime.now().year
        orders_df['order_year'] = current_year - 1  # Default to previous year
        orders_df['order_month'] = np.random.randint(1, 13, size=len(orders_df))
        print("Warning: Created default order_year and order_month values")
    
    # Calculate monthly order counts for seasonality analysis
    if 'order_year' in orders_df.columns and 'order_month' in orders_df.columns:
        monthly_counts = orders_df.groupby(['order_year', 'order_month']).size().reset_index(name='count')
        monthly_counts['date'] = pd.to_datetime(
            monthly_counts['order_year'].astype(str) + '-' + 
            monthly_counts['order_month'].astype(str).str.zfill(2) + '-01'
        )
        monthly_counts = monthly_counts.sort_values('date')
        
        # Apply seasonal adjustment if enough data is available and requested
        if len(monthly_counts) >= 24 and apply_seasonal:  # Need at least 2 years of data
            try:
                monthly_counts_adjusted, seasonal_factors = apply_seasonal_adjustment(
                    monthly_counts, 'count', period=12
                )
                if seasonal_factors is not None:
                    seasonal_factors_path = os.path.join(output_dir, 'seasonal_factors.csv') if output_dir else None
                    if seasonal_factors_path:
                        seasonal_factors.to_csv(seasonal_factors_path)
                        print(f"Saved seasonal factors to {seasonal_factors_path}")
            except Exception as e:
                print(f"Warning: Could not apply seasonal adjustment: {e}")
    
    # Calculate delivery performance metrics
    delivery_metrics = {
        'on_time_delivery_rate': orders_df['on_time_delivery'].mean() * 100 if 'on_time_delivery' in orders_df.columns else 85.0,
        'avg_delivery_days': orders_df['delivery_days'].mean(),
        'avg_processing_time': orders_df['processing_time'].mean() if 'processing_time' in orders_df.columns else 1.0,
        'delivery_days_std': orders_df['delivery_days'].std(),
        'processing_time_std': orders_df['processing_time'].std() if 'processing_time' in orders_df.columns else 0.3
    }
    
    # Add flags for data quality
    delivery_metrics['has_actual_delivery_data'] = 'order_delivered_timestamp' in orders_df.columns
    delivery_metrics['has_actual_processing_data'] = 'order_approved_at' in orders_df.columns
    
    # Save processed data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame([delivery_metrics]).to_csv(os.path.join(output_dir, 'delivery_performance.csv'), index=False)
        orders_df.to_csv(os.path.join(output_dir, 'processed_orders.csv'), index=False)
    
    print("Order data preprocessing complete")
    return orders_df, delivery_metrics

def preprocess_product_data(products_df, output_dir=None):
    """
    Enhanced product data preprocessing with advanced imputation techniques.
    
    Args:
        products_df: DataFrame containing product data
        output_dir: Directory to save processed data (optional)
        
    Returns:
        Processed DataFrame with improved data quality
    """
    print("Preprocessing product data with advanced imputation...")
    df = products_df.copy()
    
    # Identify numeric attributes
    numeric = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    
    # Check if there are missing values that need imputation
    total_missing = df[numeric].isna().sum().sum()
    if total_missing > 0:
        print(f"Found {total_missing} missing values in numeric columns")
        
        # KNN imputation by category approach
        if 'product_category_name' in df.columns and df['product_category_name'].isna().sum() < len(df) * 0.5:
            print("Using category-based KNN imputation")
            
            # First fill missing categories
            if df['product_category_name'].isna().sum() > 0:
                # Use KNN on numeric features to estimate missing categories
                knn_imputer = KNNImputer(n_neighbors=5)
                numeric_data = df[numeric].copy()
                
                # Standardize numeric data for better KNN performance
                numeric_means = numeric_data.mean()
                numeric_stds = numeric_data.std()
                for col in numeric:
                    numeric_data[col] = (numeric_data[col] - numeric_means[col]) / numeric_stds[col]
                
                # Fill missing numeric values with initial approximation
                numeric_data = numeric_data.fillna(0)
                
                # Get category dummies for non-null categories
                category_mask = ~df['product_category_name'].isna()
                categories = pd.get_dummies(df.loc[category_mask, 'product_category_name'])
                
                # Train a KNN imputer on categories
                X = numeric_data.loc[category_mask].values
                Y = categories.values
                
                # For each missing category, predict using KNN
                missing_category_mask = df['product_category_name'].isna()
                for idx in df[missing_category_mask].index:
                    # Get numeric features for this product
                    features = numeric_data.loc[idx].values.reshape(1, -1)
                    
                    # Find k nearest neighbors
                    distances = np.sqrt(((X - features) ** 2).sum(axis=1))
                    nearest_indices = np.argsort(distances)[:5]
                    
                    # Get categories of neighbors
                    neighbor_categories = df.loc[category_mask, 'product_category_name'].iloc[nearest_indices]
                    
                    # Assign most common category
                    df.loc[idx, 'product_category_name'] = neighbor_categories.mode()[0]
            
            # Now impute missing numeric values by category
            for category in df['product_category_name'].unique():
                category_mask = df['product_category_name'] == category
                category_products = df[category_mask]
                
                if len(category_products) >= 5:  # Need enough products for meaningful imputation
                    category_imputer = KNNImputer(n_neighbors=min(5, len(category_products)-1))
                    imputed_values = category_imputer.fit_transform(category_products[numeric])
                    df.loc[category_mask, numeric] = imputed_values
                else:
                    # For small categories, use simpler imputation
                    for col in numeric:
                        median = category_products[col].median()
                        if pd.notna(median):
                            df.loc[category_mask & df[col].isna(), col] = median
        
        # Global imputation for any remaining missing values
        global_imputer = KNNImputer(n_neighbors=10)
        numeric_data = df[numeric].copy()
        imputed_values = global_imputer.fit_transform(numeric_data)
        df[numeric] = imputed_values
    
    # Ensure product dimensions make sense
    # (e.g., height, length, width should be positive; weight should be reasonable)
    for col in numeric:
        # Cap extreme values at 99.5 percentile
        cap = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=0, upper=cap)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'processed_products.csv'), index=False)
        
        # Save a report on imputation and data quality
        imputation_report = {
            'original_missing': products_df[numeric].isna().sum().to_dict(),
            'final_missing': df[numeric].isna().sum().to_dict(),
            'capped_values_count': ((df[numeric] != products_df[numeric]) & ~products_df[numeric].isna()).sum().to_dict()
        }
        pd.DataFrame([imputation_report]).to_csv(os.path.join(output_dir, 'product_imputation_report.csv'), index=False)
    
    print("Product data preprocessing complete")
    return df
def preprocess_order_items(order_items_df, product_categories=None, output_dir=None):
    """
    Preprocess order items data with improved imputation and anomaly detection.
    
    Args:
        order_items_df: DataFrame containing order items data
        product_categories: Optional DataFrame or dict mapping product_id to category
        output_dir: Directory to save processed data (optional)
        
    Returns:
        Processed DataFrame with improved data quality
    """
    print("Preprocessing order items data...")
    df = order_items_df.copy()
    
    # Identify price-related columns
    price_columns = ['price', 'shipping_charges', 'freight_value']
    available_price_cols = [col for col in price_columns if col in df.columns]
    
    if not available_price_cols:
        print("Warning: No price-related columns found in order items data")
        return df
        
    # Check if product categories are available
    have_categories = product_categories is not None
    
    if have_categories:
        # Create a product_id to category mapping if not already a dict
        if isinstance(product_categories, pd.DataFrame):
            if 'product_id' in product_categories.columns and 'product_category_name' in product_categories.columns:
                category_map = dict(zip(product_categories['product_id'], product_categories['product_category_name']))
            else:
                print("Warning: product_categories DataFrame missing required columns")
                have_categories = False
        else:
            # Assume it's already a dict
            category_map = product_categories
    
    for col in available_price_cols:
        # Check for negative prices
        neg_mask = df[col] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            print(f"Found {neg_count} negative values in {col}, replacing with absolute values")
            df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
        
        # Replace missing or zero prices with appropriate values
        zero_mask = (df[col] == 0) | df[col].isna()
        zero_count = zero_mask.sum()
        
        if zero_count > 0:
            print(f"Found {zero_count} zero or missing values in {col}")
            
            if have_categories:
                # Fill by category median
                for product_id in df.loc[zero_mask, 'product_id'].unique():
                    if product_id in category_map:
                        category = category_map[product_id]
                        
                        # Find median price for this category
                        category_products = []
                        for pid, cat in category_map.items():
                            if cat == category:
                                category_products.append(pid)
                                
                        category_mask = df['product_id'].isin(category_products)
                        category_prices = df.loc[category_mask & ~zero_mask, col]
                        
                        if len(category_prices) > 0:
                            median_price = category_prices.median()
                            product_mask = (df['product_id'] == product_id) & zero_mask
                            df.loc[product_mask, col] = median_price
            
            # For any remaining missing values, use product-specific medians
            for product_id in df.loc[zero_mask, 'product_id'].unique():
                product_mask = df['product_id'] == product_id
                product_prices = df.loc[product_mask & ~zero_mask, col]
                
                if len(product_prices) > 0:
                    median_price = product_prices.median()
                    mask = product_mask & zero_mask
                    df.loc[mask, col] = median_price
            
            # Any remaining missing values get global median
            global_median = df.loc[~zero_mask, col].median()
            if pd.isna(global_median):
                if col == 'price':
                    global_median = 50.0  # Reasonable default for price
                elif col == 'shipping_charges' or col == 'freight_value':
                    global_median = 10.0  # Reasonable default for shipping
                else:
                    global_median = 1.0
                    
            df[col] = df[col].fillna(global_median)
            
            # Replace any lingering zeros
            df.loc[df[col] == 0, col] = global_median
    
    # Detect and handle anomalies in price columns
    for col in available_price_cols:
        df = detect_and_handle_anomalies(df, col)
    
    # Add a total value column if price is available
    if 'price' in df.columns:
        if 'shipping_charges' in df.columns:
            df['total_value'] = df['price'] + df['shipping_charges']
        elif 'freight_value' in df.columns:
            df['total_value'] = df['price'] + df['freight_value']
        else:
            df['total_value'] = df['price']
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'processed_order_items.csv'), index=False)
    
    print("Order items data preprocessing complete")
    return df

def calculate_performance_metrics(orders_df, order_items_df=None, output_dir=None):
    """
    Calculate comprehensive supply chain performance metrics with improved accuracy.
    
    Args:
        orders_df: Processed orders DataFrame
        order_items_df: Optional processed order items DataFrame
        output_dir: Directory to save metrics (optional)
        
    Returns:
        Dictionary with performance metrics
    """
    print("Calculating performance metrics...")
    metrics = {}
    
    # Basic delivery metrics
    if 'on_time_delivery' in orders_df.columns:
        metrics['on_time_delivery_rate'] = orders_df['on_time_delivery'].mean() * 100
    else:
        metrics['on_time_delivery_rate'] = 85.0  # Default industry benchmark
        
    if 'processing_time' in orders_df.columns:
        metrics['avg_processing_time'] = orders_df['processing_time'].mean()
        metrics['processing_time_std'] = orders_df['processing_time'].std()
    else:
        metrics['avg_processing_time'] = 1.0
        metrics['processing_time_std'] = 0.3
        
    if 'delivery_days' in orders_df.columns:
        metrics['avg_delivery_days'] = orders_df['delivery_days'].mean()
        metrics['delivery_days_std'] = orders_df['delivery_days'].std()
    else:
        metrics['avg_delivery_days'] = 7.0
        metrics['delivery_days_std'] = 2.0
    
    # Perfect order rate calculation (accounting for all fulfillment aspects)
    # Industry standard: % of orders delivered complete, accurate, on time, undamaged
    perfect_rate_factors = {
        'on_time_factor': metrics['on_time_delivery_rate'] / 100,
        'accuracy_factor': 0.97,  # Assuming 97% order accuracy (typical industry benchmark)
        'completeness_factor': 0.99  # Assuming 99% order completeness
    }
    metrics['perfect_order_rate'] = (
        perfect_rate_factors['on_time_factor'] * 
        perfect_rate_factors['accuracy_factor'] * 
        perfect_rate_factors['completeness_factor'] * 100
    )
    
    # Inventory metrics
    metrics['inventory_turnover'] = 8.0  # Industry average
    metrics['days_of_supply'] = 365 / metrics['inventory_turnover']
    
    # Return rate (industry average)
    metrics['return_rate'] = 3.0
    
    # Calculate financial metrics if order_items data is available
    if order_items_df is not None:
        if 'price' in order_items_df.columns:
            avg_order_value = order_items_df.groupby('order_id')['price'].sum().mean()
            metrics['average_order_value'] = avg_order_value
            
            if 'shipping_charges' in order_items_df.columns:
                shipping_ratio = (
                    order_items_df.groupby('order_id')['shipping_charges'].sum().mean() / 
                    avg_order_value * 100
                )
                metrics['shipping_cost_ratio'] = shipping_ratio
    
    # Time series metrics - find peak order months and seasonality
    if 'order_purchase_timestamp' in orders_df.columns:
        orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.month
        monthly_counts = orders_df.groupby('order_month').size()
        peak_month = monthly_counts.idxmax()
        metrics['peak_order_month'] = peak_month
        metrics['peak_to_average_ratio'] = monthly_counts.max() / monthly_counts.mean()
    
    # Add data quality metrics
    metrics['estimated_values'] = []
    for key in ['on_time_delivery_rate', 'avg_processing_time', 'avg_delivery_days']:
        if key == 'on_time_delivery_rate' and 'on_time_delivery' not in orders_df.columns:
            metrics['estimated_values'].append(key)
        elif key == 'avg_processing_time' and 'processing_time' not in orders_df.columns:
            metrics['estimated_values'].append(key)
        elif key == 'avg_delivery_days' and 'delivery_days' not in orders_df.columns:
            metrics['estimated_values'].append(key)
            
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    print("Performance metrics calculation complete")
    return metrics

def main():
    data_dir = "."
    output_dir = "./processed_data"
    print("Loading data files...")
    try:
        orders_df = pd.read_csv(os.path.join(data_dir, "df_Orders.csv"))
        order_items_df = pd.read_csv(os.path.join(data_dir, "df_OrderItems.csv"))
        products_df = pd.read_csv(os.path.join(data_dir, "df_Products.csv"))
        customers_df = pd.read_csv(os.path.join(data_dir, "df_Customers.csv"))
        payments_df = pd.read_csv(os.path.join(data_dir, "df_Payments.csv"))
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    processed_orders, delivery_metrics = preprocess_order_data(orders_df, output_dir)
    processed_products = preprocess_product_data(products_df, output_dir)
    processed_order_items = preprocess_order_items(order_items_df, output_dir)
    performance_metrics = calculate_performance_metrics(
        processed_orders, processed_order_items, output_dir
    )
    print("Data preprocessing complete")
    print(f"Performance metrics saved to {output_dir}/performance_metrics.csv")

if __name__ == "__main__":
    main()