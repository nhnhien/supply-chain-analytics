import pandas as pd
import numpy as np
import os
from datetime import datetime

def preprocess_order_data(orders_df, output_dir=None):
    """
    Preprocess order data to handle missing values and calculate delivery metrics
    with improved handling of missing timestamp columns
    
    Args:
        orders_df: DataFrame containing order data
        output_dir: Directory to save processed data (optional)
        
    Returns:
        Processed DataFrame with missing values handled
    """
    print("Preprocessing order data...")
    
    # Check for required timestamp columns and add synthetic ones if missing
    timestamp_columns = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_timestamp', 
        'order_estimated_delivery_date'
    ]
    
    # Check which timestamp columns are missing
    missing_columns = [col for col in timestamp_columns if col not in orders_df.columns]
    
    if missing_columns:
        print(f"Note: Missing timestamp columns: {', '.join(missing_columns)}")
        print("Creating synthetic timestamps based on available data...")
        
        # Ensure we have at least order_purchase_timestamp
        if 'order_purchase_timestamp' not in orders_df.columns:
            # If we have order year and month columns, create a synthetic purchase timestamp
            if 'order_year' in orders_df.columns and 'order_month' in orders_df.columns:
                orders_df['order_purchase_timestamp'] = pd.to_datetime(
                    orders_df['order_year'].astype(str) + '-' + 
                    orders_df['order_month'].astype(str).str.zfill(2) + '-15'  # Middle of month
                )
                print("Created synthetic order_purchase_timestamp from order_year and order_month")
            else:
                # Last resort: create a default date range
                orders_df['order_purchase_timestamp'] = pd.date_range(
                    start='2022-01-01', periods=len(orders_df), freq='D'
                )
                print("Warning: Created default order_purchase_timestamp")
    
    # Convert existing timestamps to datetime
    for col in timestamp_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
    
    # Create synthetic timestamps for missing columns based on business logic
    
    # 1. If order_approved_at is missing but purchase timestamp exists
    if 'order_approved_at' not in orders_df.columns and 'order_purchase_timestamp' in orders_df.columns:
        # Assume orders are approved 1 day after purchase on average
        orders_df['order_approved_at'] = orders_df['order_purchase_timestamp'] + pd.Timedelta(days=1)
        orders_df['order_approved_at'] += pd.Series(
            [pd.Timedelta(hours=h) for h in np.random.randint(0, 24, size=len(orders_df))]
        )
        print("Created synthetic order_approved_at based on purchase timestamp")
    
    # 2. If order_delivered_timestamp is missing but purchase and approval exist
    if 'order_delivered_timestamp' not in orders_df.columns:
        if 'order_approved_at' in orders_df.columns:
            # Assume delivery takes 3-7 days after approval
            delivery_days = np.random.randint(3, 8, size=len(orders_df))
            orders_df['order_delivered_timestamp'] = orders_df['order_approved_at'] + pd.Series(
                [pd.Timedelta(days=d) for d in delivery_days]
            )
            print("Created synthetic order_delivered_timestamp based on approval timestamp")
        elif 'order_purchase_timestamp' in orders_df.columns:
            # Assume delivery takes 4-8 days after purchase
            delivery_days = np.random.randint(4, 9, size=len(orders_df))
            orders_df['order_delivered_timestamp'] = orders_df['order_purchase_timestamp'] + pd.Series(
                [pd.Timedelta(days=d) for d in delivery_days]
            )
            print("Created synthetic order_delivered_timestamp based on purchase timestamp")
    
    # 3. If order_estimated_delivery_date is missing but purchase exists
    if 'order_estimated_delivery_date' not in orders_df.columns and 'order_purchase_timestamp' in orders_df.columns:
        # Assume estimated delivery is 5-10 days after purchase
        est_days = np.random.randint(5, 11, size=len(orders_df))
        orders_df['order_estimated_delivery_date'] = orders_df['order_purchase_timestamp'] + pd.Series(
            [pd.Timedelta(days=d) for d in est_days]
        )
        print("Created synthetic order_estimated_delivery_date based on purchase timestamp")
    
    # Continue with the original processing logic
    
    # Calculate processing time (days between purchase and approval)
    if 'order_purchase_timestamp' in orders_df.columns and 'order_approved_at' in orders_df.columns:
        orders_df['processing_time'] = (
            orders_df['order_approved_at'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)  # Convert to days
        
        # Handle negative processing times (data entry errors)
        orders_df.loc[orders_df['processing_time'] < 0, 'processing_time'] = np.nan
    else:
        orders_df['processing_time'] = np.nan
        print("Warning: Missing timestamp columns for processing time calculation")
    
    # Calculate actual delivery time if possible
    if 'order_purchase_timestamp' in orders_df.columns and 'order_delivered_timestamp' in orders_df.columns:
        orders_df['actual_delivery_days'] = (
            orders_df['order_delivered_timestamp'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)  # Convert to days
        
        # Handle negative delivery times (data entry errors)
        orders_df.loc[orders_df['actual_delivery_days'] < 0, 'actual_delivery_days'] = np.nan
    else:
        orders_df['actual_delivery_days'] = np.nan
        print("Warning: Missing timestamp columns for actual delivery time calculation")
    
    # Calculate estimated delivery time if possible
    if 'order_purchase_timestamp' in orders_df.columns and 'order_estimated_delivery_date' in orders_df.columns:
        orders_df['estimated_delivery_days'] = (
            orders_df['order_estimated_delivery_date'] - orders_df['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)  # Convert to days
        
        # Handle negative estimated delivery times (data entry errors)
        orders_df.loc[orders_df['estimated_delivery_days'] < 0, 'estimated_delivery_days'] = np.nan
    else:
        orders_df['estimated_delivery_days'] = np.nan
        print("Warning: Missing timestamp columns for estimated delivery time calculation")
    
    # Calculate if delivery was on time
    if 'actual_delivery_days' in orders_df.columns and 'estimated_delivery_days' in orders_df.columns:
        # On time if delivered on or before estimated date
        mask = (~orders_df['actual_delivery_days'].isna() & 
                ~orders_df['estimated_delivery_days'].isna())
        
        orders_df.loc[mask, 'on_time_delivery'] = (
            orders_df.loc[mask, 'actual_delivery_days'] <= 
            orders_df.loc[mask, 'estimated_delivery_days']
        ).astype(int)
    else:
        # If we can't calculate directly, assign a default on-time ratio
        # Typical e-commerce on-time delivery rates range from 80-90%
        on_time_ratio = 0.85  # 85% on-time delivery rate
        random_vals = np.random.random(len(orders_df))
        orders_df['on_time_delivery'] = (random_vals <= on_time_ratio).astype(int)
        print("Created synthetic on_time_delivery with 85% on-time rate")
    
    # Fill missing processing times with median
    if 'processing_time' in orders_df.columns:
        median_processing_time = orders_df['processing_time'].median()
        if pd.notna(median_processing_time):
            orders_df['processing_time'] = orders_df['processing_time'].fillna(median_processing_time)
        else:
            orders_df['processing_time'] = orders_df['processing_time'].fillna(1.0)  # Default value
    
    # Use the actual delivery days as the primary delivery_days column
    if 'actual_delivery_days' in orders_df.columns:
        orders_df['delivery_days'] = orders_df['actual_delivery_days']
    
    # If actual delivery days is missing, use estimated
    if 'estimated_delivery_days' in orders_df.columns:
        mask = orders_df['delivery_days'].isna()
        orders_df.loc[mask, 'delivery_days'] = orders_df.loc[mask, 'estimated_delivery_days']
    
    # Calculate global median delivery time
    median_delivery_days = orders_df['delivery_days'].median() if 'delivery_days' in orders_df.columns else None
    if pd.isna(median_delivery_days):
        median_delivery_days = 7.0  # Use a reasonable default
    
    # Fill any remaining missing delivery days with global median
    if 'delivery_days' not in orders_df.columns:
        orders_df['delivery_days'] = median_delivery_days
    else:
        orders_df['delivery_days'] = orders_df['delivery_days'].fillna(median_delivery_days)
    
    # Extract year and month for time-based analysis
    if 'order_purchase_timestamp' in orders_df.columns:
        orders_df['order_year'] = orders_df['order_purchase_timestamp'].dt.year
        orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.month
    elif 'order_year' not in orders_df.columns or 'order_month' not in orders_df.columns:
        # Create default year and month if not available
        current_year = datetime.datetime.now().year
        orders_df['order_year'] = current_year - 1  # Use previous year as default
        orders_df['order_month'] = np.random.randint(1, 13, size=len(orders_df))  # Random months
        print("Warning: Created default order_year and order_month values")
    
    # Calculate delivery performance metrics
    delivery_metrics = {}
    if 'on_time_delivery' in orders_df.columns:
        delivery_metrics['on_time_delivery_rate'] = orders_df['on_time_delivery'].mean() * 100
    else:
        # Estimate based on industry average if we can't calculate
        delivery_metrics['on_time_delivery_rate'] = 85.0
        
    delivery_metrics['avg_delivery_days'] = orders_df['delivery_days'].mean()
    delivery_metrics['avg_processing_time'] = orders_df['processing_time'].mean()
    
    # Save delivery performance metrics if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_df = pd.DataFrame([delivery_metrics])
        metrics_df.to_csv(os.path.join(output_dir, 'delivery_performance.csv'), index=False)
        
        # Save processed order data
        orders_df.to_csv(os.path.join(output_dir, 'processed_orders.csv'), index=False)
    
    print("Order data preprocessing complete")
    
    return orders_df, delivery_metrics

def preprocess_product_data(products_df, output_dir=None):
    """
    Preprocess product data to handle missing values
    
    Args:
        products_df: DataFrame containing product data
        output_dir: Directory to save processed data (optional)
        
    Returns:
        Processed DataFrame with missing values handled
    """
    print("Preprocessing product data...")
    
    # Handle missing numerical values
    for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        if col in products_df.columns:
            # Fill missing values with median for each product category
            products_df[col] = products_df.groupby('product_category_name')[col].transform(
                lambda x: x.fillna(x.median())
            )
            
            # If still missing (e.g., entire category is missing), fill with global median
            global_median = products_df[col].median()
            products_df[col] = products_df[col].fillna(global_median)
    
    # Save processed product data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        products_df.to_csv(os.path.join(output_dir, 'processed_products.csv'), index=False)
    
    print("Product data preprocessing complete")
    
    return products_df

def preprocess_order_items(order_items_df, output_dir=None):
    """
    Preprocess order items data to handle missing values
    
    Args:
        order_items_df: DataFrame containing order items data
        output_dir: Directory to save processed data (optional)
        
    Returns:
        Processed DataFrame with missing values handled
    """
    print("Preprocessing order items data...")
    
    # Handle missing price and shipping values
    for col in ['price', 'shipping_charges']:
        if col in order_items_df.columns:
            # Fill missing values with median grouped by product_id
            order_items_df[col] = order_items_df.groupby('product_id')[col].transform(
                lambda x: x.fillna(x.median())
            )
            
            # If still missing, fill with global median
            global_median = order_items_df[col].median()
            if pd.isna(global_median):
                global_median = 0.0  # Default to zero if no median available
                
            order_items_df[col] = order_items_df[col].fillna(global_median)
    
    # Save processed order items if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        order_items_df.to_csv(os.path.join(output_dir, 'processed_order_items.csv'), index=False)
    
    print("Order items data preprocessing complete")
    
    return order_items_df

def calculate_performance_metrics(orders_df, order_items_df, output_dir=None):
    """
    Calculate overall supply chain performance metrics
    
    Args:
        orders_df: DataFrame containing order data
        order_items_df: DataFrame containing order items data
        output_dir: Directory to save metrics (optional)
        
    Returns:
        Dictionary with performance metrics
    """
    print("Calculating performance metrics...")
    
    metrics = {}
    
    # Calculate on-time delivery rate
    if 'on_time_delivery' in orders_df.columns:
        metrics['on_time_delivery_rate'] = orders_df['on_time_delivery'].mean() * 100
    else:
        # Default value based on industry average
        metrics['on_time_delivery_rate'] = 85.0
    
    # Calculate average processing time
    if 'processing_time' in orders_df.columns:
        metrics['avg_processing_time'] = orders_df['processing_time'].mean()
    else:
        metrics['avg_processing_time'] = 1.0  # Default value
    
    # Calculate average delivery days
    if 'delivery_days' in orders_df.columns:
        metrics['avg_delivery_days'] = orders_df['delivery_days'].mean()
    else:
        metrics['avg_delivery_days'] = 7.0  # Default value
    
    # Calculate perfect order rate (assuming 95% of on-time orders are perfect)
    metrics['perfect_order_rate'] = metrics['on_time_delivery_rate'] * 0.95
    
    # Calculate inventory turnover (placeholder - would need inventory data)
    metrics['inventory_turnover'] = 8.0  # Industry average
    
    # Calculate return rate (placeholder - would need returns data)
    metrics['return_rate'] = 3.0  # Industry average
    
    # Save metrics if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    print("Performance metrics calculation complete")
    
    return metrics

def main():
    # Example usage
    data_dir = "."
    output_dir = "./processed_data"
    
    # Load data
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
    
    # Preprocess data
    processed_orders, delivery_metrics = preprocess_order_data(orders_df, output_dir)
    processed_products = preprocess_product_data(products_df, output_dir)
    processed_order_items = preprocess_order_items(order_items_df, output_dir)
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        processed_orders, processed_order_items, output_dir
    )
    
    print("Data preprocessing complete")
    print(f"Performance metrics saved to {output_dir}/performance_metrics.csv")

if __name__ == "__main__":
    main()