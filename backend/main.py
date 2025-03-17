#!/usr/bin/env python3
"""
Supply Chain Analytics for Demand Forecasting
Main application script that orchestrates the entire analytics pipeline.

This script integrates data loading, preprocessing, time series forecasting,
and visualization components to provide a comprehensive analysis of
e-commerce supply chain data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
from datetime import datetime

# Import our modules (you would need the other Python files in the same directory)
from arima_forecasting import DemandForecaster
from visualization_module import SupplyChainVisualizer
from data_preprocessing import preprocess_order_data, preprocess_product_data, preprocess_order_items, calculate_performance_metrics

def calculate_delivery_days(orders, supply_chain=None):
    """
    Calculate delivery days based on timestamp data.
    
    Args:
        orders: DataFrame containing order data.
        supply_chain: Optional unified supply chain DataFrame for category/state-based estimations.
    
    Returns:
        Updated orders DataFrame with delivery_days calculated.
    """
    # If delivery_days already exists and has valid values, use them (potentially filling gaps later)
    if 'delivery_days' in orders.columns and not orders['delivery_days'].isna().all():
        pass
    else:
        # Convert timestamp columns to datetime
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
        
        # Calculate processing time (days between purchase and approval)
        orders['processing_time'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.days
        
        # Calculate actual delivery days if delivery timestamp is available
        if 'order_delivered_customer_date' in orders.columns:
            orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
            mask = orders['order_delivered_customer_date'].notna() & orders['order_purchase_timestamp'].notna()
            orders.loc[mask, 'delivery_days'] = (
                orders.loc[mask, 'order_delivered_customer_date'] - 
                orders.loc[mask, 'order_purchase_timestamp']
            ).dt.days
        else:
            # Otherwise, use estimated delivery dates if available
            if 'order_estimated_delivery_date' in orders.columns:
                orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
                mask = orders['order_estimated_delivery_date'].notna() & orders['order_purchase_timestamp'].notna()
                orders.loc[mask, 'estimated_delivery_days'] = (
                    orders.loc[mask, 'order_estimated_delivery_date'] - 
                    orders.loc[mask, 'order_purchase_timestamp']
                ).dt.days
                # Fill delivery_days with estimated values where actual values are missing
                orders['delivery_days'] = orders.get('delivery_days', pd.Series(index=orders.index)).fillna(orders['estimated_delivery_days'])
            else:
                # Create placeholder column if delivery_days does not exist
                if 'delivery_days' not in orders.columns:
                    orders['delivery_days'] = None
    
    # Use supply_chain data to fill in missing delivery_days if available
    if supply_chain is not None and orders['delivery_days'].isna().any():
        # Use category-level medians if product_category_name is available
        if 'product_category_name' in supply_chain.columns:
            # Calculate median delivery days per category using valid values
            category_medians = supply_chain.dropna(subset=['delivery_days']).groupby('product_category_name')['delivery_days'].median().to_dict()
            
            # Merge orders with category info from supply_chain using order_id
            if 'order_id' in supply_chain.columns:
                order_categories = supply_chain[['order_id', 'product_category_name']].drop_duplicates()
                orders_with_categories = orders.merge(order_categories, on='order_id', how='left')
                
                # For each category, update orders that are missing delivery_days
                for category, median in category_medians.items():
                    if pd.notna(median):
                        mask = (orders_with_categories['product_category_name'] == category) & (orders_with_categories['delivery_days'].isna())
                        # Extract the matching order IDs from the merged DataFrame
                        matching_order_ids = orders_with_categories.loc[mask, 'order_id']
                        orders.loc[orders['order_id'].isin(matching_order_ids), 'delivery_days'] = median
        
        # Fill remaining NAs using customer state medians if available
        if 'customer_state' in supply_chain.columns and orders['delivery_days'].isna().any():
            state_medians = supply_chain.dropna(subset=['delivery_days']).groupby('customer_state')['delivery_days'].median().to_dict()
            
            if 'order_id' in supply_chain.columns:
                order_states = supply_chain[['order_id', 'customer_state']].drop_duplicates()
                orders_with_states = orders.merge(order_states, on='order_id', how='left')
                
                for state, median in state_medians.items():
                    if pd.notna(median):
                        mask = (orders_with_states['customer_state'] == state) & (orders_with_states['delivery_days'].isna())
                        matching_order_ids = orders_with_states.loc[mask, 'order_id']
                        orders.loc[orders['order_id'].isin(matching_order_ids), 'delivery_days'] = median
    
    # Fill any remaining missing values with the global median or a default value
    global_median = orders['delivery_days'].median()
    if not pd.isna(global_median):
        orders['delivery_days'] = orders['delivery_days'].fillna(global_median)
    else:
        orders['delivery_days'] = orders['delivery_days'].fillna(7)
    
    # Clip delivery days to be within reasonable bounds (1 to 30 days)
    orders['delivery_days'] = orders['delivery_days'].clip(1, 30)
    
    return orders

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Supply Chain Analytics for Demand Forecasting')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    parser.add_argument('--use-auto-arima', action='store_true', help='Use auto ARIMA parameter selection')
    parser.add_argument('--seasonal', action='store_true', help='Include seasonality in the model')
    parser.add_argument('--supplier-clusters', type=int, default=3, help='Number of supplier clusters to create')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_top_categories(data, limit=15):
    """
    Get top categories by total demand
    
    Args:
        data: Monthly demand data
        limit: Maximum number of categories to return
        
    Returns:
        Array of top category names
    """
    # Group by category and sum the demand
    category_totals = {}
    
    for _, row in data.iterrows():
        category = row.get('product_category_name')
        count = float(row.get('count', 0) or row.get('order_count', 0) or 0)
        
        if category and category not in category_totals:
            category_totals[category] = 0
        
        if category:
            category_totals[category] += count
    
    # Sort categories by total demand and get top N
    return [category for category, _ in sorted(
        category_totals.items(), 
        key=lambda item: item[1], 
        reverse=True
    )[:limit]]

def run_pandas_analysis(args):
    """
    Run supply chain analysis using pandas (for smaller datasets)
    
    Args:
        args: Command line arguments
    """
    print("Running analysis with pandas...")
    
    # Load data
    print("Loading data...")
    customers = pd.read_csv(os.path.join(args.data_dir, 'df_Customers.csv'))
    order_items = pd.read_csv(os.path.join(args.data_dir, 'df_OrderItems.csv'))
    orders = pd.read_csv(os.path.join(args.data_dir, 'df_Orders.csv'))
    payments = pd.read_csv(os.path.join(args.data_dir, 'df_Payments.csv'))
    products = pd.read_csv(os.path.join(args.data_dir, 'df_Products.csv'))
    
    # Create output directory for results
    ensure_directory(args.output_dir)
    
    # Preprocess data
    print("Preprocessing data...")
    orders, delivery_metrics = preprocess_order_data(orders, args.output_dir)
    products = preprocess_product_data(products, args.output_dir)
    order_items = preprocess_order_items(order_items, args.output_dir)
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(orders, order_items, args.output_dir)
    
    # Make sure delivery days calculation is complete
    orders = calculate_delivery_days(orders)
    
    # Merge datasets
    print("Building unified dataset...")
    # Join orders with order items
    supply_chain = orders.merge(order_items, on='order_id', how='inner')
    
    # Join with products
    supply_chain = supply_chain.merge(products, on='product_id', how='left')
    
    # Join with customers
    supply_chain = supply_chain.merge(customers, on='customer_id', how='left')
    
    # Update delivery days with category and state information
    orders = calculate_delivery_days(orders, supply_chain)
    
    # Rebuild supply_chain with updated orders
    supply_chain = orders.merge(order_items, on='order_id', how='inner')
    supply_chain = supply_chain.merge(products, on='product_id', how='left')
    supply_chain = supply_chain.merge(customers, on='customer_id', how='left')
    
    # Analyze monthly demand
    print("Analyzing monthly demand patterns...")
    monthly_demand = supply_chain.groupby(['product_category_name', 'order_year', 'order_month']).size().reset_index(name='count')
    
    # Save monthly demand data
    monthly_demand.to_csv(os.path.join(args.output_dir, 'monthly_demand.csv'), index=False)
    
    # Get top categories by volume, not alphabetically
    top_categories = get_top_categories(monthly_demand, args.top_n)
    print(f"Top {len(top_categories)} categories by volume: {', '.join(top_categories)}")
    
    # Save top categories for reference
    pd.DataFrame({'product_category_name': top_categories}).to_csv(
        os.path.join(args.output_dir, 'top_categories.csv'), index=False
    )
    
    # Run time series forecasting
    print("Running time series forecasting...")
    forecaster = DemandForecaster(os.path.join(args.output_dir, 'monthly_demand.csv'))
    forecaster.preprocess_data()
    
    # Use auto ARIMA if specified
    if args.use_auto_arima:
        print("Using auto ARIMA for parameter selection")
        forecaster.use_auto_arima = True
    
    # Include seasonality if specified
    if args.seasonal:
        print("Including seasonality in the model")
        forecaster.seasonal = True
        forecaster.seasonal_period = 12  # Monthly data
    
    forecasts = forecaster.run_all_forecasts(top_n=args.top_n, periods=args.forecast_periods)
    
    # Generate forecast report
    forecast_report = forecaster.generate_forecast_report(os.path.join(args.output_dir, 'forecast_report.csv'))
    
    # Calculate seller performance using real data
    print("Analyzing seller performance...")
    seller_performance = supply_chain.groupby('seller_id').agg({
        'order_id': 'count',
        'processing_time': 'mean',
        'delivery_days': 'mean',
        'price': 'sum',
        'shipping_charges': 'sum'
    }).reset_index()
    
    seller_performance.columns = ['seller_id', 'order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales', 'shipping_costs']
    
    # Add additional performance metrics
    seller_performance['avg_order_value'] = seller_performance['total_sales'] / seller_performance['order_count']
    seller_performance['shipping_ratio'] = seller_performance['shipping_costs'] / seller_performance['total_sales'] * 100
    
    # Calculate on-time delivery rate for each seller
    seller_on_time = supply_chain.groupby('seller_id')['on_time_delivery'].mean().reset_index()
    seller_on_time.columns = ['seller_id', 'on_time_delivery_rate']
    seller_performance = seller_performance.merge(seller_on_time, on='seller_id', how='left')
    
    # Add on-time delivery rate and convert to percentage
    seller_performance['on_time_delivery_rate'] = seller_performance['on_time_delivery_rate'].fillna(0.5) * 100
    
    # Calculate seller score - higher is better
    seller_performance['seller_score'] = (
        # Normalized order count (more is better)
        (seller_performance['order_count'] / seller_performance['order_count'].max()) * 25 +
        # Normalized avg order value (higher is better)
        (seller_performance['avg_order_value'] / seller_performance['avg_order_value'].max()) * 25 +
        # Normalized processing time (lower is better)
        (1 - (seller_performance['avg_processing_time'] / seller_performance['avg_processing_time'].max())) * 25 +
        # Normalized on-time delivery (higher is better)
        (seller_performance['on_time_delivery_rate'] / 100) * 25
    )
    
    # Perform improved clustering based on seller score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Select features for clustering
    clustering_features = ['order_count', 'avg_processing_time', 'avg_delivery_days', 
                          'total_sales', 'avg_order_value', 'on_time_delivery_rate']
    
    # Subset data for clustering
    clustering_data = seller_performance[clustering_features].copy()
    
    # Handle missing values
    for col in clustering_data.columns:
        if clustering_data[col].isna().any():
            clustering_data[col] = clustering_data[col].fillna(clustering_data[col].median())
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Determine the number of clusters using args
    n_clusters = args.supplier_clusters
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    seller_performance['prediction'] = kmeans.fit_predict(scaled_data)
    
    # Calculate cluster centers for interpretation
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=clustering_features
    )
    
    # Label the clusters based on seller score
    cluster_avg_scores = seller_performance.groupby('prediction')['seller_score'].mean().sort_values(ascending=False)
    cluster_to_performance = {}
    
    # Map clusters to performance labels (0: High, 1: Medium, 2: Low)
    for i, (cluster, _) in enumerate(cluster_avg_scores.items()):
        if i == 0:
            cluster_to_performance[cluster] = 0  # High performers
        elif i == 1:
            cluster_to_performance[cluster] = 1  # Medium performers
        else:
            cluster_to_performance[cluster] = 2  # Low performers
    
    # Apply the mapping
    seller_performance['performance_cluster'] = seller_performance['prediction'].map(cluster_to_performance)
    
    # Save cluster interpretation
    cluster_interpretation = pd.DataFrame({
        'original_cluster': list(cluster_to_performance.keys()),
        'performance_level': [
            'High Performer' if v == 0 else 'Medium Performer' if v == 1 else 'Low Performer'
            for v in cluster_to_performance.values()
        ],
        'avg_score': [cluster_avg_scores[k] for k in cluster_to_performance.keys()]
    })
    
    cluster_interpretation.to_csv(os.path.join(args.output_dir, 'cluster_interpretation.csv'), index=False)
    
    # Save seller performance data for the dashboard frontend
    seller_performance.to_csv(os.path.join(args.output_dir, 'seller_clusters.csv'), index=False)
    cluster_centers.to_csv(os.path.join(args.output_dir, 'cluster_centers.csv'), index=False)
    
    # Generate geographical metrics
    print("Analyzing geographical patterns...")
    state_metrics = supply_chain.groupby('customer_state').agg({
        'order_id': 'count',
        'processing_time': 'mean',
        'delivery_days': 'mean',
        'price': 'sum',
        'on_time_delivery': 'mean'
    }).reset_index()

    state_metrics.columns = ['customer_state', 'order_count', 'avg_processing_time', 
                           'avg_delivery_days', 'total_sales', 'on_time_delivery_rate']
    
    # Convert on-time delivery rate to percentage
    state_metrics['on_time_delivery_rate'] = state_metrics['on_time_delivery_rate'] * 100
    
    state_metrics.to_csv(os.path.join(args.output_dir, 'state_metrics.csv'), index=False)

    # Generate top category by state
    top_category_by_state = []
    for state in state_metrics['customer_state'].unique():
        state_data = supply_chain[supply_chain['customer_state'] == state]
        if len(state_data) > 0:
            top_cat = state_data.groupby('product_category_name')['order_id'].count().reset_index()
            if not top_cat.empty:
                top_cat = top_cat.sort_values('order_id', ascending=False).iloc[0]
                top_category_by_state.append({
                    'customer_state': state,
                    'product_category_name': top_cat['product_category_name'],
                    'order_count': top_cat['order_id']
                })

    if top_category_by_state:
        pd.DataFrame(top_category_by_state).to_csv(os.path.join(args.output_dir, 'top_category_by_state.csv'), index=False)
    
    # Generate recommendations
    print("Generating supply chain recommendations...")
    # Create safety stock and reorder point recommendations based on actual data
    recommendations = []
    
    for category in top_categories:
        if category in forecasts:
            # Get average monthly demand
            cat_demand = monthly_demand[monthly_demand['product_category_name'] == category]['count'].mean()
            
            # Calculate safety stock based on variability in the data
            cat_std = monthly_demand[monthly_demand['product_category_name'] == category]['count'].std()
            # Use coefficient of variation to determine safety factor
            cv = cat_std / cat_demand if cat_demand > 0 else 0.5
            
            # More sophisticated safety stock calculation based on service level
            # Assuming 95% service level (Z=1.645 for normal distribution)
            z_score = 1.645
            
            # Calculate lead time factor
            category_delivery = supply_chain[supply_chain['product_category_name'] == category]['delivery_days'].mean()
            lead_time_days = category_delivery if not pd.isna(category_delivery) else 7  # Default to 7 days
            lead_time_fraction = lead_time_days / 30.0  # As fraction of month
            
            # Safety stock calculation with lead time consideration
            safety_stock = z_score * cat_std * np.sqrt(lead_time_fraction)
            
            # Ensure safety stock isn't too small
            safety_stock = max(safety_stock, 0.3 * cat_demand)
            
            # Calculate reorder point
            reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            
            # Get next month forecast
            next_month_forecast = forecasts[category]['forecast'].iloc[0] if len(forecasts[category]) > 0 else cat_demand
            
            # Determine growth rate
            if cat_demand > 0:
                growth_rate = ((next_month_forecast - cat_demand) / cat_demand) * 100
            else:
                growth_rate = 0
            
            # Calculate order frequency based on demand and inventory holding cost
            # Assuming holding cost is 20% of item value annually
            # Economic Order Quantity (EOQ) formula
            annual_demand = cat_demand * 12
            order_cost = 50  # Assumed fixed cost per order
            holding_cost_pct = 0.2  # 20% annual holding cost
            avg_item_cost = supply_chain[supply_chain['product_category_name'] == category]['price'].mean()
            
            if pd.notna(avg_item_cost) and avg_item_cost > 0 and annual_demand > 0:
                holding_cost = avg_item_cost * holding_cost_pct
                eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
                order_frequency = annual_demand / eoq  # Orders per year
                days_between_orders = 365 / order_frequency
            else:
                days_between_orders = 30  # Default monthly ordering
            
            recommendations.append({
                'product_category': category,
                'category': category,  # Add alternative column name for compatibility
                'avg_monthly_demand': cat_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': next_month_forecast,
                'forecast_demand': next_month_forecast,  # Add alternative column name
                'growth_rate': growth_rate,
                'lead_time_days': lead_time_days,
                'days_between_orders': days_between_orders,
                'avg_item_cost': avg_item_cost
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(os.path.join(args.output_dir, 'reorder_recommendations.csv'), index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = SupplyChainVisualizer(output_dir=args.output_dir)
    
    # Generate date field for visualization
    monthly_demand['date'] = pd.to_datetime(
        monthly_demand['order_year'].astype(str) + '-' + 
        monthly_demand['order_month'].astype(str).str.zfill(2) + '-01'
    )
    
    # Visualize demand trends
    visualizer.visualize_demand_trends(monthly_demand, top_categories[:10])  # Limit to top 10 for readability
    
    # Create heatmaps for top categories
    for category in top_categories[:5]:  # Limit to top 5
        visualizer.create_demand_heatmap(monthly_demand, category)
    
    # Visualize seller clusters
    visualizer.visualize_seller_clusters(seller_performance)
    
    # Visualize recommendations
    visualizer.visualize_reorder_recommendations(recommendations_df)
    
    # Create dashboard
    visualizer.create_supply_chain_dashboard(
        monthly_demand[monthly_demand['product_category_name'].isin(top_categories[:10])],
        seller_performance,
        forecasts,
        recommendations_df
    )
    
    # Create performance summary
    performance_summary = pd.DataFrame({
        'metric': [
            'Total Orders',
            'Total Products', 
            'Total Sellers',
            'Average Order Value',
            'Average Processing Time',
            'Average Delivery Days',
            'On-Time Delivery Rate',
            'Forecast Accuracy (Avg MAPE)'
        ],
        'value': [
            len(orders),
            len(products),
            len(seller_performance),
            f"${supply_chain['price'].mean():.2f}",
            f"{supply_chain['processing_time'].mean():.2f} days",
            f"{supply_chain['delivery_days'].mean():.2f} days",
            f"{supply_chain['on_time_delivery'].mean() * 100:.2f}%",
            f"{forecast_report['mape'].mean():.2f}%"
        ]
    })
    
    performance_summary.to_csv(os.path.join(args.output_dir, 'performance_summary.csv'), index=False)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    return {
        'monthly_demand': monthly_demand,
        'top_categories': top_categories,
        'forecasts': forecasts,
        'seller_performance': seller_performance,
        'recommendations': recommendations_df,
        'state_metrics': state_metrics,  # Add state metrics to the results
        'performance_metrics': performance_metrics  # Add overall performance metrics
    }

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure warnings
    warnings.filterwarnings("ignore")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Run pandas analysis
    results = run_pandas_analysis(args)
    
    # Generate summary report
    print("Generating summary report...")
    
    report = [
        "# Supply Chain Analytics for Demand Forecasting",
        f"## Report Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Dataset Summary",
        f"- Data source: {args.data_dir}",
        f"- Top {args.top_n} product categories analyzed",
        f"- Forecast horizon: {args.forecast_periods} months",
        "",
        "### Key Findings",
    ]
    
    # Add top categories
    report.append("#### Top Product Categories by Demand")
    for i, category in enumerate(results['top_categories'][:10], 1):
        report.append(f"{i}. {category}")
    
    report.append("")
    
    # Add forecast summaries
    report.append("#### Demand Forecast Highlights")
    for category, forecast in list(results['forecasts'].items())[:5]:  # Top 5 forecasts
        avg_forecast = forecast['forecast'].mean()
        report.append(f"- {category}: Avg. forecasted demand {avg_forecast:.1f} units/month")
    
    report.append("")
    
    # Add reorder recommendations summary
    report.append("#### Inventory Recommendations")
    if isinstance(results['recommendations'], pd.DataFrame):
        # For pandas analysis
        recos = results['recommendations']
        for _, row in recos.head(5).iterrows():
            report.append(f"- {row['product_category']}: Reorder at {row['reorder_point']:.1f} units, Safety stock: {row['safety_stock']:.1f} units")
    
    report.append("")
    
    # Add performance metrics if available
    if 'performance_metrics' in results:
        report.append("#### Supply Chain Performance Metrics")
        metrics = results['performance_metrics']
        if 'avg_delivery_days' in metrics:
            report.append(f"- Average Delivery Time: {metrics['avg_delivery_days']:.1f} days")
        if 'on_time_delivery_rate' in metrics:
            report.append(f"- On-Time Delivery Rate: {metrics['on_time_delivery_rate']:.1f}%")
        if 'avg_processing_time' in metrics:
            report.append(f"- Average Order Processing Time: {metrics['avg_processing_time']:.1f} days")
        if 'average_order_value' in metrics:
            report.append(f"- Average Order Value: ${metrics['average_order_value']:.2f}")
        
        report.append("")
    
    # Add seller performance insights
    report.append("#### Seller Performance Insights")
    seller_clusters = results['seller_performance'].groupby('prediction').size().reset_index()
    seller_clusters.columns = ['cluster', 'count']
    
    total_sellers = seller_clusters['count'].sum()
    
    for _, row in seller_clusters.iterrows():
        cluster_num = row['cluster']
        cluster_size = row['count']
        percentage = (cluster_size / total_sellers) * 100
        
        cluster_label = "High Performers" if cluster_num == 0 else \
                       "Medium Performers" if cluster_num == 1 else \
                       "Low Performers"
        
        report.append(f"- {cluster_label}: {cluster_size} sellers ({percentage:.1f}%)")
    
    report.append("")
    
    report.append("### Visualization Files")
    
    # List output files
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png') or f.endswith('.csv')]
    for f in output_files[:15]:  # Limit to first 15 files
        report.append(f"- {f}")
    
    # Write report to file
    with open(os.path.join(args.output_dir, 'summary_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {os.path.join(args.output_dir, 'summary_report.md')}")

if __name__ == "__main__":
    main()