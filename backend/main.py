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
from spark_implementation import create_spark_session, load_ecommerce_data, process_orders, build_unified_supply_chain_dataset
from spark_implementation import analyze_product_demand, analyze_seller_efficiency, analyze_geographical_patterns
from spark_implementation import analyze_supply_chain_metrics, generate_supply_chain_recommendations
from data_preprocessing import preprocess_order_data, preprocess_product_data, preprocess_order_items, calculate_performance_metrics

def calculate_delivery_days(orders, supply_chain=None):
    """
    Calculate delivery days based on timestamp data
    
    Args:
        orders: DataFrame containing order data
        supply_chain: Optional unified supply chain DataFrame for category/state-based estimations
    
    Returns:
        Updated orders DataFrame with delivery_days calculated
    """
    # Check if preprocess_order_data already calculated delivery_days
    if 'delivery_days' in orders.columns and not orders['delivery_days'].isna().all():
        # If we already have some delivery days calculated, just fill in the gaps
        # using category and state information from supply_chain
        pass
    else:
        # Convert timestamp columns to datetime
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
        
        # Calculate processing time (days between purchase and approval)
        orders['processing_time'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.days
        
        # If order_delivered_timestamp exists, calculate actual delivery days
        if 'order_delivered_customer_date' in orders.columns:
            orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
            mask = orders['order_delivered_customer_date'].notna() & orders['order_purchase_timestamp'].notna()
            orders.loc[mask, 'delivery_days'] = (
                orders.loc[mask, 'order_delivered_customer_date'] - 
                orders.loc[mask, 'order_purchase_timestamp']
            ).dt.days
        else:
            # If we don't have delivery timestamp, estimate from order_estimated_delivery_date if available
            if 'order_estimated_delivery_date' in orders.columns:
                orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
                mask = orders['order_estimated_delivery_date'].notna() & orders['order_purchase_timestamp'].notna()
                orders.loc[mask, 'estimated_delivery_days'] = (
                    orders.loc[mask, 'order_estimated_delivery_date'] - 
                    orders.loc[mask, 'order_purchase_timestamp']
                ).dt.days
                # Use estimated days when actual is not available
                orders['delivery_days'] = orders.get('delivery_days', pd.Series(index=orders.index)).fillna(orders['estimated_delivery_days'])
            else:
                # Create a placeholder delivery_days column if it doesn't exist
                if 'delivery_days' not in orders.columns:
                    orders['delivery_days'] = None
    
    # If we have the supply_chain DataFrame and there are still missing values
    if supply_chain is not None and orders['delivery_days'].isna().any():
        # Use category level medians if product_category_name is available
        if 'product_category_name' in supply_chain.columns:
            # Ensure we're only using valid delivery_days for the median calculation
            category_medians = supply_chain.dropna(subset=['delivery_days']).groupby('product_category_name')['delivery_days'].median().to_dict()
            
            # Join orders with order_items and products to get category info
            if 'order_id' in supply_chain.columns:
                order_categories = supply_chain[['order_id', 'product_category_name']].drop_duplicates()
                orders_with_categories = orders.merge(order_categories, on='order_id', how='left')
                
                # Apply category medians
                for category, median in category_medians.items():
                    if pd.notna(median):
                        category_mask = (orders_with_categories['product_category_name'] == category) & (orders['delivery_days'].isna())
                        orders.loc[category_mask.index, 'delivery_days'] = median
        
        # Fill remaining NAs with customer state median if available
        if 'customer_state' in supply_chain.columns and orders['delivery_days'].isna().any():
            # Ensure we're only using valid delivery_days for the median calculation
            state_medians = supply_chain.dropna(subset=['delivery_days']).groupby('customer_state')['delivery_days'].median().to_dict()
            
            # Join orders with customers to get state info
            if 'order_id' in supply_chain.columns:
                order_states = supply_chain[['order_id', 'customer_state']].drop_duplicates()
                orders_with_states = orders.merge(order_states, on='order_id', how='left')
                
                # Apply state medians
                for state, median in state_medians.items():
                    if pd.notna(median):
                        state_mask = (orders_with_states['customer_state'] == state) & (orders['delivery_days'].isna())
                        orders.loc[state_mask.index, 'delivery_days'] = median
    
    # Fill any remaining NAs with global median
    global_median = orders['delivery_days'].median()
    if not pd.isna(global_median):
        orders['delivery_days'] = orders['delivery_days'].fillna(global_median)
    else:
        # As a last resort, use a reasonable industry standard (7 days)
        orders['delivery_days'] = orders['delivery_days'].fillna(7)
    
    # Ensure delivery days is within reasonable bounds (1-30 days)
    orders['delivery_days'] = orders['delivery_days'].clip(1, 30)
    
    return orders

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Supply Chain Analytics for Demand Forecasting')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    parser.add_argument('--use-spark', action='store_true', help='Use Apache Spark for processing')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

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
    
    # Get top categories
    top_categories = monthly_demand.groupby('product_category_name')['count'].sum().sort_values(ascending=False).head(args.top_n).index.tolist()
    
    # Run time series forecasting
    print("Running time series forecasting...")
    monthly_demand.to_csv(os.path.join(args.output_dir, 'monthly_demand.csv'), index=False)
    
    forecaster = DemandForecaster(os.path.join(args.output_dir, 'monthly_demand.csv'))
    forecaster.preprocess_data()
    forecasts = forecaster.run_all_forecasts(top_n=args.top_n, periods=args.forecast_periods)
    
    # Generate forecast report
    forecast_report = forecaster.generate_forecast_report(os.path.join(args.output_dir, 'forecast_report.csv'))
    
    # Calculate seller performance using real data
    print("Analyzing seller performance...")
    seller_performance = supply_chain.groupby('seller_id').agg({
        'order_id': 'count',
        'processing_time': 'mean',
        'delivery_days': 'mean',
        'price': 'sum'
    }).reset_index()
    
    seller_performance.columns = ['seller_id', 'order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
    
    # Perform manual clustering based on sales and processing time
    q75_sales = seller_performance['total_sales'].quantile(0.75)
    q25_sales = seller_performance['total_sales'].quantile(0.25)
    q75_time = seller_performance['avg_processing_time'].quantile(0.75)
    q25_time = seller_performance['avg_processing_time'].quantile(0.25)
    
    # Function to assign cluster
    def assign_cluster(row):
        if row['total_sales'] > q75_sales and row['avg_processing_time'] < q25_time:
            return 0  # High performers
        elif row['total_sales'] < q25_sales and row['avg_processing_time'] > q75_time:
            return 2  # Low performers
        else:
            return 1  # Average performers
    
    # Apply clustering
    seller_performance['prediction'] = seller_performance.apply(assign_cluster, axis=1)
    
    # Save seller performance data for the dashboard frontend
    seller_performance.to_csv(os.path.join(args.output_dir, 'seller_clusters.csv'), index=False)
    
    # Generate geographical metrics
    print("Analyzing geographical patterns...")
    state_metrics = supply_chain.groupby('customer_state').agg({
        'order_id': 'count',
        'processing_time': 'mean',
        'delivery_days': 'mean',
        'price': 'sum'
    }).reset_index()

    state_metrics.columns = ['customer_state', 'order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
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
            safety_factor = max(0.3, min(0.7, cv))  # Bound between 30-70%
            
            safety_stock = cat_demand * safety_factor
            
            # Calculate reorder point using actual lead time if available
            # Here we use a simple approach with the delivery days as a proxy for lead time
            category_delivery = supply_chain[supply_chain['product_category_name'] == category]['delivery_days'].mean()
            lead_time_days = category_delivery if not pd.isna(category_delivery) else 7  # Default to 7 days
            lead_time_fraction = lead_time_days / 30.0  # As fraction of month
            reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            
            # Get next month forecast
            next_month_forecast = forecasts[category]['forecast'].iloc[0] if len(forecasts[category]) > 0 else cat_demand
            
            recommendations.append({
                'product_category': category,
                'avg_monthly_demand': cat_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': next_month_forecast
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
    visualizer.visualize_demand_trends(monthly_demand, top_categories)
    
    # Create heatmaps for top categories
    for category in top_categories[:3]:  # Limit to top 3
        visualizer.create_demand_heatmap(monthly_demand, category)
    
    # Visualize seller clusters
    visualizer.visualize_seller_clusters(seller_performance)
    
    # Visualize recommendations
    visualizer.visualize_reorder_recommendations(recommendations_df)
    
    # Create dashboard
    visualizer.create_supply_chain_dashboard(
        monthly_demand[monthly_demand['product_category_name'].isin(top_categories)],
        seller_performance,
        forecasts,
        recommendations_df
    )
    
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

def run_spark_analysis(args):
    """
    Run supply chain analysis using Apache Spark (for larger datasets)
    
    Args:
        args: Command line arguments
    """
    print("Running analysis with Apache Spark...")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Load data
        print("Loading data...")
        orders, order_items, customers, products, payments = load_ecommerce_data(spark, args.data_dir)
        
        # Process orders - NOTE: You'll need to modify the process_orders function
        # in spark_implementation.py to use the new delivery days calculation logic
        print("Processing orders...")
        processed_orders = process_orders(orders)
        
        # Build unified dataset
        print("Building unified dataset...")
        supply_chain = build_unified_supply_chain_dataset(
            processed_orders, order_items, products, customers, payments
        )
        
        # Create output directory
        ensure_directory(args.output_dir)
        
        # Analyze demand patterns
        print("Analyzing demand patterns...")
        demand_by_category, top_categories, demand_growth = analyze_product_demand(supply_chain)
        
        # Convert to pandas for forecasting (statsmodels doesn't work with Spark DataFrames)
        demand_pandas = demand_by_category.toPandas()
        demand_pandas.to_csv(os.path.join(args.output_dir, 'demand_by_category.csv'), index=False)
        
        # Get top category names
        top_categories_list = [row['product_category_name'] for row in top_categories.limit(args.top_n).collect()]
        
        # Run time series forecasting
        print("Running time series forecasting...")
        forecaster = DemandForecaster(os.path.join(args.output_dir, 'demand_by_category.csv'))
        forecaster.preprocess_data()
        forecasts = forecaster.run_all_forecasts(top_n=args.top_n, periods=args.forecast_periods)
        
        # Generate forecast report
        forecast_report = forecaster.generate_forecast_report(os.path.join(args.output_dir, 'forecast_report.csv'))
        
        # Analyze seller efficiency
        print("Analyzing seller efficiency...")
        seller_metrics, seller_clusters, cluster_centers, performance_ranking = analyze_seller_efficiency(supply_chain)
        
        # Save results
        seller_metrics.toPandas().to_csv(os.path.join(args.output_dir, 'seller_metrics.csv'), index=False)
        seller_clusters.to_csv(os.path.join(args.output_dir, 'seller_clusters.csv'), index=False)
        
        # Analyze geographical patterns
        print("Analyzing geographical patterns...")
        state_metrics, top_category_by_state = analyze_geographical_patterns(supply_chain)
        
        # Save results
        state_metrics.toPandas().to_csv(os.path.join(args.output_dir, 'state_metrics.csv'), index=False)
        top_category_by_state.toPandas().to_csv(os.path.join(args.output_dir, 'top_category_by_state.csv'), index=False)
        
        # Calculate supply chain metrics
        print("Calculating supply chain performance metrics...")
        metrics = analyze_supply_chain_metrics(supply_chain)
        
        # Generate recommendations
        print("Generating supply chain recommendations...")
        recommendations = generate_supply_chain_recommendations(
            (demand_by_category, top_categories, demand_growth),
            (seller_metrics, seller_clusters, cluster_centers),
            metrics
        )
        
        # Save recommendations
        for key, df in recommendations.items():
            df.to_csv(os.path.join(args.output_dir, f'{key}.csv'), index=False)
        
        # Create visualizations
        print("Creating visualizations...")
        visualizer = SupplyChainVisualizer(output_dir=args.output_dir)
        
        # Generate date field for visualization
        demand_pandas['date'] = pd.to_datetime(
            demand_pandas['year'].astype(str) + '-' + 
            demand_pandas['month'].astype(str).str.zfill(2) + '-01'
        )
        
        # Visualize demand trends
        visualizer.visualize_demand_trends(demand_pandas, top_categories_list)
        
        # Create heatmaps for top categories
        for category in top_categories_list[:3]:  # Limit to top 3
            visualizer.create_demand_heatmap(demand_pandas, category)
        
        # Visualize seller clusters
        visualizer.visualize_seller_clusters(seller_clusters)
        
        # Visualize recommendations
        visualizer.visualize_reorder_recommendations(recommendations['inventory_recommendations'])
        
        # Create dashboard (if we have all required data)
        if 'inventory_recommendations' in recommendations:
            visualizer.create_supply_chain_dashboard(
                demand_pandas[demand_pandas['product_category_name'].isin(top_categories_list)],
                seller_clusters,
                forecasts,
                recommendations['inventory_recommendations']
            )
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
        # Return results (mainly for testing)
        return {
            'demand_by_category': demand_pandas,
            'top_categories': top_categories_list,
            'forecasts': forecasts,
            'seller_performance': seller_clusters,
            'recommendations': recommendations,
            'metrics': metrics
        }
        
    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session stopped")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure warnings
    warnings.filterwarnings("ignore")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Run appropriate analysis based on arguments
    if args.use_spark:
        results = run_spark_analysis(args)
    else:
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
    for i, category in enumerate(results['top_categories'][:5], 1):
        report.append(f"{i}. {category}")
    
    report.append("")
    
    # Add forecast summaries
    report.append("#### Demand Forecast Highlights")
    for category, forecast in list(results['forecasts'].items())[:3]:  # Top 3 forecasts
        avg_forecast = forecast['forecast'].mean()
        report.append(f"- {category}: Avg. forecasted demand {avg_forecast:.1f} units/month")
    
    report.append("")
    
    # Add reorder recommendations summary
    report.append("#### Inventory Recommendations")
    if isinstance(results['recommendations'], dict) and 'inventory_recommendations' in results['recommendations']:
        # For Spark analysis
        recos = results['recommendations']['inventory_recommendations']
        for _, row in recos.head(3).iterrows():
            report.append(f"- {row['category']}: {row['recommendation']} (Priority: {row['priority']})")
    else:
        # For pandas analysis
        recos = results['recommendations']
        for _, row in recos.head(3).iterrows():
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
    
    report.append("### Visualization Files")
    
    # List output files
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png') or f.endswith('.csv')]
    for f in output_files[:10]:  # Limit to first 10 files
        report.append(f"- {f}")
    
    # Write report to file
    with open(os.path.join(args.output_dir, 'summary_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {os.path.join(args.output_dir, 'summary_report.md')}")

if __name__ == "__main__":
    main()