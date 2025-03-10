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
    
    # Preprocess orders
    print("Preprocessing orders...")
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
    
    # Extract year and month for time-based analysis
    orders['order_year'] = orders['order_purchase_timestamp'].dt.year
    orders['order_month'] = orders['order_purchase_timestamp'].dt.month
    
    # Calculate processing time
    orders['processing_time'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.days
    
    # Add simulated delivery days (since actual data is missing)
    orders['delivery_days'] = np.random.randint(3, 11, size=len(orders))
    
    # Merge datasets
    print("Building unified dataset...")
    # Join orders with order items
    supply_chain = orders.merge(order_items, on='order_id', how='inner')
    
    # Join with products
    supply_chain = supply_chain.merge(products, on='product_id', how='left')
    
    # Join with customers
    supply_chain = supply_chain.merge(customers, on='customer_id', how='left')
    
    # Analyze monthly demand
    print("Analyzing monthly demand patterns...")
    monthly_demand = supply_chain.groupby(['product_category_name', 'order_year', 'order_month']).size().reset_index(name='count')
    
    # Get top categories
    top_categories = monthly_demand.groupby('product_category_name')['count'].sum().sort_values(ascending=False).head(args.top_n).index.tolist()
    
    # Create output directory for results
    ensure_directory(args.output_dir)
    
    # Run time series forecasting
    print("Running time series forecasting...")
    monthly_demand.to_csv(os.path.join(args.output_dir, 'monthly_demand.csv'), index=False)
    
    forecaster = DemandForecaster(os.path.join(args.output_dir, 'monthly_demand.csv'))
    forecaster.preprocess_data()
    forecasts = forecaster.run_all_forecasts(top_n=args.top_n, periods=args.forecast_periods)
    
    # Generate forecast report
    forecast_report = forecaster.generate_forecast_report(os.path.join(args.output_dir, 'forecast_report.csv'))
    
    # Calculate seller performance
    print("Analyzing seller performance...")
    seller_performance = supply_chain.groupby('seller_id').agg({
        'order_id': 'count',
        'processing_time': 'mean',
        'delivery_days': 'mean',
        'price': 'sum'
    }).reset_index()
    
    seller_performance.columns = ['seller_id', 'order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
    
    # Add simulated performance clusters (since we're not using Spark's ML clustering)
    seller_performance['prediction'] = np.random.randint(0, 3, size=len(seller_performance))
    
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
    # Create safety stock and reorder point recommendations (simplified)
    recommendations = []
    
    for category in top_categories:
        if category in forecasts:
            # Get average monthly demand
            cat_demand = monthly_demand[monthly_demand['product_category_name'] == category]['count'].mean()
            
            # Calculate safety stock (simplified formula)
            safety_stock = cat_demand * 0.5  # 50% of average monthly demand
            
            # Calculate reorder point
            lead_time_days = 7  # Assumption: 7 days lead time
            lead_time_fraction = lead_time_days / 30.0  # As fraction of month
            reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            
            # Get next month forecast
            next_month_forecast = forecasts[category]['forecast'].iloc[0]
            
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
        'state_metrics': state_metrics  # Add state metrics to the results
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
        
        # Process orders
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
            'recommendations': recommendations
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