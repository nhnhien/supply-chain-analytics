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

# Import our modules (ensure these modules are available in your backend folder)
from arima_forecasting import DemandForecaster
from visualization_module import SupplyChainVisualizer
from data_preprocessing import (
    preprocess_order_data,
    preprocess_product_data,
    preprocess_order_items,
    calculate_performance_metrics,
    calculate_delivery_days  
)

try:
    from mongodb_storage import MongoDBStorage
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("MongoDB integration not available. Install pymongo to enable MongoDB storage.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Supply Chain Analytics for Demand Forecasting')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    parser.add_argument('--use-auto-arima', action='store_true', help='Use auto ARIMA parameter selection')
    parser.add_argument('--seasonal', action='store_true', help='Include seasonality in the model')
    parser.add_argument('--supplier-clusters', type=int, default=3, help='Number of supplier clusters to create')
    parser.add_argument('--use-mongodb', action='store_true', help='Store results in MongoDB')
    parser.add_argument('--mongodb-uri', type=str, default='mongodb://localhost:27017/', help='MongoDB connection URI')
    parser.add_argument('--mongodb-db', type=str, default='supply_chain_analytics', help='MongoDB database name')
    return parser.parse_args()


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_top_categories(data, limit=15):
    category_totals = {}
    for _, row in data.iterrows():
        category = row.get('product_category_name')
        count = float(row.get('count', 0) or row.get('order_count', 0) or 0)
        if category:
            category_totals[category] = category_totals.get(category, 0) + count
    top = sorted(category_totals.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [cat for cat, _ in top]


def run_pandas_analysis(args):
    print("Running analysis with pandas...")
    
    print("Loading data...")
    customers = pd.read_csv(os.path.join(args.data_dir, 'df_Customers.csv'))
    order_items = pd.read_csv(os.path.join(args.data_dir, 'df_OrderItems.csv'))
    orders = pd.read_csv(os.path.join(args.data_dir, 'df_Orders.csv'))
    products = pd.read_csv(os.path.join(args.data_dir, 'df_Products.csv'))
    
    ensure_directory(args.output_dir)
    
    print("Preprocessing data...");
    orders, delivery_metrics = preprocess_order_data(orders, args.output_dir)
    products = preprocess_product_data(products, args.output_dir)
    order_items = preprocess_order_items(order_items, args.output_dir)
    performance_metrics = calculate_performance_metrics(orders, order_items, args.output_dir)
    
    orders = calculate_delivery_days(orders)
    
    print("Building unified dataset...");
    supply_chain = orders.merge(order_items, on='order_id', how='inner')
    supply_chain = supply_chain.merge(products, on='product_id', how='left')
    supply_chain = supply_chain.merge(customers, on='customer_id', how='left')
    
    orders = calculate_delivery_days(orders, supply_chain)
    supply_chain = orders.merge(order_items, on='order_id', how='inner')
    supply_chain = supply_chain.merge(products, on='product_id', how='left')
    supply_chain = supply_chain.merge(customers, on='customer_id', how='left')
    
    print("Analyzing monthly demand patterns...");
    monthly_demand = (supply_chain.groupby(['product_category_name', 'order_year', 'order_month'])
                    .size().reset_index(name='count'))
    
    monthly_demand['order_year'] = pd.to_numeric(monthly_demand['order_year'], errors='coerce').fillna(1970).astype(int)
    monthly_demand['order_month'] = pd.to_numeric(monthly_demand['order_month'], errors='coerce').fillna(1).astype(int)
    
    monthly_demand['date'] = pd.to_datetime(
        monthly_demand['order_year'].astype(str) + '-' + 
        monthly_demand['order_month'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    monthly_demand['date'] = monthly_demand['date'].fillna(pd.Timestamp('1970-01-01'))
    monthly_demand.to_csv(os.path.join(args.output_dir, 'monthly_demand.csv'), index=False)
    
    top_categories = get_top_categories(monthly_demand, args.top_n)
    print(f"Top {len(top_categories)} categories by volume: {', '.join(top_categories)}")
    
    pd.DataFrame({'product_category_name': top_categories}).to_csv(
        os.path.join(args.output_dir, 'top_categories.csv'), index=False
    )
    

    print("Running time series forecasting...");
    forecaster = DemandForecaster(os.path.join(args.output_dir, 'monthly_demand.csv'))
    forecaster.preprocess_forecast_data()
    if args.use_auto_arima:
        print("Using auto ARIMA for parameter selection")
        forecaster.use_auto_arima = True
    if args.seasonal:
        print("Including seasonality in the model")
        forecaster.seasonal = True
        forecaster.seasonal_period = 12
        
    # Run forecasting with progress tracking
    print("Starting forecasts for all categories...")
    forecasts = forecaster.run_all_forecasts(top_n=args.top_n, periods=args.forecast_periods)
    forecast_report = forecaster.generate_forecast_report(os.path.join(args.output_dir, 'forecast_report.csv'))
    
    # Generate only essential visualizations for the dashboard
    print("Generating essential forecast visualizations...")
    visualizer = SupplyChainVisualizer(output_dir=args.output_dir)

    # Skip individual category visualizations since they create too many files
    # and focus only on aggregate visualizations needed for the dashboard

    # Create comparative visualization of forecast accuracy
    try:
        # Extract forecast performance metrics
        mape_data = []
        for _, row in forecast_report.iterrows():
            if pd.notna(row.get('mape')):
                mape_data.append({
                    'category': row['category'],
                    'mape': min(float(row['mape']), 100),  # Cap at 100% for visualization
                    'color': '#4CAF50' if float(row['mape']) < 15 else 
                            '#FF9800' if float(row['mape']) < 30 else '#F44336'
                })
        
        if mape_data:
            # Sort by MAPE (increasing)
            mape_df = pd.DataFrame(mape_data).sort_values('mape')
            
            # Plot model performance
            plt.figure(figsize=(14, 8))
            bars = plt.bar(
                mape_df['category'], 
                mape_df['mape'],
                color=mape_df['color']
            )
            plt.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            plt.axhline(y=50, color='red', linestyle='--', alpha=0.7)
            plt.title('Forecast Accuracy by Category (MAPE)', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'forecast_accuracy.png'), dpi=300)
            plt.close()
            
            print("Created forecast accuracy visualization")
    except Exception as e:
        print(f"Warning: Could not create forecast accuracy visualization: {e}")

    # Create comparative visualization of forecasted growth rates
    try:
        growth_data = []
        for _, row in forecast_report.iterrows():
            if pd.notna(row.get('growth_rate')):
                growth_rate = float(row['growth_rate'])
                # Apply reasonable bounds for visualization
                growth_rate = max(min(growth_rate, 60), -50)
                growth_data.append({
                    'category': row['category'],
                    'growth_rate': growth_rate,
                    'color': '#4CAF50' if growth_rate > 5 else 
                            '#FF9800' if growth_rate > -5 else '#F44336'
                })
        
        if growth_data:
            # Sort by growth rate (decreasing)
            growth_df = pd.DataFrame(growth_data).sort_values('growth_rate', ascending=False)
            
            # Plot growth rates
            plt.figure(figsize=(14, 8))
            bars = plt.bar(
                growth_df['category'], 
                growth_df['growth_rate'],
                color=growth_df['color']
            )
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Forecasted Growth Rate by Category', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Growth Rate (%)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'forecast_growth_rates.png'), dpi=300)
            plt.close()
            
            print("Created growth rate visualization")
    except Exception as e:
        print(f"Warning: Could not create growth rate visualization: {e}")
        
    # Create overall demand trends visualization
    try:
        # Aggregate monthly demand across top categories
        agg_demand = monthly_demand.copy()
        agg_demand['year_month'] = pd.to_datetime(
            agg_demand['order_year'].astype(str) + '-' + 
            agg_demand['order_month'].astype(str).str.zfill(2) + '-01'
        )
        
        top_cats = monthly_demand['product_category_name'].value_counts().nlargest(5).index.tolist()
        
        plt.figure(figsize=(14, 8))
        
        for category in top_cats:
            cat_data = agg_demand[agg_demand['product_category_name'] == category]
            if len(cat_data) > 0:
                cat_data = cat_data.sort_values('year_month')
                plt.plot(
                    cat_data['year_month'], 
                    cat_data['count'],
                    marker='o',
                    linewidth=2,
                    label=category
                )
        
        plt.title('Monthly Demand Trends for Top Categories', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Order Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'demand_trends.png'), dpi=300)
        plt.close()
        
        print("Created demand trends visualization")
    except Exception as e:
        print(f"Warning: Could not create demand trends visualization: {e}")
    
    print("Analyzing seller performance...");
    seller_performance = (supply_chain.groupby('seller_id')
                          .agg({'order_id': 'count',
                                'processing_time': 'mean',
                                'delivery_days': 'mean',
                                'price': 'sum',
                                'shipping_charges': 'sum'})
                          .reset_index())
    seller_performance.columns = ['seller_id', 'order_count', 'avg_processing_time',
                                   'avg_delivery_days', 'total_sales', 'shipping_costs']
    seller_performance['avg_order_value'] = seller_performance['total_sales'] / seller_performance['order_count']
    seller_performance['shipping_ratio'] = seller_performance['shipping_costs'] / seller_performance['total_sales'] * 100
    
    seller_on_time = supply_chain.groupby('seller_id')['on_time_delivery'].mean().reset_index()
    seller_on_time.columns = ['seller_id', 'on_time_delivery_rate']
    seller_performance = seller_performance.merge(seller_on_time, on='seller_id', how='left')
    seller_performance['on_time_delivery_rate'] = seller_performance['on_time_delivery_rate'].fillna(0.5) * 100
    
    seller_performance['seller_score'] = (
        (seller_performance['order_count'] / seller_performance['order_count'].max()) * 25 +
        (seller_performance['avg_order_value'] / seller_performance['avg_order_value'].max()) * 25 +
        (1 - (seller_performance['avg_processing_time'] / seller_performance['avg_processing_time'].max())) * 25 +
        (seller_performance['on_time_delivery_rate'] / 100) * 25
    )
    
    seller_performance = seller_performance.copy()
    seller_performance['prediction'] = pd.qcut(seller_performance['seller_score'], q=3, labels=[0, 1, 2]).astype(int)
    
    seller_clusters_file = os.path.join(args.output_dir, 'seller_clusters.csv')
    seller_performance.to_csv(seller_clusters_file, index=False)
    print(f"Seller clusters saved to {seller_clusters_file}");
    
    print("Analyzing geographical patterns...");
    state_metrics = (supply_chain.groupby('customer_state')
                     .agg({'order_id': 'count',
                           'processing_time': 'mean',
                           'delivery_days': 'mean',
                           'price': 'sum',
                           'on_time_delivery': 'mean'})
                     .reset_index())
    state_metrics.columns = ['customer_state', 'order_count', 'avg_processing_time',
                             'avg_delivery_days', 'total_sales', 'on_time_delivery_rate']
    state_metrics['on_time_delivery_rate'] = state_metrics['on_time_delivery_rate'] * 100
    state_metrics.to_csv(os.path.join(args.output_dir, 'state_metrics.csv'), index=False)
    
    top_category_by_state = []
    for state in state_metrics['customer_state'].unique():
        state_data = supply_chain[supply_chain['customer_state'] == state]
        if len(state_data) > 0:
            top_cat = (state_data.groupby('product_category_name')['order_id']
                       .count().reset_index())
            if not top_cat.empty:
                top_cat = top_cat.sort_values('order_id', ascending=False).iloc[0]
                top_category_by_state.append({
                    'customer_state': state,
                    'product_category_name': top_cat['product_category_name'],
                    'order_count': top_cat['order_id']
                })
    if top_category_by_state:
        pd.DataFrame(top_category_by_state).to_csv(
            os.path.join(args.output_dir, 'top_category_by_state.csv'), index=False)
    
    print("Generating supply chain recommendations...")
    recommendations = []
    for category in top_categories:
        if category in forecasts:
            cat_demand = monthly_demand[monthly_demand['product_category_name'] == category]['count'].mean()
            cat_std = monthly_demand[monthly_demand['product_category_name'] == category]['count'].std()
            cv = cat_std / cat_demand if cat_demand > 0 else 0.5
            z_score = 1.645
            category_delivery = supply_chain[supply_chain['product_category_name'] == category]['delivery_days'].mean()
            lead_time_days = category_delivery if not pd.isna(category_delivery) else 7
            lead_time_fraction = lead_time_days / 30.0
            safety_stock = z_score * cat_std * np.sqrt(lead_time_fraction)
            safety_stock = max(safety_stock, 0.3 * cat_demand)
            reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            next_month_forecast = forecasts[category]['forecast'].iloc[0] if len(forecasts[category]) > 0 else cat_demand
            
            # Calculate growth rate with reasonable bounds (-50% to +150%)
            if cat_demand > 0:
                growth_rate = ((next_month_forecast - cat_demand) / cat_demand * 100)
                # Apply bounds to growth rate
                growth_rate = max(min(growth_rate, 150), -50)
            else:
                growth_rate = 0
                
            # If forecast seems unreasonable based on historical data, adjust it
            if cat_demand > 0 and len(forecasts[category]) > 0:
                # Check if forecast is outside reasonable bounds
                if next_month_forecast > cat_demand * 2.5 or next_month_forecast < cat_demand * 0.5:
                    print(f"Warning: Extreme forecast detected for {category}. Adjusting to more reasonable value.")
                    # Cap forecast to reasonable bounds
                    bounded_forecast = min(max(next_month_forecast, cat_demand * 0.5), cat_demand * 2.5)
                    # Adjust growth rate accordingly
                    bounded_growth_rate = ((bounded_forecast - cat_demand) / cat_demand * 100)
                    bounded_growth_rate = max(min(bounded_growth_rate, 150), -50)
                    
                    # Update values with bounded versions
                    next_month_forecast = bounded_forecast
                    growth_rate = bounded_growth_rate
            
            # Adjust safety stock and reorder point based on growth trend
            if growth_rate > 20:
                # For high growth, increase safety stock
                safety_stock_adj = safety_stock * (1 + (growth_rate / 100) * 0.5)
                safety_stock = min(safety_stock_adj, safety_stock * 2)  # Cap at double the original
                reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            elif growth_rate < -20:
                # For significant decline, reduce safety stock but ensure minimum level
                safety_stock_adj = safety_stock * (1 + (growth_rate / 100) * 0.3)
                safety_stock = max(safety_stock_adj, safety_stock * 0.6)  # Don't go below 60% of original
                reorder_point = (cat_demand * lead_time_fraction) + safety_stock
            
            # Calculate Economic Order Quantity (EOQ)
            annual_demand = cat_demand * 12
            order_cost = 50
            holding_cost_pct = 0.2
            avg_item_cost = supply_chain[supply_chain['product_category_name'] == category]['price'].mean()
            if pd.isna(avg_item_cost) or avg_item_cost <= 0:
                avg_item_cost = 50
            holding_cost = avg_item_cost * holding_cost_pct
            
            # Adjust annual demand based on growth trend for EOQ calculation
            adjusted_annual_demand = annual_demand * (1 + (growth_rate / 100))
            adjusted_annual_demand = max(adjusted_annual_demand, annual_demand * 0.5)  # Ensure positive demand
            
            eoq = np.sqrt((2 * adjusted_annual_demand * order_cost) / holding_cost)
            order_frequency = adjusted_annual_demand / eoq if eoq > 0 else 12  # Default to monthly if EOQ calculation fails
            days_between_orders = 365 / order_frequency if order_frequency > 0 else 30  # Default to monthly
            
            # Add recommendation with adjustments for extreme values
            recommendations.append({
                'product_category': category,
                'category': category,
                'avg_monthly_demand': cat_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': next_month_forecast,
                'forecast_demand': next_month_forecast,
                'growth_rate': growth_rate,  # Already bounded between -50% and +150%
                'lead_time_days': lead_time_days,
                'days_between_orders': min(days_between_orders, 90),  # Cap at 90 days maximum
                'avg_item_cost': avg_item_cost,
                'adjusted_for_growth': abs(growth_rate) > 20  # Flag to indicate growth adjustment
            })
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(os.path.join(args.output_dir, 'reorder_recommendations.csv'), index=False)
    
    print("Creating visualizations...");
    visualizer.visualize_demand_trends(monthly_demand, top_categories[:10])
    for category in top_categories[:5]:
        visualizer.create_demand_heatmap(monthly_demand, category)
    visualizer.visualize_seller_clusters(seller_performance)
    visualizer.visualize_reorder_recommendations(recommendations_df)
    
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
        'state_metrics': state_metrics,
        'performance_metrics': performance_metrics
    }


def store_results_in_mongodb(results, args):
    try:
        from mongodb_storage import MongoDBStorage
        import os
        connection_string = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
        storage = MongoDBStorage(connection_string)
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        metadata = {
            'run_id': run_id,
            'timestamp': datetime.now(),
            'parameters': {
                'data_dir': args.data_dir,
                'output_dir': args.output_dir,
                'top_n': args.top_n,
                'forecast_periods': args.forecast_periods,
                'use_auto_arima': args.use_auto_arima,
                'seasonal': args.seasonal,
                'supplier_clusters': args.supplier_clusters
            },
            'execution_environment': {
                'platform': os.sys.platform,
                'python_version': os.sys.version
            }
        }
        storage.store_analysis_metadata(metadata)
        if 'monthly_demand' in results:
            storage.store_monthly_demand(results['monthly_demand'], run_id)
        if 'forecasts' in results:
            import pandas as pd
            forecasts_list = []
            for category, forecast_df in results['forecasts'].items():
                forecast_record = {
                    'category': category,
                    'forecast_values': forecast_df['forecast'].tolist() if hasattr(forecast_df, 'tolist') else forecast_df['forecast'].values.tolist()
                }
                if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
                    forecast_record['lower_ci'] = forecast_df['lower_ci'].tolist() if hasattr(forecast_df['lower_ci'], 'tolist') else forecast_df['lower_ci'].values.tolist()
                    forecast_record['upper_ci'] = forecast_df['upper_ci'].tolist() if hasattr(forecast_df['upper_ci'], 'tolist') else forecast_df['upper_ci'].values.tolist()
                if category in results.get('growth_rates', {}):
                    forecast_record['growth_rate'] = results['growth_rates'][category]
                forecasts_list.append(forecast_record)
            forecasts_df = pd.DataFrame(forecasts_list)
            storage.store_forecasts(forecasts_df, run_id)
        if 'seller_performance' in results:
            storage.store_supplier_clusters(results['seller_performance'], run_id)
        if 'recommendations' in results:
            storage.store_inventory_recommendations(results['recommendations'], run_id)
        storage.close()
        print(f"Analysis results stored in MongoDB with run_id: {run_id}")
        return run_id
    except ImportError:
        print("Warning: MongoDB integration not available. Install pymongo to enable MongoDB storage.")
        return None
    except Exception as e:
        print(f"Error storing results in MongoDB: {e}")
        return None


def main():
    args = parse_arguments()
    warnings.filterwarnings("ignore")
    ensure_directory(args.output_dir)
    results = run_pandas_analysis(args)

    if args.use_mongodb and MONGODB_AVAILABLE:
        run_id = store_results_in_mongodb(results, args)
        if run_id:
            with open(os.path.join(args.output_dir, 'summary_report.md'), 'a') as f:
                f.write(f"\n\n## MongoDB Storage\n\nResults stored in MongoDB with run_id: {run_id}\n")

    print("Generating summary report...");
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
    
    report.append("#### Top Product Categories by Demand")
    for i, category in enumerate(results['top_categories'][:10], 1):
        report.append(f"{i}. {category}")
    report.append("")
    
    report.append("#### Demand Forecast Highlights")
    for category, forecast_df in list(results['forecasts'].items())[:5]:
        avg_forecast = forecast_df['forecast'].mean()
        report.append(f"- {category}: Avg. forecasted demand {avg_forecast:.1f} units/month")
    report.append("")
    
    report.append("#### Inventory Recommendations")
    if isinstance(results['recommendations'], pd.DataFrame):
        recos = results['recommendations']
        for _, row in recos.head(5).iterrows():
            report.append(f"- {row['product_category']}: Reorder at {row['reorder_point']:.1f} units, Safety stock: {row['safety_stock']:.1f} units")
    report.append("")
    
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
    
    report.append("#### Seller Performance Insights")
    seller_clusters = results['seller_performance'].groupby('prediction').size().reset_index()
    seller_clusters.columns = ['cluster', 'count']
    total_sellers = seller_clusters['count'].sum()
    for idx, row in seller_clusters.iterrows():
        cluster_num = row['cluster']
        cluster_size = row['count']
        percentage = (cluster_size / total_sellers) * 100
        if cluster_num == 0:
            cluster_label = 'High Performers'
        elif cluster_num == 1:
            cluster_label = 'Medium Performers'
        else:
            cluster_label = 'Low Performers'
        report.append(f"- {cluster_label}: {cluster_size} sellers ({percentage:.1f}%)")
    report.append("")
    
    report.append("#### Visualization Files")
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png') or f.endswith('.csv') or f.endswith('.md')]
    for f in output_files[:15]:
        report.append(f"- {f}")
    
    with open(os.path.join(args.output_dir, 'summary_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {os.path.join(args.output_dir, 'summary_report.md')}")
    
if __name__ == "__main__":
    main()
