import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Set style for consistent visualizations
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def format_thousands(x, pos):
    """Format axis labels to show thousands with K"""
    return f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'


class SupplyChainVisualizer:
    """
    Class to create visualizations for supply chain analytics.
    """
    def __init__(self, output_dir='.'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        self.output_dir = output_dir
        
    def visualize_demand_trends(self, monthly_demand, top_categories, title="Monthly Demand Trends"):
        """
        Visualize demand trends for top product categories.
        
        Args:
            monthly_demand: DataFrame with monthly demand data.
            top_categories: List of top product categories to visualize.
        """
        plt.figure(figsize=(14, 8))
        
        for category in top_categories:
            # Filter data for this category
            category_data = monthly_demand[monthly_demand['product_category_name'] == category].copy()
            # Create date column
            category_data['date'] = pd.to_datetime(
                category_data['order_year'].astype(str) + '-' + 
                category_data['order_month'].astype(str).str.zfill(2) + '-01'
            )
            # Sort by date
            category_data.sort_values('date', inplace=True)
            # Plot data
            plt.plot(category_data['date'], category_data['count'], marker='o', linestyle='-', label=category)
        
        # Format the plot
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Order Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        # Format x-axis dates
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gcf().autofmt_xdate()
        # Format y-axis with K for thousands
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/demand_trends.png", dpi=300)
        plt.close()
        
    def create_demand_heatmap(self, monthly_demand, category, title=None):
        """
        Create a heatmap of monthly demand patterns for a specific category.
        
        Args:
            monthly_demand: DataFrame with monthly demand data.
            category: Product category to visualize.
        """
        # Filter data for this category
        category_data = monthly_demand[monthly_demand['product_category_name'] == category].copy()
        # Create pivot table
        pivot_data = category_data.pivot_table(
            index='order_year', 
            columns='order_month',
            values='count',
            aggfunc='sum'
        )
        # Fill NaN values with 0
        pivot_data = pivot_data.fillna(0)
        # Create a custom colormap from white to blue
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        cmap = LinearSegmentedColormap.from_list('blue_gradient', colors)
        
        plt.figure(figsize=(14, 6))
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={'label': 'Order Count'}
        )
        
        # Set title and labels
        if title is None:
            title = f'Monthly Demand Patterns for {category}'
        plt.title(title, fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Year', fontsize=14)
        # Add month names instead of numbers
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, month_names, rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/heatmap_{category.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()
        
    def visualize_seller_clusters(self, seller_data, title="Seller Performance Clusters"):
        """
        Visualize seller performance clusters.
        
        Args:
            seller_data: DataFrame with seller clustering results.
        """
        print("Visualizing seller clusters with data shape:", seller_data.shape)
        
        if seller_data is None or seller_data.empty:
            print("Warning: Empty seller_data dataframe. Cannot generate visualization.")
            return
        
        required_cols = ['total_sales', 'avg_processing_time', 'prediction']
        missing_cols = [col for col in required_cols if col not in seller_data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in seller_data dataframe: {missing_cols}")
            if 'total_sales' not in seller_data.columns and 'price' in seller_data.columns:
                seller_data['total_sales'] = seller_data['price']
                print("Using 'price' as 'total_sales'")
            elif 'total_sales' not in seller_data.columns:
                seller_data['total_sales'] = 1000
                print("Using default value for total_sales")
                
            if 'avg_processing_time' not in seller_data.columns and 'processing_time' in seller_data.columns:
                seller_data['avg_processing_time'] = seller_data['processing_time']
                print("Using 'processing_time' as 'avg_processing_time'")
            elif 'avg_processing_time' not in seller_data.columns:
                seller_data['avg_processing_time'] = 2.0
                print("Using default value for avg_processing_time")
                
            if 'prediction' not in seller_data.columns:
                seller_data['prediction'] = np.random.randint(0, 3, size=len(seller_data))
                print("Assigning random clusters for visualization")
        
        if 'products_sold' not in seller_data.columns:
            if 'order_count' in seller_data.columns:
                seller_data['products_sold'] = seller_data['order_count']
                print("Using 'order_count' as 'products_sold'")
            else:
                seller_data['products_sold'] = 50
                print("Using default value for products_sold")
        
        plt.figure(figsize=(12, 10))
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        max_sales = seller_data['total_sales'].max() or 1.0
        max_time = seller_data['avg_processing_time'].max() or 1.0
        
        for cluster in sorted(seller_data['prediction'].unique()):
            cluster_data = seller_data[seller_data['prediction'] == cluster]
            sizes = 20 + (cluster_data['products_sold'] / cluster_data['products_sold'].max() * 100) if cluster_data['products_sold'].max() > 0 else 50
            plt.scatter(
                cluster_data['total_sales'] / max_sales,
                cluster_data['avg_processing_time'] / max_time,
                s=sizes,
                alpha=0.6,
                c=[cluster_colors[int(cluster) % len(cluster_colors)]],
                label=f'Cluster {int(cluster)}'
            )
        
        plt.title(title, fontsize=16)
        plt.xlabel('Normalized Total Sales', fontsize=14)
        plt.ylabel('Normalized Processing Time', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.figtext(0.1, 0.01, "Lower processing time and higher sales indicate better performance", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/seller_clusters.png", dpi=300)
        plt.close()
        
    def visualize_reorder_recommendations(self, recommendations, title="Inventory Recommendations"):
        """
        Visualize inventory reorder recommendations.
        
        Args:
            recommendations: DataFrame with reorder recommendations.
        """
        print("Columns in recommendations dataframe:", recommendations.columns)
        
        if recommendations is None or recommendations.empty:
            print("Warning: Empty recommendations dataframe. Cannot generate visualization.")
            return
            
        # Ensure required columns exist
        cat_col = None
        for col in ['product_category', 'category']:
            if col in recommendations.columns:
                cat_col = col
                break
        if cat_col is None:
            print("Warning: Missing category column in recommendations; defaulting to 'Unknown Category'")
            recommendations['product_category'] = "Unknown Category"
            cat_col = 'product_category'
        sort_col = None
        for col in ['reorder_point', 'safety_stock', 'avg_monthly_demand']:
            if col in recommendations.columns:
                sort_col = col
                break
        if sort_col is None:
            print("Warning: Missing value columns in recommendations, using default reorder_point")
            recommendations['reorder_point'] = 100
            sort_col = 'reorder_point'
        
        if 'reorder_point' not in recommendations.columns:
            if 'avg_monthly_demand' in recommendations.columns:
                recommendations['reorder_point'] = recommendations['avg_monthly_demand'] + (recommendations['avg_monthly_demand'] * 0.3)
                print("Estimating reorder_point from avg_monthly_demand")
            else:
                recommendations['reorder_point'] = 100
                print("Using default value for reorder_point")
        
        if 'safety_stock' not in recommendations.columns:
            if 'avg_monthly_demand' in recommendations.columns:
                recommendations['safety_stock'] = recommendations['avg_monthly_demand'] * 0.3
                print("Estimating safety_stock from avg_monthly_demand")
            else:
                recommendations['safety_stock'] = 30
                print("Using default value for safety_stock")
        
        sorted_recs = recommendations.sort_values(sort_col, ascending=False)
        if sorted_recs.shape[0] > 10:
            sorted_recs = sorted_recs.head(10)
        
        plt.figure(figsize=(14, 8))
        x = np.arange(sorted_recs.shape[0])
        width = 0.35
        plt.bar(x, sorted_recs['safety_stock'], width, label='Safety Stock', color='#1f77b4')
        plt.bar(x, sorted_recs['reorder_point'] - sorted_recs['safety_stock'], width, 
                bottom=sorted_recs['safety_stock'], label='Lead Time Demand', color='#ff7f0e')
        if 'next_month_forecast' in recommendations.columns:
            plt.plot(x, sorted_recs['next_month_forecast'], 'ro-', linewidth=2, markersize=8, label='Next Month Forecast')
        elif 'forecast_demand' in recommendations.columns:
            plt.plot(x, sorted_recs['forecast_demand'], 'ro-', linewidth=2, markersize=8, label='Forecast Demand')
        plt.title(title, fontsize=16)
        plt.xlabel('Product Category', fontsize=14)
        plt.ylabel('Units', fontsize=14)
        plt.xticks(x, sorted_recs[cat_col], rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reorder_recommendations.png", dpi=300)
        plt.close()
        
    def visualize_forecast_accuracy(self, performance_data, title="Forecast Model Performance"):
        """
        Visualize forecast model performance metrics.
        
        Args:
            performance_data: DataFrame with forecast performance metrics.
        """
        if 'mape' not in performance_data.columns:
            print("Missing MAPE column for accuracy visualization")
            return
        plt.figure(figsize=(12, 6))
        sorted_perf = performance_data.sort_values('mape')
        plt.bar(sorted_perf['category'], sorted_perf['mape'], color='#2ca02c')
        for i, value in enumerate(sorted_perf['mape']):
            plt.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.title(title, fontsize=16)
        plt.xlabel('Product Category', fontsize=14)
        plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='20% Threshold')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/forecast_accuracy.png", dpi=300)
        plt.close()
        
    def visualize_forecast_comparison(self, forecast_data, historical_data, title="Forecast vs. Actual Demand"):
        """
        Compare forecast with actual values (for validation periods).
        
        Args:
            forecast_data: DataFrame with forecast values.
            historical_data: DataFrame with actual historical values.
        """
        plt.figure(figsize=(14, 8))
        plt.plot(historical_data.index, historical_data.values, label='Actual Demand', color='blue', marker='o', linewidth=2)
        plt.plot(forecast_data.index, forecast_data['forecast'], label='Forecast', color='red', linestyle='--', linewidth=2)
        plt.fill_between(forecast_data.index, forecast_data['lower_ci'], forecast_data['upper_ci'], color='red', alpha=0.2, label='95% Confidence Interval')
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Order Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gcf().autofmt_xdate()
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/forecast_comparison.png", dpi=300)
        plt.close()


# Example usage
if __name__ == "__main__":
    visualizer = SupplyChainVisualizer(output_dir="./visualizations")
    # Example calls (using mock data)
    # visualizer.visualize_demand_trends(monthly_demand, top_categories)
    # visualizer.create_demand_heatmap(monthly_demand, "Electronics")
