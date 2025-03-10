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
    Class to create visualizations for supply chain analytics
    """
    def __init__(self, output_dir='.'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
    def visualize_demand_trends(self, monthly_demand, top_categories, title="Monthly Demand Trends"):
        """
        Visualize demand trends for top product categories
        
        Args:
            monthly_demand: DataFrame with monthly demand data
            top_categories: List of top product categories to visualize
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
        Create a heatmap of monthly demand patterns for a specific category
        
        Args:
            monthly_demand: DataFrame with monthly demand data
            category: Product category to visualize
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
        Visualize seller performance clusters
        
        Args:
            seller_data: DataFrame with seller clustering results
        """
        # Check for required columns
        required_cols = ['total_sales', 'avg_processing_time', 'products_sold', 'prediction']
        for col in required_cols:
            if col not in seller_data.columns:
                print(f"Missing required column: {col}")
                return
        
        plt.figure(figsize=(12, 10))
        
        # Define colors for each cluster
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Normalize data for better visualization
        max_sales = seller_data['total_sales'].max()
        max_time = seller_data['avg_processing_time'].max()
        
        # Plot clusters
        for cluster in sorted(seller_data['prediction'].unique()):
            cluster_data = seller_data[seller_data['prediction'] == cluster]
            
            # Size points by number of products sold
            sizes = 20 + (cluster_data['products_sold'] / cluster_data['products_sold'].max() * 100)
            
            plt.scatter(
                cluster_data['total_sales'] / max_sales,  # Normalize
                cluster_data['avg_processing_time'] / max_time,  # Normalize
                s=sizes,
                alpha=0.6,
                c=cluster_colors[int(cluster) % len(cluster_colors)],
                label=f'Cluster {int(cluster)}'
            )
        
        # Add labels and title
        plt.title(title, fontsize=16)
        plt.xlabel('Normalized Total Sales', fontsize=14)
        plt.ylabel('Normalized Processing Time', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add text description of clusters
        plt.figtext(0.01, 0.01, 
                   "Lower processing time and higher sales indicate better performance",
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/seller_clusters.png", dpi=300)
        plt.close()
        
    def visualize_reorder_recommendations(self, recommendations, title="Inventory Recommendations"):
        """
        Visualize inventory reorder recommendations
        
        Args:
            recommendations: DataFrame with reorder recommendations
        """
        # Sort by reorder point (descending)
        sorted_recs = recommendations.sort_values('reorder_point', ascending=False)
        
        # Limit to top 10 for readability
        if len(sorted_recs) > 10:
            sorted_recs = sorted_recs.head(10)
        
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sorted_recs))
        width = 0.35
        
        # Plot safety stock
        plt.bar(x, sorted_recs['safety_stock'], width, label='Safety Stock', color='#1f77b4')
        
        # Plot lead time demand (difference between reorder point and safety stock)
        plt.bar(x, 
               sorted_recs['reorder_point'] - sorted_recs['safety_stock'], 
               width, 
               bottom=sorted_recs['safety_stock'], 
               label='Lead Time Demand', 
               color='#ff7f0e')
        
        # Plot next month forecast as a line
        if 'next_month_forecast' in sorted_recs.columns:
            plt.plot(x, sorted_recs['next_month_forecast'], 'ro-', 
                    linewidth=2, markersize=8, label='Next Month Forecast')
        
        # Add labels and title
        plt.title(title, fontsize=16)
        plt.xlabel('Product Category', fontsize=14)
        plt.ylabel('Units', fontsize=14)
        plt.xticks(x, sorted_recs['product_category'], rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Format y-axis with K for thousands
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reorder_recommendations.png", dpi=300)
        plt.close()
        
    def visualize_forecast_accuracy(self, performance_data, title="Forecast Model Performance"):
        """
        Visualize forecast model performance metrics
        
        Args:
            performance_data: DataFrame with forecast performance metrics
        """
        # Check if we have MAPE (Mean Absolute Percentage Error)
        if 'mape' not in performance_data.columns:
            print("Missing MAPE column for accuracy visualization")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Sort by MAPE (ascending, lower is better)
        sorted_perf = performance_data.sort_values('mape')
        
        # Create bar chart
        plt.bar(sorted_perf['category'], sorted_perf['mape'], color='#2ca02c')
        
        # Add data labels on top of bars
        for i, value in enumerate(sorted_perf['mape']):
            plt.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        plt.title(title, fontsize=16)
        plt.xlabel('Product Category', fontsize=14)
        plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add reference line for good forecast (usually MAPE < 20% is considered good)
        plt.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='20% Threshold')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/forecast_accuracy.png", dpi=300)
        plt.close()
        
    def visualize_forecast_comparison(self, forecast_data, historical_data, title="Forecast vs. Actual Demand"):
        """
        Compare forecast with actual values (for validation periods)
        
        Args:
            forecast_data: DataFrame with forecast values
            historical_data: DataFrame with actual historical values
        """
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data.values, 
                label='Actual Demand', color='blue', marker='o', linewidth=2)
        
        # Plot forecast data
        plt.plot(forecast_data.index, forecast_data['forecast'], 
                label='Forecast', color='red', linestyle='--', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_data.index, 
                        forecast_data['lower_ci'], 
                        forecast_data['upper_ci'], 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Add labels and title
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Order Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gcf().autofmt_xdate()
        
        # Format y-axis with K for thousands
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/forecast_comparison.png", dpi=300)
        plt.close()
    
    def create_supply_chain_dashboard(self, top_category_data, supplier_performance, forecasts, recommendations):
        """
        Create a comprehensive supply chain dashboard with multiple subplots
        
        Args:
            top_category_data: DataFrame with top category demand data
            supplier_performance: DataFrame with supplier performance data
            forecasts: Dictionary of forecast results by category
            recommendations: DataFrame with reorder recommendations
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # 1. Demand trends for top categories (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_demand_trends(ax1, top_category_data)
        
        # 2. Supplier performance (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_supplier_performance(ax2, supplier_performance)
        
        # 3. Forecast for top category (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_forecast(ax3, next(iter(forecasts.values())))  # Plot first forecast
        
        # 4. Reorder recommendations (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_reorder_recommendations(ax4, recommendations)
        
        # 5. Supply chain KPIs (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_supply_chain_kpis(ax5, supplier_performance, forecasts)
        
        # Add title
        fig.suptitle('Supply Chain Analytics Dashboard', fontsize=20)
        
        plt.savefig(f"{self.output_dir}/supply_chain_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_demand_trends(self, ax, demand_data):
        """Helper function to plot demand trends on given axis"""
        categories = demand_data['product_category_name'].unique()
        for category in categories[:3]:  # Top 3 categories
            cat_data = demand_data[demand_data['product_category_name'] == category]
            ax.plot(cat_data['date'], cat_data['count'], marker='o', label=category)
            
        ax.set_title('Demand Trends for Top Categories', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Order Count', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_supplier_performance(self, ax, supplier_data):
        """Helper function to plot supplier performance on given axis"""
        # Sort by performance metric (e.g., delivery time)
        sorted_data = supplier_data.sort_values('avg_processing_time')[:10]  # Top 10
        
        ax.barh(sorted_data['seller_id'], sorted_data['avg_processing_time'], color='#1f77b4')
        ax.set_title('Top 10 Suppliers by Processing Time', fontsize=14)
        ax.set_xlabel('Avg. Processing Time (Days)', fontsize=12)
        ax.set_ylabel('Supplier ID', fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        
    def _plot_forecast(self, ax, forecast_data):
        """Helper function to plot forecast on given axis"""
        # Plot forecast
        ax.plot(forecast_data.index, forecast_data['forecast'], 'r-', label='Forecast')
        
        # Plot confidence intervals
        ax.fill_between(forecast_data.index, 
                      forecast_data['lower_ci'], 
                      forecast_data['upper_ci'], 
                      color='red', alpha=0.2)
        
        ax.set_title('Demand Forecast', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Forecast Demand', fontsize=12)
        ax.grid(True, alpha=0.3)
        
    def _plot_reorder_recommendations(self, ax, recommendations):
        """Helper function to plot reorder recommendations on given axis"""
        # Sort and limit to top 5
        sorted_recs = recommendations.sort_values('reorder_point', ascending=False).head(5)
        
        x = np.arange(len(sorted_recs))
        width = 0.35
        
        # Plot safety stock and lead time demand
        ax.bar(x, sorted_recs['safety_stock'], width, label='Safety Stock')
        ax.bar(x, sorted_recs['reorder_point'] - sorted_recs['safety_stock'], 
              width, bottom=sorted_recs['safety_stock'], label='Lead Time Demand')
        
        ax.set_title('Top 5 Categories by Reorder Point', fontsize=14)
        ax.set_xlabel('Product Category', fontsize=12)
        ax.set_ylabel('Units', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_recs['product_category'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
    def _plot_supply_chain_kpis(self, ax, supplier_data, forecasts):
        """Helper function to plot supply chain KPIs on given axis"""
        # Calculate some KPIs
        avg_processing_time = supplier_data['avg_processing_time'].mean()
        
        # Calculate forecast growth
        growth_rates = []
        for category, forecast in forecasts.items():
            if isinstance(forecast, pd.DataFrame) and 'forecast' in forecast.columns:
                # Calculate growth rate as percentage
                growth = ((forecast['forecast'].mean() - 100) / 100) * 100  # Assuming baseline of 100
                growth_rates.append(growth)
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        
        # Define KPIs to display
        kpis = {
            'Avg Processing Time': f"{avg_processing_time:.2f} days",
            'Forecast Growth': f"{avg_growth:.1f}%",
            'On-Time Delivery': "92.5%",  # Example value
            'Perfect Order Rate': "89.3%",  # Example value
            'Inventory Turnover': "12.4x"  # Example value
        }
        
        # Create a table
        cell_text = [[v] for v in kpis.values()]
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=cell_text, rowLabels=list(kpis.keys()), 
                       colLabels=["Value"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        ax.set_title('Supply Chain Key Performance Indicators', fontsize=14)
        
# Example usage
if __name__ == "__main__":
    # This would be used in the main application
    visualizer = SupplyChainVisualizer(output_dir="./visualizations")
    
    # Example calls (using mock data)
    # visualizer.visualize_demand_trends(monthly_demand, top_categories)
    # visualizer.create_demand_heatmap(monthly_demand, "Electronics")