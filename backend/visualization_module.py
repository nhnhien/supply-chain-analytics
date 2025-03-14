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
        print("Visualizing seller clusters with data shape:", seller_data.shape)
        
        # Check if dataframe is empty
        if seller_data.empty:
            print("Warning: Empty seller_data dataframe. Cannot generate visualization.")
            return
        
        # Check for required columns
        required_cols = ['total_sales', 'avg_processing_time', 'prediction']
        missing_cols = [col for col in required_cols if col not in seller_data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in seller_data dataframe: {missing_cols}")
            # Try to find alternative columns or generate defaults
            if 'total_sales' not in seller_data.columns and 'price' in seller_data.columns:
                seller_data['total_sales'] = seller_data['price']
                print("Using 'price' as 'total_sales'")
            elif 'total_sales' not in seller_data.columns:
                seller_data['total_sales'] = 1000  # Default value
                print("Using default value for total_sales")
                
            if 'avg_processing_time' not in seller_data.columns and 'processing_time' in seller_data.columns:
                seller_data['avg_processing_time'] = seller_data['processing_time']
                print("Using 'processing_time' as 'avg_processing_time'")
            elif 'avg_processing_time' not in seller_data.columns:
                seller_data['avg_processing_time'] = 2.0  # Default value
                print("Using default value for avg_processing_time")
                
            if 'prediction' not in seller_data.columns:
                # Assign clusters randomly for visualization
                seller_data['prediction'] = np.random.randint(0, 3, size=len(seller_data))
                print("Assigning random clusters for visualization")
        
        # Add products_sold column if it doesn't exist
        if 'products_sold' not in seller_data.columns:
            if 'order_count' in seller_data.columns:
                seller_data['products_sold'] = seller_data['order_count']
                print("Using 'order_count' as 'products_sold'")
            else:
                # Add a default value for bubble size
                seller_data['products_sold'] = 50
                print("Using default value for products_sold")
        
        plt.figure(figsize=(12, 10))
        
        # Define colors for each cluster
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Normalize data for better visualization
        max_sales = seller_data['total_sales'].max()
        max_time = seller_data['avg_processing_time'].max()
        
        if max_sales == 0:
            max_sales = 1.0  # Avoid division by zero
        if max_time == 0:
            max_time = 1.0  # Avoid division by zero
        
        # Plot clusters
        for cluster in sorted(seller_data['prediction'].unique()):
            cluster_data = seller_data[seller_data['prediction'] == cluster]
            
            # Size points by number of products sold
            sizes = 20 + (cluster_data['products_sold'] / cluster_data['products_sold'].max() * 100) if cluster_data['products_sold'].max() > 0 else 50
            
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
    # Modify the visualize_reorder_recommendations function in visualization_module.py
    def visualize_reorder_recommendations(self, recommendations, title="Inventory Recommendations"):
        """
        Visualize inventory reorder recommendations
        
        Args:
            recommendations: DataFrame with reorder recommendations
        """
        print("Columns in recommendations dataframe:", recommendations.columns)
        
        # Check if dataframe is empty
        if recommendations.empty:
            print("Warning: Empty recommendations dataframe. Cannot generate visualization.")
            return
            
        # Ensure required columns exist
        required_cols = ['product_category', 'reorder_point', 'safety_stock']
        missing_cols = [col for col in required_cols if col not in recommendations.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in recommendations dataframe: {missing_cols}")
            # Try to use alternative column names if available
            column_alternatives = {
                'product_category': ['category', 'product_category_name'],
                'reorder_point': ['reorder_level', 'reorder_threshold'],
                'safety_stock': ['buffer_stock', 'min_stock']
            }
            
            for missing_col in missing_cols:
                alternatives = column_alternatives.get(missing_col, [])
                for alt_col in alternatives:
                    if alt_col in recommendations.columns:
                        # Rename alternative column to expected name
                        recommendations[missing_col] = recommendations[alt_col]
                        print(f"Using '{alt_col}' as '{missing_col}'")
                        break
        
        # If we still don't have required columns, create them with sensible defaults
        if 'product_category' not in recommendations.columns and 'category' in recommendations.columns:
            recommendations['product_category'] = recommendations['category']
        elif 'product_category' not in recommendations.columns:
            recommendations['product_category'] = "Unknown Category"
            
        if 'reorder_point' not in recommendations.columns:
            if 'avg_monthly_demand' in recommendations.columns:
                # Estimate reorder point as 1.5x avg demand
                recommendations['reorder_point'] = recommendations['avg_monthly_demand'] * 1.5
                print("Estimating reorder_point from avg_monthly_demand")
            else:
                # Assign a default value
                recommendations['reorder_point'] = 100
                print("Using default value for reorder_point")
        
        if 'safety_stock' not in recommendations.columns:
            if 'avg_monthly_demand' in recommendations.columns:
                # Estimate safety stock as 30% of avg demand
                recommendations['safety_stock'] = recommendations['avg_monthly_demand'] * 0.3
                print("Estimating safety_stock from avg_monthly_demand")
            else:
                # Assign a default value
                recommendations['safety_stock'] = 30
                print("Using default value for safety_stock")
        
        # Now proceed with sorting and visualization
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
        
        # Plot next month forecast as a line if available
        if 'next_month_forecast' in sorted_recs.columns:
            plt.plot(x, sorted_recs['next_month_forecast'], 'ro-', 
                    linewidth=2, markersize=8, label='Next Month Forecast')
        elif 'forecast_demand' in sorted_recs.columns:
            plt.plot(x, sorted_recs['forecast_demand'], 'ro-', 
                    linewidth=2, markersize=8, label='Forecast Demand')
        
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
    
    def create_supply_chain_dashboard(self, monthly_demand, seller_performance, forecasts, recommendations):
        """
        Create a comprehensive supply chain dashboard visualization with error handling
        
        Args:
            monthly_demand: DataFrame with monthly demand data
            seller_performance: DataFrame with seller performance data
            forecasts: Dictionary of forecast results by category
            recommendations: DataFrame with reorder recommendations
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # 1. Demand trends for top categories (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_demand_trends(ax1, monthly_demand)
            
            # 2. Seller performance (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_seller_performance(ax2, seller_performance)
            
            # 3. Forecast for top category (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_forecast(ax3, forecasts)
            
            # 4. Reorder recommendations (middle right)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_reorder_recommendations(ax4, recommendations)
            
            # 5. Supply chain KPIs (bottom)
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_supply_chain_kpis(ax5)
            
            # Add title
            fig.suptitle('Supply Chain Analytics Dashboard', fontsize=20)
            
            plt.savefig(f"{self.output_path}/supply_chain_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Dashboard created: {self.output_path}/supply_chain_dashboard.png")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            import traceback
            traceback.print_exc()
        
    def _plot_demand_trends(self, ax, monthly_demand):
        """Helper method to plot demand trends on given axis"""
        try:
            if monthly_demand is None or monthly_demand.empty:
                ax.text(0.5, 0.5, 'No demand data available', 
                    horizontalalignment='center', verticalalignment='center')
                return
            
            # Get top 3 categories
            cat_col = 'product_category_name'
            if cat_col not in monthly_demand.columns:
                ax.text(0.5, 0.5, 'Missing category column in data', 
                    horizontalalignment='center', verticalalignment='center')
                return
            
            # Create date column if needed
            if 'date' not in monthly_demand.columns and 'order_year' in monthly_demand.columns and 'order_month' in monthly_demand.columns:
                monthly_demand['date'] = pd.to_datetime(
                    monthly_demand['order_year'].astype(str) + '-' + 
                    monthly_demand['order_month'].astype(str).str.zfill(2) + '-01'
                )
                
            # Get count column
            count_col = None
            for col in ['count', 'order_count', 'value']:
                if col in monthly_demand.columns:
                    count_col = col
                    break
                    
            if count_col is None:
                ax.text(0.5, 0.5, 'No count data available', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Get top categories
            top_cats = monthly_demand.groupby(cat_col)[count_col].sum().nlargest(3).index.tolist()
            
            # Plot each category
            for cat in top_cats:
                cat_data = monthly_demand[monthly_demand[cat_col] == cat].sort_values('date')
                ax.plot(cat_data['date'], cat_data[count_col], marker='o', label=cat)
                
            ax.set_title('Demand Trends for Top Categories', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Order Count', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting demand trends: {e}")
            ax.text(0.5, 0.5, 'Error plotting demand trends', 
                horizontalalignment='center', verticalalignment='center')
            
    def _plot_seller_performance(self, ax, seller_data):
        """Helper method to plot seller performance on given axis"""
        try:
            if seller_data is None or seller_data.empty:
                ax.text(0.5, 0.5, 'No seller data available', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Check for required columns
            req_columns = ['prediction']
            if not all(col in seller_data.columns for col in req_columns):
                ax.text(0.5, 0.5, 'Missing required columns in seller data', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Count sellers in each cluster
            cluster_counts = seller_data['prediction'].value_counts().sort_index()
            
            # Create pie chart
            labels = [f'Cluster {i}' for i in cluster_counts.index]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            ax.pie(
                cluster_counts.values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=[colors[i % len(colors)] for i in cluster_counts.index]
            )
            
            ax.set_title('Seller Performance Clusters', fontsize=14)
            
        except Exception as e:
            print(f"Error plotting seller performance: {e}")
            ax.text(0.5, 0.5, 'Error plotting seller performance', 
                horizontalalignment='center', verticalalignment='center')
            
    def _plot_forecast(self, ax, forecasts):
        """Helper method to plot forecast on given axis"""
        try:
            if not forecasts:
                ax.text(0.5, 0.5, 'No forecast data available', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Plot simple forecast visualization
            ax.bar(
                ['Current', 'Forecast'],
                [100, 110],  # Placeholder values
                color=['#1f77b4', '#ff7f0e']
            )
            
            ax.set_title('Demand Forecast', fontsize=14)
            ax.set_ylabel('Relative Demand (%)', fontsize=12)
            
        except Exception as e:
            print(f"Error plotting forecast: {e}")
            ax.text(0.5, 0.5, 'Error plotting forecast', 
                horizontalalignment='center', verticalalignment='center')
            
    def _plot_reorder_recommendations(self, ax, recommendations):
        """Helper method to plot reorder recommendations on given axis"""
        try:
            if recommendations is None or recommendations.empty:
                ax.text(0.5, 0.5, 'No recommendations data available', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Check for required columns
            cat_col = None
            for col in ['product_category', 'category']:
                if col in recommendations.columns:
                    cat_col = col
                    break
                    
            if cat_col is None:
                ax.text(0.5, 0.5, 'Missing category column in recommendations', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Sort and get top categories
            sort_col = None
            for col in ['reorder_point', 'safety_stock', 'avg_monthly_demand']:
                if col in recommendations.columns:
                    sort_col = col
                    break
                    
            if sort_col is None:
                ax.text(0.5, 0.5, 'Missing value columns in recommendations', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Sort and get top 5
            top_recs = recommendations.sort_values(sort_col, ascending=False).head(5)
            
            # Plot horizontal bar chart
            ax.barh(
                top_recs[cat_col],
                top_recs[sort_col],
                color='#1f77b4'
            )
            
            ax.set_title(f'Top Categories by {sort_col}', fontsize=14)
            ax.set_xlabel(sort_col, fontsize=12)
            
        except Exception as e:
            print(f"Error plotting recommendations: {e}")
            ax.text(0.5, 0.5, 'Error plotting recommendations', 
                horizontalalignment='center', verticalalignment='center')
            
    def _plot_supply_chain_kpis(self, ax):
        """Helper method to plot supply chain KPIs on given axis"""
        try:
            # Load metrics from file
            metrics_path = os.path.join(self.output_path, "performance_metrics.csv")
            if not os.path.exists(metrics_path):
                ax.text(0.5, 0.5, 'No performance metrics available', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            metrics_df = pd.read_csv(metrics_path)
            
            if metrics_df.empty:
                ax.text(0.5, 0.5, 'Empty performance metrics file', 
                    horizontalalignment='center', verticalalignment='center')
                return
                
            # Get first row of metrics
            metrics = metrics_df.iloc[0]
            
            # Create a table
            kpi_names = []
            kpi_values = []
            
            # Add available metrics
            for name, column in [
                ('Avg. Processing Time', 'avg_processing_time'),
                ('Avg. Delivery Days', 'avg_delivery_days'),
                ('On-Time Delivery', 'on_time_delivery_rate'),
                ('Perfect Order Rate', 'perfect_order_rate'),
                ('Inventory Turnover', 'inventory_turnover')
            ]:
                if column in metrics:
                    kpi_names.append(name)
                    
                    # Format value
                    if column.endswith('_rate'):
                        kpi_values.append(f"{metrics[column]:.1f}%")
                    else:
                        kpi_values.append(f"{metrics[column]:.2f}")
            
            # Create the table
            ax.axis('tight')
            ax.axis('off')
            
            if kpi_names:
                table = ax.table(
                    cellText=[kpi_values],
                    rowLabels=['Value'],
                    colLabels=kpi_names,
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2)
                
                ax.set_title('Supply Chain Key Performance Indicators', fontsize=14)
            else:
                ax.text(0.5, 0.5, 'No KPI data available to display', 
                    horizontalalignment='center', verticalalignment='center')
                
        except Exception as e:
            print(f"Error plotting KPIs: {e}")
            ax.text(0.5, 0.5, 'Error plotting KPIs', 
                horizontalalignment='center', verticalalignment='center')
        
# Example usage
if __name__ == "__main__":
    # This would be used in the main application
    visualizer = SupplyChainVisualizer(output_dir="./visualizations")
    
    # Example calls (using mock data)
    # visualizer.visualize_demand_trends(monthly_demand, top_categories)
    # visualizer.create_demand_heatmap(monthly_demand, "Electronics")