import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

class SupplierAnalyzer:
    """
    Class for analyzing supplier performance and clustering suppliers
    based on multiple performance metrics
    """
    def __init__(self, data_path="seller_clusters.csv"):
        """
        Initialize the analyzer with data
        
        Args:
            data_path: Path to CSV file with supplier data
        """
        self.data = pd.read_csv(data_path)
        self.features = []
        self.clusters = None
        self.cluster_centers = None
        self.performance_labels = None
        self.n_clusters = 3  # Default number of clusters
        
    def preprocess_data(self):
        """
        Preprocess supplier data for analysis
        """
        print("Preprocessing supplier data...")
        
        # Check for required columns
        required_columns = ['seller_id', 'order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert seller_id to string if it's not already
        self.data['seller_id'] = self.data['seller_id'].astype(str)
        
        # Handle missing values
        for col in self.data.columns:
            if col != 'seller_id' and pd.api.types.is_numeric_dtype(self.data[col]):
                # Fill missing numerical values with median
                col_median = self.data[col].median()
                self.data[col] = self.data[col].fillna(col_median)
        
        # Create additional features for clustering
        
        # Add average order value
        if 'avg_order_value' not in self.data.columns:
            self.data['avg_order_value'] = self.data['total_sales'] / self.data['order_count']
            
            # Handle division by zero
            self.data['avg_order_value'] = self.data['avg_order_value'].replace([np.inf, -np.inf], np.nan)
            self.data['avg_order_value'] = self.data['avg_order_value'].fillna(self.data['avg_order_value'].median())
        
        # Add on-time delivery rate if available
        if 'on_time_delivery_rate' not in self.data.columns and 'on_time_delivery' in self.data.columns:
            self.data['on_time_delivery_rate'] = self.data['on_time_delivery'] * 100
        
        # If shipping costs are available, calculate shipping ratio
        if 'shipping_costs' in self.data.columns:
            self.data['shipping_ratio'] = (self.data['shipping_costs'] / self.data['total_sales']) * 100
            
            # Handle division by zero
            self.data['shipping_ratio'] = self.data['shipping_ratio'].replace([np.inf, -np.inf], np.nan)
            self.data['shipping_ratio'] = self.data['shipping_ratio'].fillna(self.data['shipping_ratio'].median())
        
        # Define features to use for clustering
        self.features = [
            'order_count',
            'avg_processing_time',
            'avg_delivery_days',
            'total_sales',
            'avg_order_value'
        ]
        
        # Add optional features if available
        if 'on_time_delivery_rate' in self.data.columns:
            self.features.append('on_time_delivery_rate')
            
        if 'shipping_ratio' in self.data.columns:
            self.features.append('shipping_ratio')
        
        print(f"Preprocessed data with {len(self.data)} sellers and {len(self.features)} features")
        
    def determine_optimal_clusters(self, max_clusters=10):
        """
        Determine the optimal number of clusters using silhouette score
        
        Args:
            max_clusters: Maximum number of clusters to evaluate
            
        Returns:
            Optimal number of clusters
        """
        if len(self.features) == 0:
            self.preprocess_data()
            
        # Extract features for clustering
        X = self.data[self.features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Evaluate different numbers of clusters
        silhouette_scores = []
        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            try:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
            except Exception as e:
                print(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
                silhouette_scores.append(0)
        
        # Find the best number of clusters
        if silhouette_scores:
            optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because we started from 2
        else:
            optimal_clusters = 3  # Default to 3 clusters if calculation fails
            
        print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    def cluster_suppliers(self, n_clusters=None):
        """
        Cluster suppliers based on performance metrics
        
        Args:
            n_clusters: Number of clusters to create (if None, will determine optimal)
            
        Returns:
            DataFrame with cluster assignments
        """
        if len(self.features) == 0:
            self.preprocess_data()
            
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        
        self.n_clusters = n_clusters
        
        # Extract features for clustering
        X = self.data[self.features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Save cluster centers
        self.cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=self.features
        )
        
        # Calculate cluster performance metrics
        cluster_metrics = self.data.groupby('cluster').agg({
            'order_count': 'mean',
            'avg_processing_time': 'mean',
            'avg_delivery_days': 'mean',
            'total_sales': 'mean',
            'avg_order_value': 'mean'
        })
        
        # Add on-time delivery rate if available
        if 'on_time_delivery_rate' in self.data.columns:
            cluster_on_time = self.data.groupby('cluster')['on_time_delivery_rate'].mean()
            cluster_metrics['on_time_delivery_rate'] = cluster_on_time
        
        # Calculate performance score for each cluster
        # Higher is better
        cluster_metrics['score'] = (
            # Normalized sales (higher is better)
            (cluster_metrics['total_sales'] / cluster_metrics['total_sales'].max()) * 30 +
            # Normalized order count (higher is better)
            (cluster_metrics['order_count'] / cluster_metrics['order_count'].max()) * 20 +
            # Normalized processing time (lower is better)
            (1 - (cluster_metrics['avg_processing_time'] / cluster_metrics['avg_processing_time'].max())) * 25
        )
        
        # Add on-time delivery to score if available
        if 'on_time_delivery_rate' in cluster_metrics.columns:
            cluster_metrics['score'] += (
                # Normalized on-time delivery (higher is better)
                (cluster_metrics['on_time_delivery_rate'] / 100) * 25
            )
        else:
            # Add normalized delivery days if on-time rate isn't available
            cluster_metrics['score'] += (
                # Normalized delivery days (lower is better)
                (1 - (cluster_metrics['avg_delivery_days'] / cluster_metrics['avg_delivery_days'].max())) * 25
            )
        
        # Rank clusters by performance score
        cluster_metrics = cluster_metrics.sort_values('score', ascending=False)
        
        # Assign performance labels (High, Medium, Low)
        self.performance_labels = {}
        for i, (cluster, _) in enumerate(cluster_metrics.iterrows()):
            if i == 0:
                self.performance_labels[cluster] = 'High'
            elif i == len(cluster_metrics) - 1:
                self.performance_labels[cluster] = 'Low'
            else:
                self.performance_labels[cluster] = 'Medium'
        
        # Add performance label to data
        self.data['performance'] = self.data['cluster'].map(self.performance_labels)
        
        # Map original clusters to 0, 1, 2 for consistency with visualization
        cluster_mapping = {}
        for i, cluster in enumerate(sorted(self.performance_labels.keys(), 
                                         key=lambda x: 0 if self.performance_labels[x] == 'High' else 
                                                      1 if self.performance_labels[x] == 'Medium' else 2)):
            cluster_mapping[cluster] = i
        
        # Add standardized prediction column (0=High, 1=Medium, 2=Low)
        self.data['prediction'] = self.data['cluster'].map(cluster_mapping)
        
        # Prepare cluster interpretation data
        interpretation = pd.DataFrame({
            'cluster': list(self.performance_labels.keys()),
            'performance': [self.performance_labels[c] for c in self.performance_labels.keys()],
            'count': [sum(self.data['cluster'] == c) for c in self.performance_labels.keys()],
            'percentage': [sum(self.data['cluster'] == c) / len(self.data) * 100 
                         for c in self.performance_labels.keys()],
            'avg_sales': [self.data[self.data['cluster'] == c]['total_sales'].mean() 
                        for c in self.performance_labels.keys()],
            'avg_processing_time': [self.data[self.data['cluster'] == c]['avg_processing_time'].mean() 
                                  for c in self.performance_labels.keys()],
            'avg_order_count': [self.data[self.data['cluster'] == c]['order_count'].mean() 
                              for c in self.performance_labels.keys()]
        })
        
        # Ensure the interpretation is sorted by performance
        interpretation['performance_order'] = interpretation['performance'].map(
            {'High': 0, 'Medium': 1, 'Low': 2}
        )
        interpretation = interpretation.sort_values('performance_order').drop('performance_order', axis=1)
        
        return self.data, interpretation
    
    def visualize_clusters(self, output_file="supplier_clusters.png"):
        """
        Create visualizations of supplier clusters
        
        Args:
            output_file: Path to save the visualization
        """
        if 'cluster' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        # Create PCA for visualization
        X = self.data[self.features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Create scatter plot with PCA
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        for cluster, color in enumerate(colors):
            # Get performance label for this cluster
            for original_cluster, label in self.performance_labels.items():
                if any(self.data['cluster'] == original_cluster) and \
                   all(self.data[self.data['cluster'] == original_cluster]['prediction'] == cluster):
                    performance = label
                    break
            else:
                performance = f"Cluster {cluster}"
            
            # Get points in this cluster
            mask = self.data['prediction'] == cluster
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=50, c=[color],
                label=f"{performance} Performers ({sum(mask)} sellers)"
            )
        
        # Add centroids
        # Transform cluster centers through PCA
        if self.cluster_centers is not None:
            # Ensure cluster centers are scaled the same way
            centers_scaled = scaler.transform(self.cluster_centers)
            centers_pca = pca.transform(centers_scaled)
            
            plt.scatter(
                centers_pca[:, 0], centers_pca[:, 1],
                s=200, c='black', marker='X',
                label='Cluster Centers'
            )
        
        # Add labels and title
        plt.title('Supplier Clusters (PCA Visualization)', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)', fontsize=14)
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add text description of clusters
        plt.figtext(0.01, 0.01, 
                   "Visualization uses PCA to reduce multiple features to 2 dimensions",
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        # Create feature importance visualization
        self.visualize_feature_importance()
        
    def visualize_feature_importance(self, output_file="feature_importance.png"):
        """
        Visualize feature importance for clustering
        
        Args:
            output_file: Path to save the visualization
        """
        if self.cluster_centers is None:
            raise ValueError("Must run cluster_suppliers first")
        
        # Extract feature importance from PCA
        X = self.data[self.features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Get feature loadings from the first two components
        loadings = pd.DataFrame(
            pca.components_.T[:, :2],
            columns=['PC1', 'PC2'],
            index=self.features
        )
        
        # Create bi-plot for feature importance
        plt.figure(figsize=(12, 10))
        
        # Plot feature vectors
        for i, feature in enumerate(loadings.index):
            plt.arrow(
                0, 0,  # Start at origin
                loadings.iloc[i, 0] * 3,  # Scale for visibility
                loadings.iloc[i, 1] * 3,
                head_width=0.1, head_length=0.1,
                fc=sns.color_palette()[i % 10],
                ec=sns.color_palette()[i % 10],
                label=feature
            )
            
            # Add feature name
            plt.text(
                loadings.iloc[i, 0] * 3.1,
                loadings.iloc[i, 1] * 3.1,
                feature,
                fontsize=12
            )
        
        # Add circle for reference
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(circle)
        
        # Set plot limits and labels
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)
        plt.xlabel("Principal Component 1", fontsize=14)
        plt.ylabel("Principal Component 2", fontsize=14)
        plt.title("Feature Importance in Supplier Clustering", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    def create_supplier_performance_report(self, output_file="supplier_performance.csv"):
        """
        Create a detailed supplier performance report
        
        Args:
            output_file: Path to save the CSV report
            
        Returns:
            DataFrame with supplier performance data
        """
        if 'cluster' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        # Create a copy of the data
        report_data = self.data.copy()
        
        # Sort by performance (High, Medium, Low) and then by total sales
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        report_data['performance_order'] = report_data['performance'].map(performance_order)
        report_data = report_data.sort_values(['performance_order', 'total_sales'], ascending=[True, False])
        
        # Calculate percentile ranks
        report_data['sales_percentile'] = report_data['total_sales'].rank(pct=True) * 100
        report_data['order_percentile'] = report_data['order_count'].rank(pct=True) * 100
        report_data['processing_time_percentile'] = (1 - report_data['avg_processing_time'].rank(pct=True)) * 100
        
        # Calculate overall performance score
        report_data['performance_score'] = (
            report_data['sales_percentile'] * 0.3 +
            report_data['order_percentile'] * 0.2 +
            report_data['processing_time_percentile'] * 0.25
        )
        
        # Add on-time delivery to score if available
        if 'on_time_delivery_rate' in report_data.columns:
            report_data['on_time_percentile'] = report_data['on_time_delivery_rate'].rank(pct=True) * 100
            report_data['performance_score'] += (report_data['on_time_percentile'] * 0.25)
        else:
            # Use delivery days if on-time rate isn't available
            report_data['delivery_percentile'] = (1 - report_data['avg_delivery_days'].rank(pct=True)) * 100
            report_data['performance_score'] += (report_data['delivery_percentile'] * 0.25)
        
        # Round score to 2 decimal places
        report_data['performance_score'] = report_data['performance_score'].round(2)
        
        # Select columns for the report
        report_columns = ['seller_id', 'performance', 'performance_score', 
                         'order_count', 'total_sales', 'avg_order_value',
                         'avg_processing_time', 'avg_delivery_days', 'prediction']
        
        # Add on-time delivery if available
        if 'on_time_delivery_rate' in report_data.columns:
            report_columns.append('on_time_delivery_rate')
            
        # Add shipping ratio if available
        if 'shipping_ratio' in report_data.columns:
            report_columns.append('shipping_ratio')
        
        # Select columns and save
        report = report_data[report_columns].copy()
        report.to_csv(output_file, index=False)
        
        return report
        
    def generate_improvement_recommendations(self, output_file="supplier_recommendations.csv"):
        """
        Generate improvement recommendations for low and medium performing suppliers
        
        Args:
            output_file: Path to save recommendations
            
        Returns:
            DataFrame with supplier recommendations
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        # Focus on low and medium performers
        suppliers_to_improve = self.data[self.data['performance'] != 'High'].copy()
        
        # Create recommendations
        recommendations = []
        
        for _, supplier in suppliers_to_improve.iterrows():
            supplier_id = supplier['seller_id']
            performance = supplier['performance']
            
            # Identify areas for improvement
            areas = []
            
            # Processing time issues
            if performance == 'Low' or supplier['avg_processing_time'] > self.data['avg_processing_time'].median():
                areas.append('processing_time')
                
            # Delivery time issues
            if performance == 'Low' or supplier['avg_delivery_days'] > self.data['avg_delivery_days'].median():
                areas.append('delivery_time')
                
            # Order volume issues
            if supplier['order_count'] < self.data['order_count'].median():
                areas.append('order_volume')
                
            # On-time delivery issues
            if 'on_time_delivery_rate' in supplier and supplier['on_time_delivery_rate'] < 85:
                areas.append('on_time_delivery')
            
            # Create specific recommendations based on areas
            for area in areas:
                priority = 'High' if performance == 'Low' else 'Medium'
                
                if area == 'processing_time':
                    recommendation = "Optimize order processing workflow to reduce processing time"
                    if supplier['avg_processing_time'] > 3:
                        recommendation += ". Consider implementing automated order processing."
                    action = "Reduce processing time by at least 25%"
                    
                elif area == 'delivery_time':
                    recommendation = "Improve delivery logistics and carrier selection"
                    if supplier['avg_delivery_days'] > 7:
                        recommendation += ". Consider partnering with faster shipping carriers."
                    action = "Reduce delivery time by at least 20%"
                    
                elif area == 'order_volume':
                    recommendation = "Increase visibility through promotional activities"
                    if supplier['order_count'] < 20:
                        recommendation += ". Consider product bundling strategies."
                    action = "Increase order volume by at least 30%"
                    
                elif area == 'on_time_delivery':
                    recommendation = "Improve on-time delivery rate through better inventory management"
                    action = "Achieve at least 90% on-time delivery rate"
                
                recommendations.append({
                    'seller_id': supplier_id,
                    'performance': performance,
                    'improvement_area': area,
                    'recommendation': recommendation,
                    'suggested_action': action,
                    'priority': priority
                })
        
        # Convert to DataFrame and save
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_csv(output_file, index=False)
        
        return recommendations_df
    
    def create_performance_benchmark_chart(self, output_file="performance_benchmarks.png"):
        """
        Create a performance benchmark chart comparing cluster metrics
        
        Args:
            output_file: Path to save the visualization
        """
        if self.performance_labels is None:
            raise ValueError("Must run cluster_suppliers first")
        
        # Create cluster performance summary
        cluster_metrics = {}
        
        for cluster, performance in self.performance_labels.items():
            cluster_data = self.data[self.data['cluster'] == cluster]
            
            metrics = {
                'performance': performance,
                'count': len(cluster_data),
                'avg_order_count': cluster_data['order_count'].mean(),
                'avg_processing_time': cluster_data['avg_processing_time'].mean(),
                'avg_delivery_days': cluster_data['avg_delivery_days'].mean(),
                'avg_sales': cluster_data['total_sales'].mean(),
                'avg_order_value': cluster_data['avg_order_value'].mean()
            }
            
            # Add on-time delivery if available
            if 'on_time_delivery_rate' in cluster_data.columns:
                metrics['avg_on_time_rate'] = cluster_data['on_time_delivery_rate'].mean()
                
            cluster_metrics[cluster] = metrics
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient='index')
        
        # Sort by performance
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        metrics_df['order'] = metrics_df['performance'].map(performance_order)
        metrics_df = metrics_df.sort_values('order').drop('order', axis=1)
        
        # Create radar chart for comparing clusters
        # Select metrics to include
        radar_metrics = ['avg_order_count', 'avg_processing_time', 'avg_delivery_days', 
                        'avg_sales', 'avg_order_value']
        
        # Add on-time delivery if available
        if 'avg_on_time_rate' in metrics_df.columns:
            radar_metrics.append('avg_on_time_rate')
        
        # Normalize data for radar chart
        radar_data = metrics_df[radar_metrics].copy()
        
        # Invert metrics where lower is better
        for col in ['avg_processing_time', 'avg_delivery_days']:
            if col in radar_data.columns:
                max_val = radar_data[col].max()
                radar_data[col] = max_val - radar_data[col]
        
        # Scale all values to 0-1 range
        for col in radar_data.columns:
            min_val = radar_data[col].min()
            max_val = radar_data[col].max()
            if max_val > min_val:
                radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
            else:
                radar_data[col] = 0.5  # Default value if all are the same
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each cluster
        for i, (_, row) in enumerate(radar_data.iterrows()):
            performance = metrics_df.loc[row.name, 'performance']
            values = row.values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=f"{performance} Performers")
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('avg_', '').replace('_', ' ').title() for m in radar_metrics])
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Supplier Performance Benchmark Comparison", fontsize=15, y=1.1)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        # Create bar chart comparison
        self.create_cluster_comparison_chart(metrics_df, output_file.replace('.png', '_comparison.png'))
        
    def create_cluster_comparison_chart(self, metrics_df, output_file="cluster_comparison.png"):
        """
        Create a bar chart comparing key metrics across clusters
        
        Args:
            metrics_df: DataFrame with cluster metrics
            output_file: Path to save the visualization
        """
        # Select metrics to compare
        compare_metrics = ['avg_order_count', 'avg_processing_time', 
                         'avg_delivery_days', 'avg_sales']
        
        # Add on-time delivery if available
        if 'avg_on_time_rate' in metrics_df.columns:
            compare_metrics.append('avg_on_time_rate')
        
        # Prepare data for plotting
        plot_data = []
        for metric in compare_metrics:
            for idx, row in metrics_df.iterrows():
                plot_data.append({
                    'Metric': metric.replace('avg_', '').replace('_', ' ').title(),
                    'Value': row[metric],
                    'Performance': row['performance']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        
        # Set color palette
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        
        # Create grouped bar chart
        ax = sns.barplot(
            data=plot_df,
            x='Metric',
            y='Value',
            hue='Performance',
            palette=colors
        )
        
        # Add labels and title
        plt.title("Supplier Cluster Metric Comparison", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Performance Tier")
        
        # Format y-axis for large values
        if 'Sales' in plot_df['Metric'].values:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k' if x >= 1000 else f'${x:.0f}')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    def create_supplier_distribution_chart(self, output_file="supplier_distribution.png"):
        """
        Create a chart showing supplier distribution across performance tiers
        
        Args:
            output_file: Path to save the visualization
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        # Count suppliers in each performance tier
        performance_counts = self.data['performance'].value_counts().reset_index()
        performance_counts.columns = ['Performance', 'Count']
        
        # Add percentage
        performance_counts['Percentage'] = (performance_counts['Count'] / performance_counts['Count'].sum() * 100).round(1)
        
        # Order by performance level
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        performance_counts['Order'] = performance_counts['Performance'].map(performance_order)
        performance_counts = performance_counts.sort_values('Order').drop('Order', axis=1)
        
        # Create pie chart
        plt.figure(figsize=(10, 10))
        
        # Set color palette
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        plot_colors = [colors[p] for p in performance_counts['Performance']]
        
        # Create pie chart
        plt.pie(
            performance_counts['Count'],
            labels=[f"{p} ({c} suppliers, {pct}%)" for p, c, pct in 
                  zip(performance_counts['Performance'], 
                     performance_counts['Count'], 
                     performance_counts['Percentage'])],
            colors=plot_colors,
            autopct='',
            startangle=90,
            shadow=False,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        
        # Add title
        plt.title("Supplier Distribution by Performance Tier", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        # Create bar chart showing supplier counts by tier
        plt.figure(figsize=(8, 6))
        
        ax = sns.barplot(
            data=performance_counts,
            x='Performance',
            y='Count',
            palette=plot_colors
        )
        
        # Add data labels on top of bars
        for i, v in enumerate(performance_counts['Count']):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)
        
        # Add percentages below x-axis labels
        for i, pct in enumerate(performance_counts['Percentage']):
            ax.text(i, -5, f"{pct}%", ha='center', va='top', fontsize=10)
        
        # Add labels and title
        plt.title("Supplier Count by Performance Tier", fontsize=14)
        plt.xlabel("")  # Remove x-axis label
        plt.ylabel("Number of Suppliers", fontsize=12)
        
        # Adjust y-axis to fit data labels
        plt.ylim(0, max(performance_counts['Count']) * 1.1)
        
        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '_bar.png'), dpi=300)
        plt.close()
        
    def analyze_supplier_trends(self, output_file="supplier_trends.png"):
        """
        Create visualization of key supplier metrics by performance tier
        
        Args:
            output_file: Path to save the visualization
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
            
        # Create box plots for key metrics by performance tier
        metrics = ['order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
        
        # Add on-time delivery if available
        if 'on_time_delivery_rate' in self.data.columns:
            metrics.append('on_time_delivery_rate')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Performance tier color mapping
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        
        # Create box plots for each metric
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics for the grid
            # Get pretty name for metric
            metric_name = metric.replace('_', ' ').title()
            
            # Create box plot
            sns.boxplot(
                ax=axes[i],
                data=self.data,
                x='performance',
                y=metric,
                order=['High', 'Medium', 'Low'],
                palette=colors
            )
            
            # Add labels
            axes[i].set_title(f"{metric_name} by Performance Tier", fontsize=14)
            axes[i].set_xlabel("")
            axes[i].set_ylabel(metric_name)
            
            # Format y-axis for sales
            if metric == 'total_sales':
                axes[i].yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k' if x >= 1000 else f'${x:.0f}')
        
        # Add overall title
        plt.suptitle("Supplier Metrics by Performance Tier", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    def run_analysis(self, output_dir="./output"):
        """
        Run complete supplier analysis and generate reports and visualizations
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with analysis results
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess data
        self.preprocess_data()
        
        # Cluster suppliers
        print(f"Clustering suppliers into {self.n_clusters} clusters...")
        clustered_data, interpretation = self.cluster_suppliers()
        
        # Save clustered data
        clustered_data.to_csv(os.path.join(output_dir, "seller_clusters_enhanced.csv"), index=False)
        interpretation.to_csv(os.path.join(output_dir, "cluster_interpretation.csv"), index=False)
        
        # Generate visualizations
        print("Generating visualizations...")
        self.visualize_clusters(os.path.join(output_dir, "supplier_clusters_viz.png"))
        self.create_performance_benchmark_chart(os.path.join(output_dir, "performance_benchmarks.png"))
        self.create_supplier_distribution_chart(os.path.join(output_dir, "supplier_distribution.png"))
        self.analyze_supplier_trends(os.path.join(output_dir, "supplier_trends.png"))
        
        # Generate reports
        print("Generating reports...")
        performance_report = self.create_supplier_performance_report(
            os.path.join(output_dir, "supplier_performance_report.csv")
        )
        recommendations = self.generate_improvement_recommendations(
            os.path.join(output_dir, "supplier_recommendations.csv")
        )
        
        # Print summary
        print(f"\nSupplier Analysis Complete:")
        print(f"- Total suppliers analyzed: {len(self.data)}")
        print(f"- Performance tiers: {len(self.performance_labels)} (High, Medium, Low)")
        
        for performance in ['High', 'Medium', 'Low']:
            count = sum(self.data['performance'] == performance)
            percentage = (count / len(self.data)) * 100
            print(f"- {performance} performers: {count} ({percentage:.1f}%)")
        
        print(f"\nResults saved to {output_dir}")
        
        return {
            'clustered_data': clustered_data,
            'interpretation': interpretation,
            'performance_report': performance_report,
            'recommendations': recommendations
        }

# If run as a script
if __name__ == "__main__":
    # Example usage
    analyzer = SupplierAnalyzer("seller_clusters.csv")
    results = analyzer.run_analysis("./output")
    
    print("Supplier analysis complete.")