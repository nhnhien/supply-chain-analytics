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
    based on multiple performance metrics.
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
        
        # Convert seller_id to string
        self.data['seller_id'] = self.data['seller_id'].astype(str)
        
        # Fill missing numeric values with median
        for col in self.data.columns:
            if col != 'seller_id' and pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Winsorize extreme total_sales at 99th percentile
        cap = self.data['total_sales'].quantile(0.99)
        self.data['total_sales'] = self.data['total_sales'].clip(upper=cap)
        print(f"Capped extreme seller sales at {cap:.2f}")
        
        # Create additional clustering features
        if 'avg_order_value' not in self.data.columns:
            self.data['avg_order_value'] = (self.data['total_sales'] / self.data['order_count']).replace([np.inf, -np.inf], np.nan)
            self.data['avg_order_value'] = self.data['avg_order_value'].fillna(self.data['avg_order_value'].median())
        
        if 'on_time_delivery_rate' not in self.data.columns and 'on_time_delivery' in self.data.columns:
            self.data['on_time_delivery_rate'] = self.data['on_time_delivery'] * 100
        
        if 'shipping_costs' in self.data.columns:
            self.data['shipping_ratio'] = (self.data['shipping_costs'] / self.data['total_sales'] * 100).replace([np.inf, -np.inf], np.nan)
            self.data['shipping_ratio'] = self.data['shipping_ratio'].fillna(self.data['shipping_ratio'].median())
        
        self.features = [
            'order_count', 'avg_processing_time', 'avg_delivery_days',
            'total_sales', 'avg_order_value'
        ]
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
            
        X = self.data[self.features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        silhouette_scores = []
        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            try:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
            except Exception as e:
                print(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
                silhouette_scores.append(0)
        
        if silhouette_scores:
            optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            optimal_clusters = 3
        print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    def cluster_suppliers(self, n_clusters=None):
        """
        Cluster suppliers based on performance metrics with improved balancing.
        
        Args:
            n_clusters: Number of clusters to create (if None, will determine optimal)
            
        Returns:
            Tuple of (DataFrame with cluster assignments, interpretation DataFrame)
        """
        if len(self.features) == 0:
            self.preprocess_data()
            
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        self.n_clusters = n_clusters
        
        X = self.data[self.features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        self.cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=self.features
        )
        
        # Calculate seller scores using normalized performance metrics
        self.data['score'] = (
            (self.data['order_count'] / self.data['order_count'].max()) * 20 +
            (self.data['total_sales'] / self.data['total_sales'].max()) * 30 +
            (1 - (self.data['avg_processing_time'] / self.data['avg_processing_time'].max())) * 25
        )
        if 'on_time_delivery_rate' in self.data.columns:
            self.data['score'] += (self.data['on_time_delivery_rate'] / 100) * 25
        else:
            self.data['score'] += (1 - (self.data['avg_delivery_days'] / self.data['avg_delivery_days'].max())) * 25
        
        # Use quantile thresholds to assign performance labels
        high_threshold = self.data['score'].quantile(0.70)
        low_threshold = self.data['score'].quantile(0.30)
        self.data['performance'] = 'Medium'
        self.data.loc[self.data['score'] >= high_threshold, 'performance'] = 'High'
        self.data.loc[self.data['score'] <= low_threshold, 'performance'] = 'Low'
        
        # Map performance labels to standard prediction values (0=High, 1=Medium, 2=Low)
        performance_mapping = {'High': 0, 'Medium': 1, 'Low': 2}
        self.data['prediction'] = self.data['performance'].map(performance_mapping)
        
        # For consistent cluster interpretation, use the derived "prediction" values
        self.performance_labels = {}
        for pred in sorted(self.data['prediction'].unique()):
            cluster_df = self.data[self.data['prediction'] == pred]
            most_common = cluster_df['performance'].mode()[0]
            self.performance_labels[pred] = most_common
        
        interpretation = pd.DataFrame({
            'cluster': list(self.performance_labels.keys()),
            'performance': [self.performance_labels[k] for k in self.performance_labels.keys()],
            'count': [sum(self.data['prediction'] == k) for k in self.performance_labels.keys()],
            'percentage': [sum(self.data['prediction'] == k) / len(self.data) * 100 for k in self.performance_labels.keys()],
            'avg_sales': [self.data[self.data['prediction'] == k]['total_sales'].mean() for k in self.performance_labels.keys()],
            'avg_processing_time': [self.data[self.data['prediction'] == k]['avg_processing_time'].mean() for k in self.performance_labels.keys()],
            'avg_order_count': [self.data[self.data['prediction'] == k]['order_count'].mean() for k in self.performance_labels.keys()]
        })
        
        # Sort interpretation by standard performance order: High, Medium, Low
        interpretation['performance_order'] = interpretation['performance'].map(
            {'High': 0, 'Medium': 1, 'Low': 2}
        )
        interpretation = interpretation.sort_values('performance_order').drop('performance_order', axis=1)
        
        return self.data, interpretation
    
    def visualize_clusters(self, output_file="supplier_clusters.png"):
        """
        Create visualizations of supplier clusters.
        
        Args:
            output_file: Path to save the visualization
        """
        if 'cluster' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        X = self.data[self.features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        
        plt.figure(figsize=(12, 10))
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        
        # Now loop over unique prediction values (0=High, 1=Medium, 2=Low)
        for pred, color in zip(sorted(self.data['prediction'].unique()), colors):
            mask = self.data['prediction'] == pred
            performance = self.performance_labels.get(pred, f"Cluster {pred}")
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=50, c=[color],
                label=f"{performance} Performers ({sum(mask)} sellers)"
            )
        
        if self.cluster_centers is not None:
            centers_scaled = scaler.transform(self.cluster_centers)
            centers_pca = pca.transform(centers_scaled)
            plt.scatter(
                centers_pca[:, 0], centers_pca[:, 1],
                s=200, c='black', marker='X',
                label='Cluster Centers'
            )
        
        plt.title('Supplier Clusters (PCA Visualization)', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)', fontsize=14)
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.figtext(0.01, 0.01, "Visualization uses PCA to reduce multiple features to 2 dimensions", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        self.visualize_feature_importance()
        
    def visualize_feature_importance(self, output_file="feature_importance.png"):
        """
        Visualize feature importance for clustering.
        
        Args:
            output_file: Path to save the visualization
        """
        if self.cluster_centers is None:
            raise ValueError("Must run cluster_suppliers first")
        
        X = self.data[self.features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        
        loadings = pd.DataFrame(
            pca.components_.T[:, :2],
            columns=['PC1', 'PC2'],
            index=self.features
        )
        
        plt.figure(figsize=(12, 10))
        for i, feature in enumerate(loadings.index):
            plt.arrow(
                0, 0,
                loadings.iloc[i, 0] * 3,
                loadings.iloc[i, 1] * 3,
                head_width=0.1, head_length=0.1,
                fc=sns.color_palette()[i % 10],
                ec=sns.color_palette()[i % 10],
                label=feature
            )
            plt.text(
                loadings.iloc[i, 0] * 3.1,
                loadings.iloc[i, 1] * 3.1,
                feature,
                fontsize=12
            )
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(circle)
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
        Create a detailed supplier performance report.
        
        Args:
            output_file: Path to save the CSV report
            
        Returns:
            DataFrame with supplier performance data
        """
        if 'cluster' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        report_data = self.data.copy()
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        report_data['performance_order'] = report_data['performance'].map(performance_order)
        report_data = report_data.sort_values(['performance_order', 'total_sales'], ascending=[True, False])
        
        report_data['sales_percentile'] = report_data['total_sales'].rank(pct=True) * 100
        report_data['order_percentile'] = report_data['order_count'].rank(pct=True) * 100
        report_data['processing_time_percentile'] = (1 - report_data['avg_processing_time'].rank(pct=True)) * 100
        
        report_data['performance_score'] = (
            report_data['sales_percentile'] * 0.3 +
            report_data['order_percentile'] * 0.2 +
            report_data['processing_time_percentile'] * 0.25
        )
        if 'on_time_delivery_rate' in report_data.columns:
            report_data['on_time_percentile'] = report_data['on_time_delivery_rate'].rank(pct=True) * 100
            report_data['performance_score'] += (report_data['on_time_percentile'] * 0.25)
        else:
            report_data['delivery_percentile'] = (1 - report_data['avg_delivery_days'].rank(pct=True)) * 100
            report_data['performance_score'] += (report_data['delivery_percentile'] * 0.25)
        
        report_data['performance_score'] = report_data['performance_score'].round(2)
        
        report_columns = ['seller_id', 'performance', 'performance_score', 
                         'order_count', 'total_sales', 'avg_order_value',
                         'avg_processing_time', 'avg_delivery_days', 'prediction']
        if 'on_time_delivery_rate' in report_data.columns:
            report_columns.append('on_time_delivery_rate')
        if 'shipping_ratio' in report_data.columns:
            report_columns.append('shipping_ratio')
        
        report = report_data[report_columns].copy()
        report.to_csv(output_file, index=False)
        return report
        
    def generate_improvement_recommendations(self, output_file="supplier_recommendations.csv"):
        """
        Generate improvement recommendations for low and medium performing suppliers.
        
        Args:
            output_file: Path to save recommendations
            
        Returns:
            DataFrame with supplier recommendations
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        suppliers_to_improve = self.data[self.data['performance'] != 'High'].copy()
        recommendations = []
        for _, supplier in suppliers_to_improve.iterrows():
            supplier_id = supplier['seller_id']
            performance = supplier['performance']
            areas = []
            if performance == 'Low' or supplier['avg_processing_time'] > self.data['avg_processing_time'].median():
                areas.append('processing_time')
            if performance == 'Low' or supplier['avg_delivery_days'] > self.data['avg_delivery_days'].median():
                areas.append('delivery_time')
            if supplier['order_count'] < self.data['order_count'].median():
                areas.append('order_volume')
            if 'on_time_delivery_rate' in supplier and supplier['on_time_delivery_rate'] < 85:
                areas.append('on_time_delivery')
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
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_csv(output_file, index=False)
        return recommendations_df
    
    def create_performance_benchmark_chart(self, output_file="performance_benchmarks.png"):
        """
        Create a performance benchmark chart comparing cluster metrics.
        
        Args:
            output_file: Path to save the visualization
        """
        if self.performance_labels is None:
            raise ValueError("Must run cluster_suppliers first")
        
        cluster_metrics = {}
        for pred, performance in self.performance_labels.items():
            cluster_data = self.data[self.data['prediction'] == pred]
            metrics = {
                'performance': performance,
                'count': len(cluster_data),
                'avg_order_count': cluster_data['order_count'].mean(),
                'avg_processing_time': cluster_data['avg_processing_time'].mean(),
                'avg_delivery_days': cluster_data['avg_delivery_days'].mean(),
                'avg_sales': cluster_data['total_sales'].mean(),
                'avg_order_value': cluster_data['avg_order_value'].mean()
            }
            if 'on_time_delivery_rate' in cluster_data.columns:
                metrics['avg_on_time_rate'] = cluster_data['on_time_delivery_rate'].mean()
            cluster_metrics[pred] = metrics
        
        metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient='index')
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        metrics_df['order'] = metrics_df['performance'].map(performance_order)
        metrics_df = metrics_df.sort_values('order').drop('order', axis=1)
        
        radar_metrics = ['avg_order_count', 'avg_processing_time', 'avg_delivery_days', 
                        'avg_sales', 'avg_order_value']
        if 'avg_on_time_rate' in metrics_df.columns:
            radar_metrics.append('avg_on_time_rate')
        
        radar_data = metrics_df[radar_metrics].copy()
        for col in ['avg_processing_time', 'avg_delivery_days']:
            if col in radar_data.columns:
                max_val = radar_data[col].max()
                radar_data[col] = max_val - radar_data[col]
        for col in radar_data.columns:
            min_val = radar_data[col].min()
            max_val = radar_data[col].max()
            if max_val > min_val:
                radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
            else:
                radar_data[col] = 0.5
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for i, (_, row) in enumerate(radar_data.iterrows()):
            performance = metrics_df.loc[row.name, 'performance']
            values = row.values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=f"{performance} Performers")
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('avg_', '').replace('_', ' ').title() for m in radar_metrics])
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Supplier Performance Benchmark Comparison", fontsize=15, y=1.1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        self.create_cluster_comparison_chart(metrics_df, output_file.replace('.png', '_comparison.png'))
        
    def create_cluster_comparison_chart(self, metrics_df, output_file="cluster_comparison.png"):
        """
        Create a bar chart comparing key metrics across clusters.
        
        Args:
            metrics_df: DataFrame with cluster metrics
            output_file: Path to save the visualization
        """
        compare_metrics = ['avg_order_count', 'avg_processing_time', 
                         'avg_delivery_days', 'avg_sales']
        if 'avg_on_time_rate' in metrics_df.columns:
            compare_metrics.append('avg_on_time_rate')
        plot_data = []
        for metric in compare_metrics:
            for idx, row in metrics_df.iterrows():
                plot_data.append({
                    'Metric': metric.replace('avg_', '').replace('_', ' ').title(),
                    'Value': row[metric],
                    'Performance': row['performance']
                })
        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(14, 8))
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        ax = sns.barplot(
            data=plot_df,
            x='Metric',
            y='Value',
            hue='Performance',
            palette=colors
        )
        plt.title("Supplier Cluster Metric Comparison", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Performance Tier")
        if 'Sales' in plot_df['Metric'].values:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k' if x >= 1000 else f'${x:.0f}')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    def create_supplier_distribution_chart(self, output_file="supplier_distribution.png"):
        """
        Create a chart showing supplier distribution across performance tiers.
        
        Args:
            output_file: Path to save the visualization
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
        
        performance_counts = self.data['performance'].value_counts().reset_index()
        performance_counts.columns = ['Performance', 'Count']
        performance_counts['Percentage'] = (performance_counts['Count'] / performance_counts['Count'].sum() * 100).round(1)
        performance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        performance_counts['Order'] = performance_counts['Performance'].map(performance_order)
        performance_counts = performance_counts.sort_values('Order').drop('Order', axis=1)
        
        plt.figure(figsize=(10, 10))
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        plot_colors = [colors[p] for p in performance_counts['Performance']]
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
        plt.axis('equal')
        plt.title("Supplier Distribution by Performance Tier", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=performance_counts,
            x='Performance',
            y='Count',
            palette=plot_colors
        )
        for i, v in enumerate(performance_counts['Count']):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)
        for i, pct in enumerate(performance_counts['Percentage']):
            ax.text(i, -5, f"{pct}%", ha='center', va='top', fontsize=10)
        plt.title("Supplier Count by Performance Tier", fontsize=14)
        plt.xlabel("")
        plt.ylabel("Number of Suppliers", fontsize=12)
        plt.ylim(0, max(performance_counts['Count']) * 1.1)
        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '_bar.png'), dpi=300)
        plt.close()
        
    def analyze_supplier_trends(self, output_file="supplier_trends.png"):
        """
        Create visualization of key supplier metrics by performance tier.
        
        Args:
            output_file: Path to save the visualization
        """
        if 'performance' not in self.data.columns:
            raise ValueError("Must run cluster_suppliers first")
            
        metrics = ['order_count', 'avg_processing_time', 'avg_delivery_days', 'total_sales']
        if 'on_time_delivery_rate' in self.data.columns:
            metrics.append('on_time_delivery_rate')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        colors = {'High': '#1f77b4', 'Medium': '#ff7f0e', 'Low': '#d62728'}
        for i, metric in enumerate(metrics[:4]):
            metric_name = metric.replace('_', ' ').title()
            sns.boxplot(
                ax=axes[i],
                data=self.data,
                x='performance',
                y=metric,
                order=['High', 'Medium', 'Low'],
                palette=colors
            )
            axes[i].set_title(f"{metric_name} by Performance Tier", fontsize=14)
            axes[i].set_xlabel("")
            axes[i].set_ylabel(metric_name)
            if metric == 'total_sales':
                axes[i].yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k' if x >= 1000 else f'${x:.0f}')
        plt.suptitle("Supplier Metrics by Performance Tier", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    def run_analysis(self, output_dir="./output"):
        """
        Run complete supplier analysis and generate reports and visualizations.
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.preprocess_data()
        print(f"Clustering suppliers into {self.n_clusters} clusters...")
        clustered_data, interpretation = self.cluster_suppliers()
        clustered_data.to_csv(os.path.join(output_dir, "seller_clusters_enhanced.csv"), index=False)
        interpretation.to_csv(os.path.join(output_dir, "cluster_interpretation.csv"), index=False)
        
        print("Generating visualizations...")
        self.visualize_clusters(os.path.join(output_dir, "supplier_clusters_viz.png"))
        self.create_performance_benchmark_chart(os.path.join(output_dir, "performance_benchmarks.png"))
        self.create_supplier_distribution_chart(os.path.join(output_dir, "supplier_distribution.png"))
        self.analyze_supplier_trends(os.path.join(output_dir, "supplier_trends.png"))
        
        print("Generating reports...")
        performance_report = self.create_supplier_performance_report(os.path.join(output_dir, "supplier_performance_report.csv"))
        recommendations = self.generate_improvement_recommendations(os.path.join(output_dir, "supplier_recommendations.csv"))
        
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
    analyzer = SupplierAnalyzer("seller_clusters.csv")
    results = analyzer.run_analysis("./output")
    print("Supplier analysis complete.")
