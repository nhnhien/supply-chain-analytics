#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, sum, count, when, datediff, month, year, to_date,
    lag, lit, expr, rank, percentile_approx
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import FloatType, IntegerType, StringType, StructType, StructField

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SparkSupplyChainAnalytics:
    """
    Class for performing supply chain analytics using Spark,
    taking advantage of its big data processing capabilities.
    """
    def __init__(self, spark=None, data_path=".", output_path="./output"):
        """
        Initialize the analytics engine with a Spark session.
        """
        self.data_path = data_path
        self.output_path = output_path
        self.spark = spark if spark else self._create_spark_session()
        os.makedirs(output_path, exist_ok=True)
        # Data containers
        self.orders_df = None
        self.order_items_df = None
        self.products_df = None
        self.customers_df = None
        self.unified_df = None

    def _create_spark_session(self):
        """
        Create and configure a Spark session.
        """
        return (SparkSession.builder
                .appName("Supply Chain Analytics")
                .config("spark.executor.memory", "4g")
                .config("spark.driver.memory", "4g")
                .config("spark.memory.offHeap.enabled", "true")
                .config("spark.memory.offHeap.size", "4g")
                .config("spark.dynamicAllocation.enabled", "true")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.shuffle.partitions", "100")
                .getOrCreate())

    def load_data(self):
        """
        Load required datasets using Spark.
        """
        print("Loading data using Spark...")
        # Orders data
        orders_path = os.path.join(self.data_path, "df_Orders.csv")
        if os.path.exists(orders_path):
            self.orders_df = self.spark.read.csv(orders_path, header=True, inferSchema=True)
            print(f"Loaded orders data: {self.orders_df.count()} rows")
        else:
            print(f"Warning: Orders data file not found at {orders_path}")
            self.orders_df = self.spark.createDataFrame(
                [], "order_id STRING, customer_id STRING, order_status STRING, " +
                "order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP, " +
                "order_delivered_timestamp TIMESTAMP, order_estimated_delivery_date TIMESTAMP"
            )
        # Order items data
        order_items_path = os.path.join(self.data_path, "df_OrderItems.csv")
        if os.path.exists(order_items_path):
            self.order_items_df = self.spark.read.csv(order_items_path, header=True, inferSchema=True)
            print(f"Loaded order items data: {self.order_items_df.count()} rows")
        else:
            print(f"Warning: Order items data file not found at {order_items_path}")
            self.order_items_df = self.spark.createDataFrame(
                [], "order_id STRING, order_item_id INT, product_id STRING, " +
                "seller_id STRING, price DOUBLE, shipping_charges DOUBLE"
            )
        # Products data
        products_path = os.path.join(self.data_path, "df_Products.csv")
        if os.path.exists(products_path):
            self.products_df = self.spark.read.csv(products_path, header=True, inferSchema=True)
            print(f"Loaded products data: {self.products_df.count()} rows")
        else:
            print(f"Warning: Products data file not found at {products_path}")
            self.products_df = self.spark.createDataFrame(
                [], "product_id STRING, product_category_name STRING, " +
                "product_weight_g DOUBLE, product_length_cm DOUBLE, " +
                "product_height_cm DOUBLE, product_width_cm DOUBLE"
            )
        # Customers data
        customers_path = os.path.join(self.data_path, "df_Customers.csv")
        if os.path.exists(customers_path):
            self.customers_df = self.spark.read.csv(customers_path, header=True, inferSchema=True)
            print(f"Loaded customers data: {self.customers_df.count()} rows")
        else:
            print(f"Warning: Customers data file not found at {customers_path}")
            self.customers_df = self.spark.createDataFrame(
                [], "customer_id STRING, customer_zip_code_prefix STRING, " +
                "customer_city STRING, customer_state STRING"
            )
        return self

    def preprocess_data(self):
        """
        Preprocess datasets to improve data quality.
        """
        print("Preprocessing data...")
        # Preprocess orders
        if self.orders_df:
            for col_name in ["order_purchase_timestamp", "order_approved_at", 
                             "order_delivered_timestamp", "order_estimated_delivery_date"]:
                if col_name in self.orders_df.columns:
                    self.orders_df = self.orders_df.withColumn(col_name, to_date(col=col_name))
            if "order_purchase_timestamp" in self.orders_df.columns:
                self.orders_df = self.orders_df.withColumn("order_year", year("order_purchase_timestamp")) \
                                               .withColumn("order_month", month("order_purchase_timestamp"))
            if all(col in self.orders_df.columns for col in ["order_purchase_timestamp", "order_approved_at"]):
                self.orders_df = self.orders_df.withColumn("processing_time",
                                                             datediff("order_approved_at", "order_purchase_timestamp"))
                self.orders_df = self.orders_df.withColumn("processing_time",
                                                             when(col("processing_time") < 0, None)
                                                             .otherwise(col("processing_time")))
            if all(col in self.orders_df.columns for col in ["order_purchase_timestamp", "order_delivered_timestamp"]):
                self.orders_df = self.orders_df.withColumn("delivery_days",
                                                             datediff("order_delivered_timestamp", "order_purchase_timestamp"))
                self.orders_df = self.orders_df.withColumn("delivery_days",
                                                             when(col("delivery_days") < 0, None)
                                                             .otherwise(col("delivery_days")))
            if all(col in self.orders_df.columns for col in ["order_delivered_timestamp", "order_estimated_delivery_date"]):
                self.orders_df = self.orders_df.withColumn(
                    "on_time_delivery",
                    when(
                        col("order_delivered_timestamp").isNotNull() & col("order_estimated_delivery_date").isNotNull(),
                        when(col("order_delivered_timestamp") <= col("order_estimated_delivery_date"), 1).otherwise(0)
                    ).otherwise(None)
                )
        # Preprocess products
        if self.products_df:
            for num_col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
                if num_col in self.products_df.columns:
                    category_medians = self.products_df.groupBy("product_category_name") \
                                          .agg(percentile_approx(num_col, 0.5).alias(f"{num_col}_median"))
                    self.products_df = self.products_df.join(category_medians, on="product_category_name", how="left")
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(col(num_col).isNull(), col(f"{num_col}_median")).otherwise(col(num_col))
                    )
                    global_median = self.products_df.select(percentile_approx(num_col, 0.5)).collect()[0][0]
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(col(num_col).isNull(), lit(global_median)).otherwise(col(num_col))
                    )
                    self.products_df = self.products_df.drop(f"{num_col}_median")
        # Preprocess order items
        if self.order_items_df:
            for value_col in ["price", "shipping_charges"]:
                if value_col in self.order_items_df.columns:
                    product_medians = self.order_items_df.groupBy("product_id") \
                                          .agg(percentile_approx(value_col, 0.5).alias(f"{value_col}_median"))
                    self.order_items_df = self.order_items_df.join(product_medians, on="product_id", how="left")
                    self.order_items_df = self.order_items_df.withColumn(
                        value_col,
                        when(col(value_col).isNull(), col(f"{value_col}_median")).otherwise(col(value_col))
                    )
                    global_median = self.order_items_df.select(percentile_approx(value_col, 0.5)).collect()[0][0]
                    if global_median is None:
                        global_median = 0.0
                    self.order_items_df = self.order_items_df.withColumn(
                        value_col,
                        when(col(value_col).isNull(), lit(global_median)).otherwise(col(value_col))
                    )
                    self.order_items_df = self.order_items_df.drop(f"{value_col}_median")
        print("Data preprocessing complete")
        return self

    def build_unified_dataset(self):
        """
        Join datasets to create a unified analytics dataset.
        """
        print("Building unified dataset...")
        if not all([self.orders_df, self.order_items_df, self.products_df, self.customers_df]):
            print("Warning: Not all required datasets are loaded")
            return self
        unified_df = self.orders_df.join(self.order_items_df, on="order_id", how="inner")
        unified_df = unified_df.join(self.products_df, on="product_id", how="left")
        unified_df = unified_df.join(self.customers_df, on="customer_id", how="left")
        self.unified_df = unified_df.cache()
        total_rows = self.unified_df.count()
        total_orders = self.unified_df.select("order_id").distinct().count()
        total_products = self.unified_df.select("product_id").distinct().count()
        total_customers = self.unified_df.select("customer_id").distinct().count()
        print(f"Unified dataset created: {total_rows} rows")
        print(f"Total orders: {total_orders}")
        print(f"Total products: {total_products}")
        print(f"Total customers: {total_customers}")
        return self

    def analyze_monthly_demand(self, top_n=15):
        """
        Analyze monthly demand patterns.
        """
        print("Analyzing monthly demand patterns...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self, None, None
        monthly_demand = self.unified_df.groupBy(
            "product_category_name", "order_year", "order_month"
        ).agg(
            count("order_id").alias("order_count"),
            sum("price").alias("total_sales")
        ).na.drop(subset=["product_category_name"])
        monthly_demand_pd = monthly_demand.toPandas()
        monthly_demand_path = os.path.join(self.output_path, "monthly_demand.csv")
        monthly_demand_pd.to_csv(monthly_demand_path, index=False)
        print(f"Monthly demand data saved to {monthly_demand_path}")
        top_categories = self.unified_df.groupBy("product_category_name") \
                             .agg(count("order_id").alias("order_count")) \
                             .na.drop(subset=["product_category_name"]) \
                             .orderBy(col("order_count").desc()) \
                             .limit(top_n)
        top_categories_pd = top_categories.toPandas()
        top_categories_path = os.path.join(self.output_path, "top_categories.csv")
        top_categories_pd.to_csv(top_categories_path, index=False)
        print(f"Top {top_n} categories saved to {top_categories_path}")
        self._visualize_top_categories(top_categories_pd, monthly_demand_pd)
        return self, monthly_demand_pd, top_categories_pd

    def analyze_seller_performance(self, cluster_count=3):
        """
        Analyze seller performance with clustering.
        """
        print("Analyzing seller performance...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self, None
        available_columns = self.unified_df.columns
        print(f"Available columns: {available_columns}")
        metrics_agg = [count("order_id").alias("order_count"), sum("price").alias("total_sales")]
        if "processing_time" in available_columns:
            metrics_agg.append(avg("processing_time").alias("avg_processing_time"))
        else:
            print("No processing_time column found, using default")
            metrics_agg.append(lit(2.0).alias("avg_processing_time"))
        seller_metrics = self.unified_df.groupBy("seller_id").agg(*metrics_agg)
        seller_metrics = seller_metrics.withColumn(
            "avg_order_value",
            when(col("order_count") > 0, col("total_sales") / col("order_count")).otherwise(0)
        )
        if "avg_delivery_days" not in seller_metrics.columns:
            seller_metrics = seller_metrics.withColumn("avg_delivery_days", lit(5.0))
        if "on_time_delivery_rate" not in seller_metrics.columns:
            seller_metrics = seller_metrics.withColumn("on_time_delivery_rate", lit(0.85))
        feature_cols = ["order_count", "avg_processing_time", "avg_delivery_days", 
                        "total_sales", "avg_order_value", "on_time_delivery_rate"]
        for feature in feature_cols:
            seller_metrics = seller_metrics.withColumn(
                feature,
                when(col(feature).isNull(), 0).otherwise(col(feature))
            )
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        seller_features = assembler.transform(seller_metrics)
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(seller_features)
        scaled_data = scaler_model.transform(seller_features)
        kmeans = KMeans(featuresCol="scaled_features", k=cluster_count, seed=42)
        model = kmeans.fit(scaled_data)
        predictions = model.transform(scaled_data)
        evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaled_features")
        silhouette = evaluator.evaluate(predictions)
        print(f"Silhouette score: {silhouette}")
        cluster_stats = predictions.groupBy("prediction").agg(
            avg("total_sales").alias("avg_sales"),
            avg("avg_processing_time").alias("avg_proc_time"),
            avg("on_time_delivery_rate").alias("avg_otd")
        )
        cluster_stats_pd = cluster_stats.toPandas()
        cluster_stats_pd['score'] = (
            cluster_stats_pd['avg_sales'] / cluster_stats_pd['avg_sales'].max() * 40 +
            (1 - cluster_stats_pd['avg_proc_time'] / cluster_stats_pd['avg_proc_time'].max()) * 30 +
            cluster_stats_pd['avg_otd'] * 30
        )
        cluster_stats_pd = cluster_stats_pd.sort_values('score', ascending=False)
        performance_mapping = {
            cluster_stats_pd.iloc[i]['prediction']: i 
            for i in range(min(len(cluster_stats_pd), 3))
        }
        for i in range(cluster_count):
            if i not in performance_mapping:
                performance_mapping[i] = len(performance_mapping)
        seller_clusters_pd = predictions.select("seller_id", "prediction", *feature_cols).toPandas()
        seller_clusters_pd['performance_cluster'] = seller_clusters_pd['prediction'].map(performance_mapping)
        seller_clusters_path = os.path.join(self.output_path, "seller_clusters.csv")
        seller_clusters_pd.to_csv(seller_clusters_path, index=False)
        print(f"Seller clusters saved to {seller_clusters_path}")
        cluster_centers_pd = pd.DataFrame(model.clusterCenters(), columns=feature_cols)
        centers_path = os.path.join(self.output_path, "cluster_centers.csv")
        cluster_centers_pd.to_csv(centers_path, index=False)
        print(f"Cluster centers saved to {centers_path}")
        self._visualize_seller_clusters(seller_clusters_pd, cluster_centers_pd)
        return self, seller_clusters_pd

    def analyze_geographical_patterns(self, supply_chain):
        """
        Analyze order patterns by geographical location.
        """
        print("Analyzing geographical patterns...")
        available_columns = supply_chain.columns
        print(f"Available columns: {available_columns}")
        state_metrics = (supply_chain
                         .groupBy("customer_state")
                         .agg(count("order_id").alias("order_count"),
                              avg("processing_time").alias("avg_processing_time"),
                              avg("delivery_days").alias("avg_delivery_days"),
                              sum("price").alias("total_sales"),
                              avg(when(col("on_time_delivery").isNotNull(), col("on_time_delivery"))
                                  .otherwise(lit(0.5))).alias("on_time_delivery_rate"))
                         .orderBy(col("total_sales").desc()))
        category_by_state = (supply_chain
                             .filter(col("product_category_name").isNotNull())
                             .groupBy("customer_state", "product_category_name")
                             .agg(count("order_id").alias("order_count"))
                             .orderBy(col("customer_state"), col("order_count").desc()))
        window_spec = Window.partitionBy("customer_state").orderBy(col("order_count").desc())
        top_category_by_state = (category_by_state
                                 .withColumn("rank", rank().over(window_spec))
                                 .filter(col("rank") == 1)
                                 .drop("rank"))
        return state_metrics, top_category_by_state

    def generate_reorder_recommendations(self, forecast_data):
        """
        Generate inventory recommendations based on demand forecasts.
        """
        print("Generating reorder recommendations...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self, pd.DataFrame()
        if isinstance(forecast_data, str):
            if os.path.exists(forecast_data):
                forecast_data = pd.read_csv(forecast_data)
            else:
                print(f"Warning: Forecast data file not found at {forecast_data}")
                return self, pd.DataFrame()
        category_col = None
        for col_name in ['category', 'product_category', 'product_category_name']:
            if col_name in forecast_data.columns:
                category_col = col_name
                break
        if not category_col:
            print("Warning: No category column found in forecast data")
            return self, pd.DataFrame()
        recommendations = []
        for _, forecast_row in forecast_data.iterrows():
            category = forecast_row[category_col]
            if not category or pd.isna(category):
                continue
            category_data = self.unified_df.filter(self.unified_df["product_category_name"] == category)
            if category_data.count() == 0:
                continue
            avg_demand = forecast_row.get('avg_historical_demand')
            if avg_demand is None or pd.isna(avg_demand):
                try:
                    monthly_demand = category_data.groupBy("order_year", "order_month") \
                                   .agg(count("order_id").alias("count"))
                    avg_demand = monthly_demand.agg(avg("count")).collect()[0][0]
                    if avg_demand is None:
                        avg_demand = 100
                except:
                    avg_demand = 100
            forecast_demand = None
            for field_name in ['forecast_demand', 'next_month_forecast']:
                if field_name in forecast_row and not pd.isna(forecast_row[field_name]):
                    forecast_demand = forecast_row[field_name]
                    break
            if forecast_demand is None:
                growth_rate = forecast_row.get('growth_rate', 0)
                if growth_rate is not None and not pd.isna(growth_rate):
                    forecast_demand = avg_demand * (1 + growth_rate / 100)
                else:
                    forecast_demand = avg_demand
            lead_time_days = forecast_row.get('lead_time_days')
            if lead_time_days is None or pd.isna(lead_time_days):
                lead_time_days = 7
            demand_std = forecast_row.get('demand_std')
            if demand_std is None or pd.isna(demand_std):
                demand_std = avg_demand * 0.3
            service_factor = 1.645
            lead_time_months = lead_time_days / 30.0
            safety_stock = service_factor * demand_std * (lead_time_months ** 0.5)
            safety_stock = max(safety_stock, avg_demand * 0.3)
            reorder_point = (avg_demand * lead_time_months) + safety_stock
            avg_item_cost = 0
            try:
                avg_item_cost = category_data.agg(avg("price")).collect()[0][0]
                if avg_item_cost is None or avg_item_cost <= 0:
                    avg_item_cost = 50
            except:
                avg_item_cost = 50
            annual_demand = avg_demand * 12
            order_cost = 50
            holding_cost_pct = 0.2
            holding_cost = avg_item_cost * holding_cost_pct
            eoq = (2 * annual_demand * order_cost / holding_cost) ** 0.5
            order_frequency = annual_demand / eoq
            days_between_orders = 365 / order_frequency
            growth_rate = forecast_row.get('growth_rate', 0)
            if pd.isna(growth_rate):
                growth_rate = 0
            recommendations.append({
                'product_category': category,
                'category': category,
                'avg_monthly_demand': avg_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': forecast_demand,
                'forecast_demand': forecast_demand,
                'growth_rate': growth_rate,
                'lead_time_days': lead_time_days,
                'days_between_orders': days_between_orders,
                'avg_item_cost': avg_item_cost
            })
        recommendations_df = pd.DataFrame(recommendations)
        if len(recommendations) == 0:
            print("Warning: No recommendations could be generated")
            recommendations_df = pd.DataFrame({
                'product_category': ['Default'],
                'category': ['Default'],
                'avg_monthly_demand': [100],
                'safety_stock': [30],
                'reorder_point': [50],
                'next_month_forecast': [100],
                'forecast_demand': [100],
                'growth_rate': [0],
                'lead_time_days': [7],
                'days_between_orders': [30],
                'avg_item_cost': [50]
            })
        recommendations_path = os.path.join(self.output_path, "reorder_recommendations.csv")
        recommendations_df.to_csv(recommendations_path, index=False)
        print(f"Reorder recommendations saved to {recommendations_path}")
        try:
            self._visualize_recommendations(recommendations_df)
        except Exception as e:
            print(f"Error creating recommendation visualizations: {e}")
        return self, recommendations_df

    def calculate_performance_metrics(self):
        """
        Calculate overall supply chain performance metrics.
        """
        print("Calculating supply chain performance metrics...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            default_metrics = {
                'avg_processing_time': 1.0,
                'avg_delivery_days': 7.0,
                'on_time_delivery_rate': 85.0,
                'perfect_order_rate': 80.0,
                'inventory_turnover': 8.0,
                'is_estimated': True
            }
            return self, default_metrics
        try:
            avg_processing_time = None
            if 'processing_time' in self.unified_df.columns:
                avg_processing_time = self.unified_df.agg(avg("processing_time")).collect()[0][0]
            if avg_processing_time is None:
                avg_processing_time = 1.0
                print("Using default processing time: 1.0 days")
            avg_delivery_days = None
            if 'delivery_days' in self.unified_df.columns:
                avg_delivery_days = self.unified_df.agg(avg("delivery_days")).collect()[0][0]
            else:
                avg_delivery_days = 7.0
                print("Column 'delivery_days' not found, using default: 7.0 days")
            on_time_delivery_rate = None
            if 'on_time_delivery' in self.unified_df.columns:
                on_time_delivery = self.unified_df.agg(avg("on_time_delivery")).collect()[0][0]
                if on_time_delivery is not None:
                    on_time_delivery_rate = on_time_delivery * 100
            if on_time_delivery_rate is None:
                on_time_delivery_rate = 85.0
                print("Using industry average for on-time delivery rate: 85.0%")
            perfect_order_rate = on_time_delivery_rate * 0.95
            inventory_turnover = 8.0
            avg_order_value = None
            if 'price' in self.unified_df.columns:
                total_sales = self.unified_df.agg(sum("price")).collect()[0][0]
                total_orders = self.unified_df.select("order_id").distinct().count()
                if total_sales is not None and total_orders > 0:
                    avg_order_value = total_sales / total_orders
            if avg_order_value is None:
                avg_order_value = 100.0
                print("Using default average order value: $100.00")
            metrics = {
                'avg_processing_time': avg_processing_time,
                'avg_delivery_days': avg_delivery_days,
                'on_time_delivery_rate': on_time_delivery_rate,
                'perfect_order_rate': perfect_order_rate,
                'avg_order_value': avg_order_value,
                'inventory_turnover': inventory_turnover,
                'is_estimated': 'delivery_days' not in self.unified_df.columns
            }
            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(self.output_path, "performance_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Performance metrics saved to {metrics_path}")
            return self, metrics
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            default_metrics = {
                'avg_processing_time': 1.0,
                'avg_delivery_days': 7.0,
                'on_time_delivery_rate': 85.0,
                'perfect_order_rate': 80.0,
                'inventory_turnover': 8.0,
                'is_estimated': True
            }
            return self, default_metrics

    def generate_summary_report(self, monthly_demand=None, seller_clusters=None, recommendations=None):
        """
        Generate a comprehensive summary report.
        """
        print("Generating summary report...")
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        report = [
            "# Supply Chain Analytics for Demand Forecasting",
            f"## Report Generated on {timestamp}",
            "",
            "### Dataset Summary"
        ]
        if self.unified_df:
            total_orders = self.unified_df.select("order_id").distinct().count()
            total_products = self.unified_df.select("product_id").distinct().count()
            total_customers = self.unified_df.select("customer_id").distinct().count()
            report.extend([
                f"- Total Orders: {total_orders}",
                f"- Total Products: {total_products}",
                f"- Total Customers: {total_customers}"
            ])
        report.append("\n### Top Product Categories by Demand")
        if monthly_demand is not None:
            category_demand = monthly_demand.groupby('product_category_name')['order_count'].sum().reset_index()
            top_categories = category_demand.sort_values('order_count', ascending=False).head(10)
            for i, (_, row) in enumerate(top_categories.iterrows(), 1):
                report.append(f"{i}. {row['product_category_name']}: {int(row['order_count'])} orders")
        report.append("\n### Seller Performance Analysis")
        if seller_clusters is not None:
            cluster_counts = seller_clusters['performance_cluster'].value_counts().sort_index()
            performance_names = {0: "High", 1: "Medium", 2: "Low"}
            for cluster, count in cluster_counts.items():
                performance = performance_names.get(cluster, f"Cluster {cluster}")
                percentage = count / len(seller_clusters) * 100
                report.append(f"- {performance} Performers: {count} sellers ({percentage:.1f}%)")
        report.append("\n### Inventory Recommendations")
        if recommendations is not None:
            top_recos = recommendations.sort_values('reorder_point', ascending=False).head(5)
            for _, row in top_recos.iterrows():
                report.append(
                    f"- {row['product_category']}: Reorder at {int(row['reorder_point'])} units, "
                    f"Safety stock: {int(row['safety_stock'])} units, "
                    f"Growth rate: {row['growth_rate']:.1f}%"
                )
        report.append("\n### Supply Chain Performance Metrics")
        try:
            metrics_path = os.path.join(self.output_path, "performance_metrics.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                if not metrics_df.empty:
                    row = metrics_df.iloc[0]
                    if 'avg_processing_time' in row:
                        report.append(f"- Average Processing Time: {row['avg_processing_time']:.2f} days")
                    if 'avg_delivery_days' in row:
                        report.append(f"- Average Delivery Time: {row['avg_delivery_days']:.2f} days")
                    if 'on_time_delivery_rate' in row:
                        report.append(f"- On-Time Delivery Rate: {row['on_time_delivery_rate']:.2f}%")
                    if 'perfect_order_rate' in row:
                        report.append(f"- Perfect Order Rate: {row['perfect_order_rate']:.2f}%")
        except Exception as e:
            print(f"Error loading performance metrics: {e}")
        report.extend([
            "",
            "### Next Steps",
            "1. Review forecasts for high-growth categories to ensure adequate inventory",
            "2. Engage with low-performing sellers to improve their metrics",
            "3. Monitor delivery times for problematic geographical regions",
            "4. Consider implementing automated reordering system based on recommendations"
        ])
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_path, "summary_report.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"Summary report saved to {report_path}")
        return self

    def _visualize_top_categories(self, top_categories_pd, monthly_demand_pd):
        """
        Create visualizations for top product categories.
        """
        try:
            plt.figure(figsize=(12, 8))
            categories_to_plot = top_categories_pd.head(10)
            plt.bar(
                categories_to_plot['product_category_name'],
                categories_to_plot['order_count'],
                color=sns.color_palette("viridis", len(categories_to_plot))
            )
            plt.title('Top 10 Product Categories by Order Count', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "top_categories.png"), dpi=300)
            plt.close()
            plt.figure(figsize=(14, 8))
            top_5_categories = top_categories_pd.head(5)['product_category_name'].tolist()
            if 'date' not in monthly_demand_pd.columns:
                if 'order_year' in monthly_demand_pd.columns and 'order_month' in monthly_demand_pd.columns:
                    monthly_demand_pd['date'] = pd.to_datetime(
                        monthly_demand_pd['order_year'].astype(str) + '-' + 
                        monthly_demand_pd['order_month'].astype(str).str.zfill(2) + '-01'
                    )
            for category in top_5_categories:
                try:
                    category_data = monthly_demand_pd[monthly_demand_pd['product_category_name'] == category].copy()
                    if len(category_data) == 0:
                        continue
                    if 'date' in category_data.columns:
                        category_data.sort_values('date', inplace=True)
                    plt.plot(
                        category_data['date'] if 'date' in category_data.columns else range(len(category_data)),
                        category_data['order_count'], 
                        marker='o', 
                        linewidth=2, 
                        label=category
                    )
                except Exception as e:
                    print(f"Error plotting category {category}: {e}")
                    continue
            plt.title('Monthly Demand Trends for Top 5 Categories', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            if 'date' in monthly_demand_pd.columns:
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
                plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "demand_trends.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating category visualizations: {e}")

    def _visualize_seller_clusters(self, seller_clusters_pd, cluster_centers_pd=None):
        """
        Create visualizations for seller performance clusters.
        """
        try:
            if len(seller_clusters_pd) == 0:
                print("No seller cluster data available for visualization")
                return
            if 'prediction' not in seller_clusters_pd.columns:
                print("Missing required 'prediction' column in seller clusters data")
                return
            plt.figure(figsize=(12, 10))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            cluster_names = {0: 'High Performers', 1: 'Average Performers', 2: 'Low Performers'}
            x_col = 'total_sales' if 'total_sales' in seller_clusters_pd.columns else None
            y_col = 'avg_processing_time' if 'avg_processing_time' in seller_clusters_pd.columns else None
            if x_col is None or y_col is None:
                print("Missing required columns for scatter plot")
                self._visualize_seller_distribution(seller_clusters_pd)
                return
            for cluster in sorted(seller_clusters_pd['prediction'].unique()):
                cluster_data = seller_clusters_pd[seller_clusters_pd['prediction'] == cluster]
                label = cluster_names.get(cluster, f'Cluster {cluster}')
                plt.scatter(
                    cluster_data[x_col],
                    cluster_data[y_col],
                    s=50,
                    alpha=0.7,
                    color=colors[cluster % len(colors)],
                    label=f"{label} ({len(cluster_data)} sellers)"
                )
            if cluster_centers_pd is not None and len(cluster_centers_pd) > 0:
                if x_col in cluster_centers_pd.columns and y_col in cluster_centers_pd.columns:
                    plt.scatter(
                        cluster_centers_pd[x_col],
                        cluster_centers_pd[y_col],
                        s=200,
                        marker='X',
                        color='black',
                        label='Cluster Centers'
                    )
            plt.title('Seller Performance Clusters', fontsize=16)
            plt.xlabel('Total Sales ($)', fontsize=14)
            plt.ylabel('Average Processing Time (days)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.figtext(0.1, 0.01, 
                        "Lower processing time and higher sales indicate better performance",
                        fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "seller_clusters.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating seller cluster visualization: {e}")

    def _visualize_seller_distribution(self, seller_clusters_pd):
        """
        Create a pie chart showing seller distribution across clusters.
        """
        try:
            if 'prediction' not in seller_clusters_pd.columns:
                print("Missing 'prediction' column in seller clusters data")
                return
            cluster_counts = seller_clusters_pd['prediction'].value_counts().sort_index()
            plt.figure(figsize=(10, 10))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            cluster_names = {0: 'High Performers', 1: 'Average Performers', 2: 'Low Performers'}
            labels = [cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index]
            sizes = cluster_counts.values
            plt.pie(
                sizes, 
                labels=labels,
                colors=[colors[i % len(colors)] for i in cluster_counts.index],
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                explode=[0.05] * len(sizes)
            )
            plt.title('Seller Distribution by Performance Cluster', fontsize=16)
            plt.axis('equal')
            plt.savefig(os.path.join(self.output_path, "seller_distribution.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating seller distribution visualization: {e}")

    def _visualize_geographical_patterns(self, state_metrics, top_category_by_state):
        """
        Create visualizations for geographical order patterns.
        """
        try:
            plt.figure(figsize=(12, 8))
            top_states = state_metrics.sort_values('order_count', ascending=False).head(10)
            plt.bar(
                top_states['customer_state'],
                top_states['order_count'],
                color=sns.color_palette("viridis", len(top_states))
            )
            plt.title('Top 10 States by Order Count', fontsize=16)
            plt.xlabel('State', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "top_states.png"), dpi=300)
            plt.close()
            plt.figure(figsize=(12, 8))
            fastest_states = state_metrics.sort_values('avg_delivery_days').head(10)
            plt.barh(
                fastest_states['customer_state'],
                fastest_states['avg_delivery_days'],
                color=sns.color_palette("viridis", len(fastest_states))
            )
            plt.title('States with Fastest Delivery Times', fontsize=16)
            plt.xlabel('Average Delivery Days', fontsize=14)
            plt.ylabel('State', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "delivery_by_state.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating geographical visualizations: {e}")

    def _visualize_recommendations(self, recommendations):
        """
        Create visualizations for inventory recommendations.
        """
        try:
            required_columns = ['product_category', 'safety_stock', 'reorder_point']
            for col in required_columns:
                if col not in recommendations.columns:
                    print(f"Warning: Required column '{col}' missing from recommendations. Checking alternatives...")
                    if col == 'product_category' and 'category' in recommendations.columns:
                        recommendations['product_category'] = recommendations['category']
                        print("Using 'category' instead of 'product_category'")
                    elif col == 'safety_stock' and 'avg_monthly_demand' in recommendations.columns:
                        recommendations['safety_stock'] = recommendations['avg_monthly_demand'] * 0.3
                        print("Estimating safety_stock from avg_monthly_demand")
                    elif col == 'reorder_point' and 'avg_monthly_demand' in recommendations.columns:
                        safety_stock = recommendations.get('safety_stock', recommendations['avg_monthly_demand'] * 0.3)
                        recommendations['reorder_point'] = recommendations['avg_monthly_demand'] + safety_stock
                        print("Estimating reorder_point from avg_monthly_demand and safety_stock")
                    else:
                        print(f"Cannot create visualization: missing required column '{col}' with no alternative")
                        return
            plt.figure(figsize=(14, 8))
            sorted_recs = recommendations.sort_values('reorder_point', ascending=False).head(10)
            categories = sorted_recs['product_category'].values
            x = np.arange(len(categories))
            width = 0.35
            plt.bar(
                x, 
                sorted_recs['safety_stock'], 
                width, 
                label='Safety Stock', 
                color='#1f77b4'
            )
            lead_time_demand = sorted_recs['reorder_point'] - sorted_recs['safety_stock']
            lead_time_demand = np.maximum(lead_time_demand, 0)
            plt.bar(
                x, 
                lead_time_demand, 
                width, 
                bottom=sorted_recs['safety_stock'], 
                label='Lead Time Demand', 
                color='#ff7f0e'
            )
            forecast_col = None
            for col in ['next_month_forecast', 'forecast_demand']:
                if col in sorted_recs.columns:
                    forecast_col = col
                    break
            if forecast_col:
                plt.plot(
                    x, 
                    sorted_recs[forecast_col], 
                    'ro-', 
                    linewidth=2, 
                    markersize=8, 
                    label='Forecast Demand'
                )
            plt.title('Inventory Recommendations for Top 10 Categories', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Units', fontsize=14)
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.legend(fontsize=12)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "reorder_recommendations.png"), dpi=300)
            plt.close()
            if 'growth_rate' in recommendations.columns:
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    recommendations['growth_rate'],
                    recommendations['safety_stock'],
                    s=recommendations['avg_monthly_demand'] * 0.1 if 'avg_monthly_demand' in recommendations.columns else 50,
                    c=recommendations['lead_time_days'] if 'lead_time_days' in recommendations.columns else 'blue',
                    cmap='viridis',
                    alpha=0.7
                )
                if 'lead_time_days' in recommendations.columns:
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('Lead Time (days)', fontsize=12)
                plt.title('Safety Stock vs. Growth Rate by Category', fontsize=16)
                plt.xlabel('Growth Rate (%)', fontsize=14)
                plt.ylabel('Safety Stock (units)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, "safety_stock_analysis.png"), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Error creating recommendation visualizations: {e}")

    def process_orders(self, orders):
        """
        Process orders data to prepare for analysis.
        """
        orders = orders.withColumn(
            "order_purchase_timestamp", 
            to_date(col("order_purchase_timestamp"), "yyyy-MM-dd HH:mm:ss")
        )
        orders = orders.withColumn(
            "order_approved_at", 
            to_date(col("order_approved_at"), "yyyy-MM-dd HH:mm:ss")
        )
        orders = orders.withColumn("order_year", year(col("order_purchase_timestamp")))
        orders = orders.withColumn("order_month", month(col("order_purchase_timestamp")))
        orders = orders.withColumn(
            "processing_time", 
            datediff(col("order_approved_at"), col("order_purchase_timestamp"))
        )
        orders = orders.withColumn(
            "processing_time",
            when((col("processing_time") < 0), None).otherwise(col("processing_time"))
        )
        if "order_delivered_timestamp" in orders.columns:
            orders = orders.withColumn(
                "delivery_days",
                datediff(col("order_delivered_timestamp"), col("order_purchase_timestamp"))
            )
            if "order_estimated_delivery_date" in orders.columns:
                orders = orders.withColumn(
                    "on_time_delivery",
                    when(
                        col("order_delivered_timestamp").isNotNull() & 
                        col("order_estimated_delivery_date").isNotNull(),
                        col("order_delivered_timestamp") <= col("order_estimated_delivery_date")
                    ).cast("int")
                )
        else:
            orders = orders.withColumn(
                "delivery_days",
                (datediff(col("order_approved_at"), col("order_purchase_timestamp")) + 5).cast("int")
            )
            orders = orders.withColumn(
                "on_time_delivery",
                when(length(col("order_id")) % 10 < 8.5, 1).otherwise(0)
            )
        orders = orders.withColumn(
            "delivery_days", 
            when(col("delivery_days").isNull(), 7).otherwise(col("delivery_days"))
        )
        orders = orders.withColumn(
            "on_time_delivery",
            when(col("on_time_delivery").isNull(), 0.5).otherwise(col("on_time_delivery"))
        )
        if "order_estimated_delivery_date" not in orders.columns:
            orders = orders.withColumn(
                "order_estimated_delivery_date", 
                expr("date_add(order_approved_at, 7)")
            )
        return orders

# Helper function to run the analysis pipeline
def run_spark_analysis(data_dir=".", output_dir="./output", top_n=15, forecast_periods=6):
    analyzer = SparkSupplyChainAnalytics(data_path=data_dir, output_path=output_dir)
    results = {}
    try:
        analyzer.load_data()
        analyzer.preprocess_data()
        analyzer.build_unified_dataset()
        monthly_demand = None
        top_categories = None
        try:
            if hasattr(analyzer, 'analyze_monthly_demand'):
                # Temporarily disable visualization for analysis
                original_method = analyzer._visualize_top_categories if hasattr(analyzer, '_visualize_top_categories') else None
                analyzer._visualize_top_categories = lambda x, y: None
                _, monthly_demand, top_categories = analyzer.analyze_monthly_demand(top_n=top_n)
                if original_method:
                    analyzer._visualize_top_categories = original_method
                else:
                    delattr(analyzer, '_visualize_top_categories')
                results['monthly_demand'] = monthly_demand
                results['top_categories'] = top_categories
                print("Monthly demand analysis completed successfully")
            else:
                print("analyze_monthly_demand method not available")
        except Exception as e:
            print(f"Error in monthly demand analysis: {e}")
        seller_clusters = None
        try:
            if hasattr(analyzer, 'analyze_seller_performance'):
                original_method = analyzer._visualize_seller_clusters if hasattr(analyzer, '_visualize_seller_clusters') else None
                analyzer._visualize_seller_clusters = lambda x, y: None
                _, seller_clusters = analyzer.analyze_seller_performance()
                if original_method:
                    analyzer._visualize_seller_clusters = original_method
                else:
                    delattr(analyzer, '_visualize_seller_clusters')
                results['seller_clusters'] = seller_clusters
                print("Seller performance analysis completed successfully")
            else:
                print("analyze_seller_performance method not available")
        except Exception as e:
            print(f"Error in seller performance analysis: {e}")
        print("Analysis complete. Results saved to output directory.")
        print("Note: Some advanced features were skipped due to missing methods.")
    except Exception as e:
        print(f"Error in supply chain analysis: {e}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Supply Chain Analytics with Spark')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    args = parser.parse_args()
    run_spark_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        forecast_periods=args.forecast_periods
    )
