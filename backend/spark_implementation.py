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
    Includes defensive checks for missing columns and uses checkpointing
    to manage memory usage with large datasets.
    """
    def __init__(self, spark=None, data_path=".", output_path="./output"):
        self.data_path = data_path
        self.output_path = output_path
        self.spark = spark if spark else self._create_spark_session()
        os.makedirs(output_path, exist_ok=True)
        # Set checkpoint directory for Spark to break lineage chains
        self.spark.sparkContext.setCheckpointDir(os.path.join(self.output_path, "checkpoints"))
        
        # Data containers
        self.orders_df = None
        self.order_items_df = None
        self.products_df = None
        self.customers_df = None
        self.unified_df = None

    def _create_spark_session(self):
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
        print("Loading data using Spark...")
        # Load orders data
        orders_path = os.path.join(self.data_path, "df_Orders.csv")
        if os.path.exists(orders_path):
            self.orders_df = self.spark.read.csv(orders_path, header=True, inferSchema=True)
            print(f"Loaded orders data: {self.orders_df.count()} rows")
        else:
            print(f"Warning: Orders data file not found at {orders_path}")
            self.orders_df = self.spark.createDataFrame([], "order_id STRING, customer_id STRING, order_status STRING, " +
                                                        "order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP, " +
                                                        "order_delivered_timestamp TIMESTAMP, order_estimated_delivery_date TIMESTAMP")
        # Load order items data
        let_order_items_path = os.path.join(self.data_path, "df_OrderItems.csv")
        if os.path.exists(let_order_items_path):
            self.order_items_df = self.spark.read.csv(let_order_items_path, header=True, inferSchema=True)
            print(f"Loaded order items data: {self.order_items_df.count()} rows")
        else:
            print(f"Warning: Order items data file not found at {let_order_items_path}")
            self.order_items_df = self.spark.createDataFrame([], "order_id STRING, order_item_id INT, product_id STRING, " +
                                                           "seller_id STRING, price DOUBLE, shipping_charges DOUBLE")
        # Load products data
        products_path = os.path.join(self.data_path, "df_Products.csv")
        if os.path.exists(products_path):
            self.products_df = self.spark.read.csv(products_path, header=True, inferSchema=True)
            print(f"Loaded products data: {self.products_df.count()} rows")
        else:
            print(f"Warning: Products data file not found at {products_path}")
            self.products_df = self.spark.createDataFrame([], "product_id STRING, product_category_name STRING, " + 
                                                        "product_weight_g DOUBLE, product_length_cm DOUBLE, " +
                                                        "product_height_cm DOUBLE, product_width_cm DOUBLE")
        # Load customers data
        customers_path = os.path.join(self.data_path, "df_Customers.csv")
        if os.path.exists(customers_path):
            self.customers_df = self.spark.read.csv(customers_path, header=True, inferSchema=True)
            print(f"Loaded customers data: {self.customers_df.count()} rows")
        else:
            print(f"Warning: Customers data file not found at {customers_path}")
            self.customers_df = self.spark.createDataFrame([], "customer_id STRING, customer_zip_code_prefix STRING, " +
                                                         "customer_city STRING, customer_state STRING")
        return self

    def preprocess_data(self):
        print("Preprocessing data...")
        # Preprocess orders data with defensive column checks
        if self.orders_df:
            for col_name in ["order_purchase_timestamp", "order_approved_at", 
                             "order_delivered_timestamp", "order_estimated_delivery_date"]:
                if col_name in self.orders_df.columns:
                    self.orders_df = self.orders_df.withColumn(col_name, to_date(col(col_name)))
            if "order_purchase_timestamp" in self.orders_df.columns:
                self.orders_df = self.orders_df.withColumn("order_year", year(col("order_purchase_timestamp"))) \
                                               .withColumn("order_month", month(col("order_purchase_timestamp")))
            if all(c in self.orders_df.columns for c in ["order_purchase_timestamp", "order_approved_at"]):
                self.orders_df = self.orders_df.withColumn("processing_time",
                                                             datediff(col("order_approved_at"), col("order_purchase_timestamp")))
                self.orders_df = self.orders_df.withColumn("processing_time",
                                                             when(col("processing_time") < 0, None).otherwise(col("processing_time")))
            if all(c in self.orders_df.columns for c in ["order_purchase_timestamp", "order_delivered_timestamp"]):
                self.orders_df = self.orders_df.withColumn("delivery_days",
                                                             datediff(col("order_delivered_timestamp"), col("order_purchase_timestamp")))
                self.orders_df = self.orders_df.withColumn("delivery_days",
                                                             when(col("delivery_days") < 0, None).otherwise(col("delivery_days")))
            if all(c in self.orders_df.columns for c in ["order_delivered_timestamp", "order_estimated_delivery_date"]):
                self.orders_df = self.orders_df.withColumn(
                    "on_time_delivery",
                    when(col("order_delivered_timestamp").isNotNull() & col("order_estimated_delivery_date").isNotNull(),
                         when(col("order_delivered_timestamp") <= col("order_estimated_delivery_date"), 1).otherwise(0)
                    ).otherwise(None)
                )
        # Preprocess products data with check for column existence
        if self.products_df:
            for num_col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
                if num_col in self.products_df.columns:
                    category_medians = self.products_df.groupBy("product_category_name") \
                                          .agg(percentile_approx(col(num_col), 0.5).alias(f"{num_col}_median"))
                    self.products_df = self.products_df.join(category_medians, on="product_category_name", how="left")
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(col(num_col).isNull(), col(f"{num_col}_median")).otherwise(col(num_col))
                    )
                    global_median = self.products_df.select(percentile_approx(col(num_col), 0.5)).collect()[0][0]
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(col(num_col).isNull(), lit(global_median)).otherwise(col(num_col))
                    )
                    self.products_df = self.products_df.drop(f"{num_col}_median")
        # Preprocess order items data with fallback defaults
        if self.order_items_df:
            for value_col in ["price", "shipping_charges"]:
                if value_col in self.order_items_df.columns:
                    product_medians = self.order_items_df.groupBy("product_id") \
                                          .agg(percentile_approx(col(value_col), 0.5).alias(`${value_col}_median`))
                    self.order_items_df = self.order_items_df.join(product_medians, on="product_id", how="left")
                    self.order_items_df = self.order_items_df.withColumn(
                        value_col,
                        when(col(value_col).isNull(), col(`${value_col}_median`)).otherwise(col(value_col))
                    )
                    global_median = self.order_items_df.select(percentile_approx(col(value_col), 0.5)).collect()[0][0]
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
        print("Building unified dataset...")
        required_dfs = [self.orders_df, self.order_items_df, self.products_df, self.customers_df]
        if not all(required_dfs):
            print("Warning: Not all required datasets are loaded")
            return self
        # Check required join columns; log warnings if missing.
        for col in ["order_id"]:
            if col not in self.orders_df.columns:
                print(f"Warning: '{col}' column missing from orders data")
        unified_df = self.orders_df.join(self.order_items_df, on="order_id", how="inner")
        unified_df = unified_df.join(self.products_df, on="product_id", how="left")
        unified_df = unified_df.join(self.customers_df, on="customer_id", how="left")
        # Cache and checkpoint to manage memory usage.
        self.unified_df = unified_df.persist().checkpoint()
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
        print("Analyzing monthly demand patterns...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self, None, None
        # Check for required columns; if missing, log warning.
        for col in ["product_category_name", "order_year", "order_month"]:
            if col not in self.unified_df.columns:
                print(f"Warning: Column '{col}' is missing from unified dataset.")
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
        print("Analyzing geographical patterns...")
        available_columns = supply_chain.columns
        print(f"Available columns: {available_columns}")
        state_metrics = (supply_chain
                    .groupBy("customer_state")
                    .agg(count("order_id").alias("order_count"),
                         avg("processing_time").alias("avg_processing_time"),
                         avg("delivery_days").alias("avg_delivery_days"),
                         sum("price").alias("total_sales"),
                         avg(when(col("on_time_delivery").isNotNull(), col("on_time_delivery")).otherwise(lit(0.5))).alias("on_time_delivery_rate"))
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
                try {
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
                const_growth = forecast_row.get('growth_rate', 0)
                if const_growth is not None and not pd.isna(const_growth):
                    forecast_demand = avg_demand * (1 + const_growth / 100)
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
            try {
                avg_item_cost = category_data.agg(avg("price")).collect()[0][0]
                if avg_item_cost is None or avg_item_cost <= 0:
                    avg_item_cost = 50
            } catch:
                avg_item_cost = 50
            annual_demand = avg_demand * 12
            order_cost = 50
            holding_cost_pct = 0.2
            holding_cost = avg_item_cost * holding_cost_pct
            eoq = (2 * annual_demand * order_cost / holding_cost) ** 0.5
            order_frequency = annual_demand / eoq
            days_between_orders = 365 / order_frequency
            const_growth = forecast_row.get('growth_rate', 0)
            if pd.isna(const_growth):
                const_growth = 0
            recommendations.push({
                'product_category': category,
                'category': category,
                'avg_monthly_demand': avg_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': forecast_demand,
                'forecast_demand': forecast_demand,
                'growth_rate': const_growth,
                'lead_time_days': lead_time_days,
                'days_between_orders': days_between_orders,
                'avg_item_cost': avg_item_cost
            });
        recommendations_df = pd.DataFrame(recommendations)
        if len(recommendations) === 0:
            print("Warning: No recommendations could be generated");
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
            });
        const recommendations_path = os.path.join(self.output_path, "reorder_recommendations.csv")
        recommendations_df.to_csv(recommendations_path, index=False)
        print(`Reorder recommendations saved to ${recommendations_path}`)
        try:
            self._visualize_recommendations(recommendations_df)
        except Exception as e:
            print(`Error creating recommendation visualizations: ${e}`)
        return self, recommendations_df

    def calculate_performance_metrics(self):
        print("Calculating supply chain performance metrics...")
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            const default_metrics = {
                'avg_processing_time': 1.0,
                'avg_delivery_days': 7.0,
                'on_time_delivery_rate': 85.0,
                'perfect_order_rate': 80.0,
                'inventory_turnover': 8.0,
                'is_estimated': True
            };
            return self, default_metrics;
        try {
            let avg_processing_time = null;
            if ('processing_time' in self.unified_df.columns) {
                avg_processing_time = self.unified_df.agg(avg("processing_time")).collect()[0][0];
            }
            if (avg_processing_time === null) {
                avg_processing_time = 1.0;
                print("Using default processing time: 1.0 days");
            }
            let avg_delivery_days = null;
            if ('delivery_days' in self.unified_df.columns) {
                avg_delivery_days = self.unified_df.agg(avg("delivery_days")).collect()[0][0];
            } else {
                avg_delivery_days = 7.0;
                print("Column 'delivery_days' not found, using default: 7.0 days");
            }
            let on_time_delivery_rate = null;
            if ('on_time_delivery' in self.unified_df.columns) {
                const on_time_delivery = self.unified_df.agg(avg("on_time_delivery")).collect()[0][0];
                if (on_time_delivery !== null) {
                    on_time_delivery_rate = on_time_delivery * 100;
                }
            }
            if (on_time_delivery_rate === null) {
                on_time_delivery_rate = 85.0;
                print("Using industry average for on-time delivery rate: 85.0%");
            }
            const perfect_order_rate = on_time_delivery_rate * 0.95;
            const inventory_turnover = 8.0;
            let avg_order_value = null;
            if ('price' in self.unified_df.columns) {
                const total_sales = self.unified_df.agg(sum("price")).collect()[0][0];
                const total_orders = self.unified_df.select("order_id").distinct().count();
                if (total_sales !== null && total_orders > 0) {
                    avg_order_value = total_sales / total_orders;
                }
            }
            if (avg_order_value === null) {
                avg_order_value = 100.0;
                print("Using default average order value: $100.00");
            }
            const metrics = {
                'avg_processing_time': avg_processing_time,
                'avg_delivery_days': avg_delivery_days,
                'on_time_delivery_rate': on_time_delivery_rate,
                'perfect_order_rate': perfect_order_rate,
                'avg_order_value': avg_order_value,
                'inventory_turnover': inventory_turnover,
                'is_estimated': !('delivery_days' in self.unified_df.columns)
            };
            const metrics_df = pd.DataFrame([metrics]);
            const metrics_path = os.path.join(self.output_path, "performance_metrics.csv");
            metrics_df.to_csv(metrics_path, index=False);
            print(`Performance metrics saved to ${metrics_path}`);
            return self, metrics;
        } catch (e) {
            print(`Error calculating performance metrics: ${e}`);
            const default_metrics = {
                'avg_processing_time': 1.0,
                'avg_delivery_days': 7.0,
                'on_time_delivery_rate': 85.0,
                'perfect_order_rate': 80.0,
                'inventory_turnover': 8.0,
                'is_estimated': True
            };
            return self, default_metrics;
        }
    }

    def generate_summary_report(self, monthly_demand=null, seller_clusters=null, recommendations=null):
        print("Generating summary report...");
        const timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S");
        const report = [
            "# Supply Chain Analytics for Demand Forecasting",
            `## Report Generated on ${timestamp}`,
            "",
            "### Dataset Summary"
        ];
        if (self.unified_df) {
            const total_orders = self.unified_df.select("order_id").distinct().count();
            const total_products = self.unified_df.select("product_id").distinct().count();
            const total_customers = self.unified_df.select("customer_id").distinct().count();
            report.push(`- Total Orders: ${total_orders}`);
            report.push(`- Total Products: ${total_products}`);
            report.push(`- Total Customers: ${total_customers}`);
        }
        report.push("\n### Top Product Categories by Demand");
        if (monthly_demand !== null) {
            const category_demand = monthly_demand.groupby('product_category_name')['order_count'].sum().reset_index();
            const top_categories = category_demand.sort_values('order_count', ascending=false).head(10);
            top_categories.iterrows().forEach((_, row, i) => {
                report.push(`${i+1}. ${row['product_category_name']}: ${parseInt(row['order_count'])} orders`);
            });
        }
        report.push("\n### Seller Performance Analysis");
        if (seller_clusters !== null) {
            const cluster_counts = seller_clusters['performance_cluster'].value_counts().sort_index();
            const performance_names = {0: "High", 1: "Medium", 2: "Low"};
            Object.keys(cluster_counts).forEach(cluster => {
                const count = cluster_counts[cluster];
                const performance = performance_names[cluster] || `Cluster ${cluster}`;
                const percentage = (count / seller_clusters.length) * 100;
                report.push(`- ${performance} Performers: ${count} sellers (${percentage.toFixed(1)}%)`);
            });
        }
        report.push("\n### Inventory Recommendations");
        if (recommendations !== null) {
            const top_recos = recommendations.sort_values('reorder_point', ascending=false).head(5);
            top_recos.iterrows().forEach(row => {
                report.push(`- ${row['product_category']}: Reorder at ${parseInt(row['reorder_point'])} units, Safety stock: ${parseInt(row['safety_stock'])} units, Growth rate: ${parseFloat(row['growth_rate']).toFixed(1)}%`);
            });
        }
        report.push("\n### Supply Chain Performance Metrics");
        try {
            const metrics_path = os.path.join(self.output_path, "performance_metrics.csv");
            if (os.path.exists(metrics_path)) {
                const metrics_df = pd.read_csv(metrics_path);
                if (!metrics_df.empty) {
                    const row = metrics_df.iloc[0];
                    if ('avg_processing_time' in row) {
                        report.push(`- Average Processing Time: ${row['avg_processing_time'].toFixed(2)} days`);
                    }
                    if ('avg_delivery_days' in row) {
                        report.push(`- Average Delivery Time: ${row['avg_delivery_days'].toFixed(2)} days`);
                    }
                    if ('on_time_delivery_rate' in row) {
                        report.push(`- On-Time Delivery Rate: ${row['on_time_delivery_rate'].toFixed(2)}%`);
                    }
                    if ('perfect_order_rate' in row) {
                        report.push(`- Perfect Order Rate: ${row['perfect_order_rate'].toFixed(2)}%`);
                    }
                }
            }
        } catch (e) {
            print(`Error loading performance metrics: ${e}`);
        }
        report.push("");
        report.push("### Next Steps");
        report.push("1. Review forecasts for high-growth categories to ensure adequate inventory");
        report.push("2. Engage with low-performing sellers to improve their metrics");
        report.push("3. Monitor delivery times for problematic geographical regions");
        report.push("4. Consider implementing automated reordering system based on recommendations");
        const report_text = report.join("\n");
        const report_path = os.path.join(self.output_path, "summary_report.md");
        with open(report_path, "w") as f:
            f.write(report_text);
        print(`Summary report saved to ${report_path}`);
        return self;
    }

    // Visualization functions below remain largely unchanged,
    // but include try/catch blocks for consistent error handling.

    def _visualize_top_categories(self, top_categories_pd, monthly_demand_pd):
        try:
            plt.figure(figsize=(12, 8))
            const categories_to_plot = top_categories_pd.head(10);
            plt.bar(
                categories_to_plot['product_category_name'],
                categories_to_plot['order_count'],
                color=sns.color_palette("viridis", len(categories_to_plot))
            );
            plt.title('Top 10 Product Categories by Order Count', fontsize=16);
            plt.xlabel('Product Category', fontsize=14);
            plt.ylabel('Order Count', fontsize=14);
            plt.xticks(rotation=45, ha='right');
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "top_categories.png"), dpi=300);
            plt.close();
            plt.figure(figsize=(14, 8));
            const top_5_categories = top_categories_pd.head(5)['product_category_name'].tolist();
            if (!('date' in monthly_demand_pd.columns) && ('order_year' in monthly_demand_pd.columns && 'order_month' in monthly_demand_pd.columns)) {
                monthly_demand_pd['date'] = pd.to_datetime(
                    monthly_demand_pd['order_year'].astype(str) + '-' + monthly_demand_pd['order_month'].astype(str).str.zfill(2) + '-01'
                );
            }
            for (const category of top_5_categories) {
                try {
                    const category_data = monthly_demand_pd[monthly_demand_pd['product_category_name'] === category].copy();
                    if (category_data.length === 0) continue;
                    if ('date' in category_data.columns) {
                        category_data.sort_values('date', inplace=True);
                    }
                    plt.plot(
                        'date' in category_data.columns ? category_data['date'] : Array.from({length: category_data.length}, (_, i) => i),
                        category_data['order_count'],
                        marker='o',
                        linewidth=2,
                        label=category
                    );
                } catch (e) {
                    print(`Error plotting category ${category}: ${e}`);
                    continue;
                }
            }
            plt.title('Monthly Demand Trends for Top 5 Categories', fontsize=16);
            plt.xlabel('Date', fontsize=14);
            plt.ylabel('Order Count', fontsize=14);
            plt.legend(fontsize=12);
            plt.grid(True, alpha=0.3);
            if ('date' in monthly_demand_pd.columns) {
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'));
                plt.gcf().autofmt_xdate();
            }
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "demand_trends.png"), dpi=300);
            plt.close();
        } except (e) {
            print(`Error creating category visualizations: ${e}`);
        }

    def _visualize_seller_clusters(self, seller_clusters_pd, cluster_centers_pd=None):
        try {
            if (seller_clusters_pd.length === 0) {
                print("No seller cluster data available for visualization");
                return;
            }
            if (!('prediction' in seller_clusters_pd.columns)) {
                print("Missing required 'prediction' column in seller clusters data");
                return;
            }
            plt.figure(figsize=(12, 10));
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
            const cluster_names = {0: 'High Performers', 1: 'Average Performers', 2: 'Low Performers'};
            const x_col = 'total_sales' in seller_clusters_pd.columns ? 'total_sales' : null;
            const y_col = 'avg_processing_time' in seller_clusters_pd.columns ? 'avg_processing_time' : null;
            if (x_col === null || y_col === null) {
                print("Missing required columns for scatter plot");
                self._visualize_seller_distribution(seller_clusters_pd);
                return;
            }
            for (const cluster of Array.from(new Set(seller_clusters_pd['prediction']))) {
                const cluster_data = seller_clusters_pd.filter(row => row['prediction'] === cluster);
                const label = cluster_names[cluster] || `Cluster ${cluster}`;
                plt.scatter(
                    cluster_data.map(row => row[x_col]),
                    cluster_data.map(row => row[y_col]),
                    s=50,
                    alpha=0.7,
                    color=colors[cluster % colors.length],
                    label=`${label} (${cluster_data.length} sellers)`
                );
            }
            if (cluster_centers_pd !== null && cluster_centers_pd.length > 0) {
                if (x_col in cluster_centers_pd.columns && y_col in cluster_centers_pd.columns) {
                    plt.scatter(
                        cluster_centers_pd[x_col],
                        cluster_centers_pd[y_col],
                        s=200,
                        marker='X',
                        color='black',
                        label='Cluster Centers'
                    );
                }
            }
            plt.title('Seller Performance Clusters', fontsize=16);
            plt.xlabel('Normalized Total Sales', fontsize=14);
            plt.ylabel('Normalized Processing Time', fontsize=14);
            plt.legend(fontsize=12);
            plt.grid(True, alpha=0.3);
            plt.figtext(0.01, 0.01, "Lower processing time and higher sales indicate better performance", fontsize=10);
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "seller_clusters.png"), dpi=300);
            plt.close();
        } catch (e) {
            print(`Error creating seller cluster visualization: ${e}`);
        }

    def _visualize_seller_distribution(self, seller_clusters_pd) {
        try {
            if (!('prediction' in seller_clusters_pd.columns)) {
                print("Missing 'prediction' column in seller clusters data");
                return;
            }
            const cluster_counts = {};
            seller_clusters_pd.forEach(row => {
                const cluster = row['prediction'];
                if (cluster !== undefined) {
                    cluster_counts[cluster] = (cluster_counts[cluster] || 0) + 1;
                }
            });
            plt.figure(figsize=(10, 10));
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
            const cluster_names = {0: 'High Performers', 1: 'Average Performers', 2: 'Low Performers'};
            const labels = Object.keys(cluster_counts).map(i => cluster_names[i] || `Cluster ${i}`);
            const sizes = Object.values(cluster_counts);
            plt.pie(
                sizes,
                labels=labels,
                colors=Object.keys(cluster_counts).map(i => colors[i % colors.length]),
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                explode=Array(sizes.length).fill(0.05)
            );
            plt.title('Seller Distribution by Performance Cluster', fontsize=16);
            plt.axis('equal');
            plt.savefig(os.path.join(self.output_path, "seller_distribution.png"), dpi=300);
            plt.close();
        } catch (e) {
            print(`Error creating seller distribution visualization: ${e}`);
        }
    }

    def _visualize_geographical_patterns(self, state_metrics, top_category_by_state) {
        try {
            plt.figure(figsize=(12, 8));
            const top_states = state_metrics.sort_values('order_count', ascending=false).head(10);
            plt.bar(
                top_states['customer_state'],
                top_states['order_count'],
                color=sns.color_palette("viridis", top_states.shape[0])
            );
            plt.title('Top 10 States by Order Count', fontsize=16);
            plt.xlabel('State', fontsize=14);
            plt.ylabel('Order Count', fontsize=14);
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "top_states.png"), dpi=300);
            plt.close();
            plt.figure(figsize=(12, 8));
            const fastest_states = state_metrics.sort_values('avg_delivery_days').head(10);
            plt.barh(
                fastest_states['customer_state'],
                fastest_states['avg_delivery_days'],
                color=sns.color_palette("viridis", fastest_states.shape[0])
            );
            plt.title('States with Fastest Delivery Times', fontsize=16);
            plt.xlabel('Average Delivery Days', fontsize=14);
            plt.ylabel('State', fontsize=14);
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "delivery_by_state.png"), dpi=300);
            plt.close();
        } catch (e) {
            print(`Error creating geographical visualizations: ${e}`);
        }
    }

    def _visualize_recommendations(self, recommendations) {
        try {
            const required_cols = ['product_category', 'reorder_point', 'safety_stock'];
            required_cols.forEach(col => {
                if (!(col in recommendations.columns)) {
                    print(`Warning: Required column '${col}' missing from recommendations. Checking alternatives...`);
                    if (col === 'product_category' && 'category' in recommendations.columns) {
                        recommendations['product_category'] = recommendations['category'];
                        print("Using 'category' instead of 'product_category'");
                    } else if (col === 'safety_stock' && 'avg_monthly_demand' in recommendations.columns) {
                        recommendations['safety_stock'] = recommendations['avg_monthly_demand'] * 0.3;
                        print("Estimating safety_stock from avg_monthly_demand");
                    } else if (col === 'reorder_point' && 'avg_monthly_demand' in recommendations.columns) {
                        const safety_stock = recommendations.get('safety_stock', recommendations['avg_monthly_demand'] * 0.3);
                        recommendations['reorder_point'] = recommendations['avg_monthly_demand'] + safety_stock;
                        print("Estimating reorder_point from avg_monthly_demand and safety_stock");
                    } else {
                        print(`Cannot create visualization: missing required column '${col}' with no alternative`);
                        return;
                    }
            });
            plt.figure(figsize=(14, 8));
            let sorted_recs = recommendations.sort_values('reorder_point', ascending=false).head(10);
            const categories = sorted_recs['product_category'].values;
            const x = Array.from({ length: categories.length }, (_, i) => i);
            const width = 0.35;
            plt.bar(x, sorted_recs['safety_stock'], width, label='Safety Stock', color='#1f77b4');
            let lead_time_demand = sorted_recs['reorder_point'] - sorted_recs['safety_stock'];
            lead_time_demand = lead_time_demand.map(val => Math.max(val, 0));
            plt.bar(x, lead_time_demand, width, { bottom: sorted_recs['safety_stock'], label: 'Lead Time Demand', color: '#ff7f0e' });
            let forecast_col = null;
            for (const col of ['next_month_forecast', 'forecast_demand']) {
                if (col in sorted_recs.columns) {
                    forecast_col = col;
                    break;
                }
            }
            if (forecast_col) {
                plt.plot(x, sorted_recs[forecast_col], 'ro-', { linewidth: 2, markersize: 8, label: 'Forecast Demand' });
            }
            plt.title('Inventory Recommendations for Top 10 Categories', { fontsize: 16 });
            plt.xlabel('Product Category', { fontsize: 14 });
            plt.ylabel('Units', { fontsize: 14 });
            plt.xticks(x, categories, { rotation: 45, ha: 'right' });
            plt.legend({ fontsize: 12 });
            plt.grid(true, { axis: 'y', alpha: 0.3 });
            plt.tight_layout();
            plt.savefig(os.path.join(self.output_path, "reorder_recommendations.png"), { dpi: 300 });
            plt.close();
            if ('growth_rate' in recommendations.columns) {
                plt.figure(figsize=(12, 8));
                const scatter = plt.scatter(
                    recommendations['growth_rate'],
                    recommendations['safety_stock'],
                    { s: ('avg_monthly_demand' in recommendations.columns ? recommendations['avg_monthly_demand'].map(val => val * 0.1) : 50),
                      c: ('lead_time_days' in recommendations.columns ? recommendations['lead_time_days'] : 'blue'),
                      cmap: 'viridis',
                      alpha: 0.7 }
                );
                if ('lead_time_days' in recommendations.columns) {
                    const cbar = plt.colorbar(scatter);
                    cbar.set_label('Lead Time (days)', { fontsize: 12 });
                }
                plt.title('Safety Stock vs. Growth Rate by Category', { fontsize: 16 });
                plt.xlabel('Growth Rate (%)', { fontsize: 14 });
                plt.ylabel('Safety Stock (units)', { fontsize: 14 });
                plt.grid(true, { alpha: 0.3 });
                plt.tight_layout();
                plt.savefig(os.path.join(self.output_path, "safety_stock_analysis.png"), { dpi: 300 });
                plt.close();
            }
        } catch (e) {
            print(`Error creating recommendation visualizations: ${e}`);
        }
    }

    def process_orders(self, orders):
        orders = orders.withColumn("order_purchase_timestamp", to_date(col("order_purchase_timestamp"), "yyyy-MM-dd HH:mm:ss"));
        orders = orders.withColumn("order_approved_at", to_date(col("order_approved_at"), "yyyy-MM-dd HH:mm:ss"));
        orders = orders.withColumn("order_year", year(col("order_purchase_timestamp")));
        orders = orders.withColumn("order_month", month(col("order_purchase_timestamp")));
        orders = orders.withColumn("processing_time", datediff(col("order_approved_at"), col("order_purchase_timestamp")));
        orders = orders.withColumn("processing_time", when(col("processing_time") < 0, None).otherwise(col("processing_time")));
        if ("order_delivered_timestamp" in orders.columns) {
            orders = orders.withColumn("delivery_days", datediff(col("order_delivered_timestamp"), col("order_purchase_timestamp")));
            if ("order_estimated_delivery_date" in orders.columns) {
                orders = orders.withColumn("on_time_delivery", when(
                    col("order_delivered_timestamp").isNotNull() & col("order_estimated_delivery_date").isNotNull(),
                    col("order_delivered_timestamp") <= col("order_estimated_delivery_date")
                ).cast("int"));
            }
        } else {
            orders = orders.withColumn("delivery_days", (datediff(col("order_approved_at"), col("order_purchase_timestamp")) + 5).cast("int"));
            orders = orders.withColumn("on_time_delivery", when((col("order_id").rlike(".")), 1).otherwise(0));
        }
        orders = orders.withColumn("delivery_days", when(col("delivery_days").isNull(), 7).otherwise(col("delivery_days")));
        orders = orders.withColumn("on_time_delivery", when(col("on_time_delivery").isNull(), 0.5).otherwise(col("on_time_delivery")));
        if (!("order_estimated_delivery_date" in orders.columns)) {
            orders = orders.withColumn("order_estimated_delivery_date", expr("date_add(order_approved_at, 7)"));
        }
        return orders;

# Helper function to run the analysis pipeline
def run_spark_analysis(data_dir=".", output_dir="./output", top_n=15, forecast_periods=6):
    analyzer = SparkSupplyChainAnalytics(data_path=data_dir, output_path=output_dir);
    results = {};
    try:
        analyzer.load_data();
        analyzer.preprocess_data();
        analyzer.build_unified_dataset();
        let monthly_demand = null;
        let top_categories = null;
        try {
            if (hasattr(analyzer, 'analyze_monthly_demand')) {
                const original_method = analyzer._visualize_top_categories if hasattr(analyzer, '_visualize_top_categories') else null;
                analyzer._visualize_top_categories = (x, y) => {};
                const analysisResults = analyzer.analyze_monthly_demand(top_n=top_n);
                monthly_demand = analysisResults[1];
                top_categories = analysisResults[2];
                if (original_method) {
                    analyzer._visualize_top_categories = original_method;
                } else {
                    delattr(analyzer, '_visualize_top_categories');
                }
                results['monthly_demand'] = monthly_demand;
                results['top_categories'] = top_categories;
                print("Monthly demand analysis completed successfully");
            } else {
                print("analyze_monthly_demand method not available");
            }
        } except (e) {
            print(`Error in monthly demand analysis: ${e}`);
        }
        let seller_clusters = null;
        try {
            if (hasattr(analyzer, 'analyze_seller_performance')) {
                const original_method = analyzer._visualize_seller_clusters if hasattr(analyzer, '_visualize_seller_clusters') else null;
                analyzer._visualize_seller_clusters = (x, y) => {};
                const sellerResults = analyzer.analyze_seller_performance();
                seller_clusters = sellerResults[1];
                if (original_method) {
                    analyzer._visualize_seller_clusters = original_method;
                } else {
                    delattr(analyzer, '_visualize_seller_clusters');
                }
                results['seller_clusters'] = seller_clusters;
                print("Seller performance analysis completed successfully");
            } else {
                print("analyze_seller_performance method not available");
            }
        } except (e) {
            print(`Error in seller performance analysis: ${e}`);
        }
        print("Analysis complete. Results saved to output directory.");
        print("Note: Some advanced features were skipped due to missing methods.");
    } except (e) {
        print(`Error in supply chain analysis: ${e}`);
    }
    return results;

if __name__ == "__main__":
    import argparse;
    const parser = argparse.ArgumentParser(description='Supply Chain Analytics with Spark');
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files');
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files');
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze');
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast');
    const args = parser.parse_args();
    run_spark_analysis(data_dir=args.data_dir, output_dir=args.output_dir, top_n=args.top_n, forecast_periods=args.forecast_periods);
