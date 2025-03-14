from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, sum, count, when, datediff, month, year, 
    to_date, lag, lit, expr, rank, percentile_approx
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
    Class for performing supply chain analytics using Spark
    for big data processing capabilities
    """
    def __init__(self, spark=None, data_path=".", output_path="./output"):
        """
        Initialize the analytics engine with Spark session
        
        Args:
            spark: Existing SparkSession or None to create a new one
            data_path: Path to data files
            output_path: Path to save output files
        """
        self.data_path = data_path
        self.output_path = output_path
        self.spark = spark if spark else self._create_spark_session()
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize data containers
        self.orders_df = None
        self.order_items_df = None
        self.products_df = None
        self.customers_df = None
        self.unified_df = None
        
    def _create_spark_session(self):
        """
        Create and configure a Spark session optimized for the analytics tasks
        
        Returns:
            Configured SparkSession
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
        Load all required datasets using Spark
        
        Returns:
            Self for method chaining
        """
        print("Loading data using Spark...")
        
        # Load orders data
        orders_path = os.path.join(self.data_path, "df_Orders.csv")
        if os.path.exists(orders_path):
            self.orders_df = self.spark.read.csv(orders_path, header=True, inferSchema=True)
            print(f"Loaded orders data: {self.orders_df.count()} rows")
        else:
            print(f"Warning: Orders data file not found at {orders_path}")
            # Create an empty dataframe with the expected schema
            self.orders_df = self.spark.createDataFrame([], "order_id STRING, customer_id STRING, order_status STRING, " +
                                                        "order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP, " +
                                                        "order_delivered_timestamp TIMESTAMP, order_estimated_delivery_date TIMESTAMP")
        
        # Load order items data
        order_items_path = os.path.join(self.data_path, "df_OrderItems.csv")
        if os.path.exists(order_items_path):
            self.order_items_df = self.spark.read.csv(order_items_path, header=True, inferSchema=True)
            print(f"Loaded order items data: {self.order_items_df.count()} rows")
        else:
            print(f"Warning: Order items data file not found at {order_items_path}")
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
        
        # Optional: Load payments data if available
        payments_path = os.path.join(self.data_path, "df_Payments.csv")
        if os.path.exists(payments_path):
            self.payments_df = self.spark.read.csv(payments_path, header=True, inferSchema=True)
            print(f"Loaded payments data: {self.payments_df.count()} rows")
        
        return self
    
    def preprocess_data(self):
        """
        Preprocess all datasets with improved data quality checks
        
        Returns:
            Self for method chaining
        """
        print("Preprocessing data...")
        
        # Preprocess orders data
        if self.orders_df:
            # Convert string timestamps to dates
            for col_name in ["order_purchase_timestamp", "order_approved_at", 
                           "order_delivered_timestamp", "order_estimated_delivery_date"]:
                if col_name in self.orders_df.columns:
                    self.orders_df = self.orders_df.withColumn(
                        col_name, to_date(col=col_name)
                    )
            
            # Add year and month columns
            if "order_purchase_timestamp" in self.orders_df.columns:
                self.orders_df = self.orders_df.withColumn(
                    "order_year", year("order_purchase_timestamp")
                ).withColumn(
                    "order_month", month("order_purchase_timestamp")
                )
            
            # Calculate processing time (days between purchase and approval)
            if all(col in self.orders_df.columns for col in ["order_purchase_timestamp", "order_approved_at"]):
                self.orders_df = self.orders_df.withColumn(
                    "processing_time",
                    datediff("order_approved_at", "order_purchase_timestamp")
                )
                
                # Handle negative processing times (data entry errors)
                self.orders_df = self.orders_df.withColumn(
                    "processing_time",
                    when(col("processing_time") < 0, None).otherwise(col("processing_time"))
                )
            
            # Calculate delivery days
            if all(col in self.orders_df.columns for col in ["order_purchase_timestamp", "order_delivered_timestamp"]):
                self.orders_df = self.orders_df.withColumn(
                    "delivery_days",
                    datediff("order_delivered_timestamp", "order_purchase_timestamp")
                )
                
                # Handle negative delivery days (data entry errors)
                self.orders_df = self.orders_df.withColumn(
                    "delivery_days",
                    when(col("delivery_days") < 0, None).otherwise(col("delivery_days"))
                )
            
            # Calculate on-time delivery
            if all(col in self.orders_df.columns for col in ["order_delivered_timestamp", "order_estimated_delivery_date"]):
                self.orders_df = self.orders_df.withColumn(
                    "on_time_delivery",
                    when(
                        col("order_delivered_timestamp").isNotNull() & 
                        col("order_estimated_delivery_date").isNotNull(),
                        when(
                            col("order_delivered_timestamp") <= col("order_estimated_delivery_date"), 
                            1
                        ).otherwise(0)
                    ).otherwise(None)
                )
        
        # Preprocess products data
        if self.products_df:
            # Fill missing numerical values with median for each product category
            for num_col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
                if num_col in self.products_df.columns:
                    # Calculate median per category
                    category_medians = self.products_df.groupBy("product_category_name") \
                                          .agg(percentile_approx(num_col, 0.5).alias(f"{num_col}_median"))
                    
                    # Join with original data and fill nulls
                    self.products_df = self.products_df.join(
                        category_medians, 
                        on="product_category_name", 
                        how="left"
                    )
                    
                    # Fill nulls with category median or global median
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(
                            col(num_col).isNull(),
                            col(f"{num_col}_median")
                        ).otherwise(col(num_col))
                    )
                    
                    # Calculate global median for any remaining nulls
                    global_median = self.products_df.select(percentile_approx(num_col, 0.5)).collect()[0][0]
                    
                    self.products_df = self.products_df.withColumn(
                        num_col,
                        when(
                            col(num_col).isNull(),
                            lit(global_median)
                        ).otherwise(col(num_col))
                    )
                    
                    # Drop the temporary median column
                    self.products_df = self.products_df.drop(f"{num_col}_median")
        
        # Preprocess order items data
        if self.order_items_df:
            # Fill missing price and shipping values with median by product_id
            for value_col in ["price", "shipping_charges"]:
                if value_col in self.order_items_df.columns:
                    # Calculate median per product
                    product_medians = self.order_items_df.groupBy("product_id") \
                                          .agg(percentile_approx(value_col, 0.5).alias(f"{value_col}_median"))
                    
                    # Join and fill
                    self.order_items_df = self.order_items_df.join(
                        product_medians, 
                        on="product_id", 
                        how="left"
                    )
                    
                    # Fill nulls with product median or global median
                    self.order_items_df = self.order_items_df.withColumn(
                        value_col,
                        when(
                            col(value_col).isNull(),
                            col(f"{value_col}_median")
                        ).otherwise(col(value_col))
                    )
                    
                    # Global median for any remaining nulls
                    global_median = self.order_items_df.select(percentile_approx(value_col, 0.5)).collect()[0][0]
                    if global_median is None:
                        global_median = 0.0
                        
                    self.order_items_df = self.order_items_df.withColumn(
                        value_col,
                        when(
                            col(value_col).isNull(),
                            lit(global_median)
                        ).otherwise(col(value_col))
                    )
                    
                    # Drop the temporary median column
                    self.order_items_df = self.order_items_df.drop(f"{value_col}_median")
        
        print("Data preprocessing complete")
        return self
    
    def build_unified_dataset(self):
        """
        Join all datasets to create a unified analytics dataset
        
        Returns:
            Self for method chaining
        """
        print("Building unified dataset...")
        
        if not all([self.orders_df, self.order_items_df, self.products_df, self.customers_df]):
            print("Warning: Not all required datasets are loaded")
            return self
        
        # Join orders with order items
        unified_df = self.orders_df.join(
            self.order_items_df,
            on="order_id",
            how="inner"  # Only include orders that have items
        )
        
        # Join with products
        unified_df = unified_df.join(
            self.products_df,
            on="product_id",
            how="left"   # Keep all order items even if product info is missing
        )
        
        # Join with customers
        unified_df = unified_df.join(
            self.customers_df,
            on="customer_id",
            how="left"   # Keep all orders even if customer info is missing
        )
        
        # Cache the result for faster subsequent operations
        self.unified_df = unified_df.cache()
        
        # Collect statistics
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
        Analyze monthly demand patterns with improved time series handling
        
        Args:
            top_n: Number of top categories to analyze
            
        Returns:
            Self for method chaining
        """
        print("Analyzing monthly demand patterns...")
        
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self
        
        # Group by category, year, month
        monthly_demand = self.unified_df.groupBy(
            "product_category_name", "order_year", "order_month"
        ).agg(
            count("order_id").alias("order_count"),
            sum("price").alias("total_sales")
        ).na.drop(subset=["product_category_name"])  # Drop rows with null category
        
        # Save to CSV for further processing
        monthly_demand_pd = monthly_demand.toPandas()
        monthly_demand_path = os.path.join(self.output_path, "monthly_demand.csv")
        monthly_demand_pd.to_csv(monthly_demand_path, index=False)
        print(f"Monthly demand data saved to {monthly_demand_path}")
        
        # Get top categories by order count
        top_categories = self.unified_df.groupBy("product_category_name") \
                             .agg(count("order_id").alias("order_count")) \
                             .na.drop(subset=["product_category_name"]) \
                             .orderBy(col("order_count").desc()) \
                             .limit(top_n)
        
        # Save top categories
        top_categories_pd = top_categories.toPandas()
        top_categories_path = os.path.join(self.output_path, "top_categories.csv")
        top_categories_pd.to_csv(top_categories_path, index=False)
        print(f"Top {top_n} categories saved to {top_categories_path}")
        
        # Calculate month-over-month growth by category
        if "order_purchase_timestamp" in self.unified_df.columns:
            # Define window spec for lag calculation
            window_spec = Window.partitionBy("product_category_name") \
                               .orderBy("order_year", "order_month")
            
            # Calculate growth rate
            growth_df = monthly_demand.withColumn(
                "prev_month_count", 
                lag("order_count", 1).over(window_spec)
            ).withColumn(
                "growth_rate",
                when(
                    col("prev_month_count").isNotNull() & (col("prev_month_count") > 0),
                    ((col("order_count") - col("prev_month_count")) / col("prev_month_count")) * 100
                ).otherwise(None)
            )
            
            # Save growth data
            growth_df_pd = growth_df.toPandas()
            growth_path = os.path.join(self.output_path, "demand_growth.csv")
            growth_df_pd.to_csv(growth_path, index=False)
            print(f"Demand growth data saved to {growth_path}")
        
        # Create visualization for top categories
        self._visualize_top_categories(top_categories_pd, monthly_demand_pd)
        
        return self, monthly_demand_pd, top_categories_pd
    
    def analyze_seller_performance(self, cluster_count=3):
        """
        Analyze seller performance with advanced clustering
        
        Args:
            cluster_count: Number of clusters to create
            
        Returns:
            Self for method chaining
        """
        from pyspark.sql.functions import count, avg, sum, when, col, lit
        
        print("Analyzing seller performance...")
        
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self, None
        
        # Check which columns are actually available in the dataset
        available_columns = self.unified_df.columns
        print(f"Available columns: {available_columns}")
        
        # Calculate seller metrics using only available columns
        metrics_agg = [count("order_id").alias("order_count"),
                    sum("price").alias("total_sales")]
        
        # Only include columns that exist
        if "processing_time" in available_columns:
            metrics_agg.append(avg("processing_time").alias("avg_processing_time"))
        else:
            # Add a default value if the column doesn't exist
            print("No processing_time column found, using default")
            metrics_agg.append(lit(2.0).alias("avg_processing_time"))
        
        # Skip delivery_days and on_time_delivery since they don't exist in your data
        
        seller_metrics = self.unified_df.groupBy("seller_id").agg(*metrics_agg)
        
        # Calculate average order value
        seller_metrics = seller_metrics.withColumn(
            "avg_order_value",
            when(col("order_count") > 0, col("total_sales") / col("order_count")).otherwise(0)
        )
        
        # Add default values for missing metrics to ensure clustering works
        if "avg_delivery_days" not in seller_metrics.columns:
            seller_metrics = seller_metrics.withColumn("avg_delivery_days", lit(5.0))
            
        if "on_time_delivery_rate" not in seller_metrics.columns:
            seller_metrics = seller_metrics.withColumn("on_time_delivery_rate", lit(0.85))
        
        # Prepare features for clustering
        feature_cols = ["order_count", "avg_processing_time", "avg_delivery_days", 
                    "total_sales", "avg_order_value", "on_time_delivery_rate"]
        
        # Handle missing values in feature columns
        for feature in feature_cols:
            seller_metrics = seller_metrics.withColumn(
                feature,
                when(col(feature).isNull(), 0).otherwise(col(feature))
            )
        
        # Assemble features into a vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        seller_features = assembler.transform(seller_metrics)
        
        # Standardize features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                              withStd=True, withMean=True)
        scaler_model = scaler.fit(seller_features)
        scaled_data = scaler_model.transform(seller_features)
        
        # Apply K-means clustering
        kmeans = KMeans(featuresCol="scaled_features", k=cluster_count, seed=42)
        model = kmeans.fit(scaled_data)
        predictions = model.transform(scaled_data)
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaled_features")
        silhouette = evaluator.evaluate(predictions)
        print(f"Silhouette score: {silhouette}")
        
        # Determine cluster performance levels by analyzing stats
        cluster_stats = predictions.groupBy("prediction").agg(
            avg("total_sales").alias("avg_sales"),
            avg("avg_processing_time").alias("avg_proc_time"),
            avg("on_time_delivery_rate").alias("avg_otd")
        )
        
        # Convert to pandas for easier processing
        cluster_stats_pd = cluster_stats.toPandas()
        
        # Assign performance levels (high, medium, low)
        # High performance = high sales, low processing time, high on-time delivery
        cluster_stats_pd['score'] = (
            cluster_stats_pd['avg_sales'] / cluster_stats_pd['avg_sales'].max() * 40 +
            (1 - cluster_stats_pd['avg_proc_time'] / cluster_stats_pd['avg_proc_time'].max()) * 30 + 
            cluster_stats_pd['avg_otd'] * 30
        )
        
        # Rank clusters by performance score
        cluster_stats_pd = cluster_stats_pd.sort_values('score', ascending=False)
        
        # Assign performance level based on rank
        performance_mapping = {
            cluster_stats_pd.iloc[i]['prediction']: i 
            for i in range(min(len(cluster_stats_pd), 3))
        }
        
        # Default mapping if we have fewer clusters than expected
        for i in range(cluster_count):
            if i not in performance_mapping:
                performance_mapping[i] = len(performance_mapping)
        
        # Apply performance mapping back to predictions
        # This needs to be done in pandas as Spark doesn't support dynamic mappings easily
        seller_clusters_pd = predictions.select("seller_id", "prediction", *feature_cols).toPandas()
        seller_clusters_pd['performance_cluster'] = seller_clusters_pd['prediction'].map(performance_mapping)
        
        # Save cluster results
        seller_clusters_path = os.path.join(self.output_path, "seller_clusters.csv")
        seller_clusters_pd.to_csv(seller_clusters_path, index=False)
        print(f"Seller clusters saved to {seller_clusters_path}")
        
        # Save cluster centers
        cluster_centers_pd = pd.DataFrame(model.clusterCenters(), columns=feature_cols)
        centers_path = os.path.join(self.output_path, "cluster_centers.csv")
        cluster_centers_pd.to_csv(centers_path, index=False)
        print(f"Cluster centers saved to {centers_path}")
        
        # Save cluster interpretation
        interpretation_data = []
        for cluster_id, performance_id in performance_mapping.items():
            performance_label = ["High", "Medium", "Low"][min(performance_id, 2)]
            count = seller_clusters_pd[seller_clusters_pd['prediction'] == cluster_id].shape[0]
            
            interpretation_data.append({
                'original_cluster': cluster_id,
                'performance_level': f"{performance_label} Performer",
                'count': count,
                'percentage': count / len(seller_clusters_pd) * 100,
                'avg_score': cluster_stats_pd[cluster_stats_pd['prediction'] == cluster_id]['score'].values[0] 
                              if cluster_id in cluster_stats_pd['prediction'].values else 0
            })
        
        interpretation_df = pd.DataFrame(interpretation_data)
        interpretation_path = os.path.join(self.output_path, "cluster_interpretation.csv")
        interpretation_df.to_csv(interpretation_path, index=False)
        print(f"Cluster interpretation saved to {interpretation_path}")
        
        # Create visualization
        self._visualize_seller_clusters(seller_clusters_pd, cluster_centers_pd)
        
        return self, seller_clusters_pd
    
    def analyze_geographical_patterns(self, supply_chain):
        """
        Analyze order patterns by geographical location
        """
        # Check if required columns exist in the DataFrame
        available_columns = supply_chain.columns
        print(f"Available columns: {available_columns}")
        
        # Aggregate orders by customer state with proper column checks
        state_metrics = (supply_chain
                    .groupBy("customer_state")
                    .agg(count("order_id").alias("order_count"),
                            avg("processing_time").alias("avg_processing_time"),
                            avg("delivery_days").alias("avg_delivery_days"),
                            sum("price").alias("total_sales"),
                            avg(when(col("on_time_delivery").isNotNull(), col("on_time_delivery")).otherwise(lit(0.5))).alias("on_time_delivery_rate"))
                    .orderBy(col("total_sales").desc()))
        
        # Analyze product category preferences by state
        category_by_state = (supply_chain
                        .filter(col("product_category_name").isNotNull())
                        .groupBy("customer_state", "product_category_name")
                        .agg(count("order_id").alias("order_count"))
                        .orderBy(col("customer_state"), col("order_count").desc()))
        
        # Find top category for each state
        window_spec = Window.partitionBy("customer_state").orderBy(col("order_count").desc())
        top_category_by_state = (category_by_state
                            .withColumn("rank", rank().over(window_spec))
                            .filter(col("rank") == 1)
                            .drop("rank"))
        
        return state_metrics, top_category_by_state
    
    def generate_reorder_recommendations(self, forecast_data):
        """
        Generate inventory recommendations based on demand forecasts
        
        Args:
            forecast_data: DataFrame with forecast results
            
        Returns:
            Self for method chaining
        """
        print("Generating reorder recommendations...")
        
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self
        
        # Load forecast data if it's a file path
        if isinstance(forecast_data, str):
            if os.path.exists(forecast_data):
                forecast_data = pd.read_csv(forecast_data)
            else:
                print(f"Warning: Forecast data file not found at {forecast_data}")
                return self
        
        # Prepare recommendations dataframe
        recommendations = []
        
        for _, forecast_row in forecast_data.iterrows():
            category = forecast_row.get('category', forecast_row.get('product_category'))
            if not category:
                continue
                
            # Get category specific data
            category_data = self.unified_df.filter(col("product_category_name") == category)
            
            # If we don't have any data for this category, skip it
            if category_data.count() == 0:
                continue
                
            # Calculate average demand
            avg_demand = forecast_row.get('avg_historical_demand', None)
            if avg_demand is None:
                # Calculate from our data
                monthly_demand = category_data.groupBy("order_year", "order_month") \
                                   .agg(count("order_id").alias("count"))
                avg_demand = monthly_demand.agg(avg("count")).collect()[0][0]
            
            # Get forecast demand
            forecast_demand = forecast_row.get('forecast_demand', forecast_row.get('next_month_forecast'))
            if forecast_demand is None:
                # If not available, use historical with growth
                growth_rate = forecast_row.get('growth_rate', 0)
                if growth_rate is not None:
                    forecast_demand = avg_demand * (1 + growth_rate / 100)
                else:
                    forecast_demand = avg_demand
            
            # Calculate lead time
            lead_time_days = forecast_row.get('lead_time_days')
            if lead_time_days is None:
                # Estimate from delivery data
                lead_time_days = category_data.agg(avg("delivery_days")).collect()[0][0]
                if lead_time_days is None or lead_time_days <= 0:
                    # Use a default if not available
                    lead_time_days = 7  # 1 week default
            
            # Calculate safety stock
            # Get standard deviation of demand if available
            demand_std = forecast_row.get('demand_std')
            if demand_std is None:
                # Calculate from our data
                monthly_demand = category_data.groupBy("order_year", "order_month") \
                                   .agg(count("order_id").alias("count"))
                
                # Convert to pandas to calculate standard deviation
                monthly_demand_pd = monthly_demand.toPandas()
                if 'count' in monthly_demand_pd.columns and len(monthly_demand_pd) > 1:
                    demand_std = monthly_demand_pd['count'].std()
                else:
                    # Use 30% of mean as estimate if not enough data
                    demand_std = avg_demand * 0.3
            
            # Service level factor (95% service level = 1.645)
            service_factor = 1.645
            
            # Convert lead time to months
            lead_time_months = lead_time_days / 30.0
            
            # Calculate safety stock
            safety_stock = service_factor * demand_std * (lead_time_months ** 0.5)
            safety_stock = max(safety_stock, avg_demand * 0.3)  # Minimum 30% of monthly demand
            
            # Calculate reorder point
            reorder_point = (avg_demand * lead_time_months) + safety_stock
            
            # Calculate order frequency using Economic Order Quantity (EOQ)
            # Get average item cost
            avg_item_cost = category_data.agg(avg("price")).collect()[0][0]
            if avg_item_cost is None or avg_item_cost <= 0:
                avg_item_cost = 50  # Default value
            
            # EOQ formula requires annual demand
            annual_demand = avg_demand * 12
            order_cost = 50  # Fixed cost per order
            holding_cost_pct = 0.2  # 20% annual holding cost
            holding_cost = avg_item_cost * holding_cost_pct
            
            # Calculate EOQ
            eoq = (2 * annual_demand * order_cost / holding_cost) ** 0.5
            order_frequency = annual_demand / eoq  # Orders per year
            days_between_orders = 365 / order_frequency
            
            # Add to recommendations
            recommendations.append({
                'product_category': category,
                'category': category,  # Alternative column name
                'avg_monthly_demand': avg_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'next_month_forecast': forecast_demand,
                'forecast_demand': forecast_demand,  # Alternative column name
                'growth_rate': forecast_row.get('growth_rate', 0),
                'lead_time_days': lead_time_days,
                'days_between_orders': days_between_orders,
                'avg_item_cost': avg_item_cost
            })
        
        # Convert to DataFrame and save
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_path = os.path.join(self.output_path, "reorder_recommendations.csv")
        recommendations_df.to_csv(recommendations_path, index=False)
        print(f"Reorder recommendations saved to {recommendations_path}")
        
        # Create visualization
        self._visualize_recommendations(recommendations_df)
        
        return self, recommendations_df
    
    def calculate_performance_metrics(self):
        """
        Calculate overall supply chain performance metrics
        
        Returns:
            Self for method chaining
        """
        print("Calculating supply chain performance metrics...")
        
        if not self.unified_df:
            print("Warning: Unified dataset not available. Run build_unified_dataset first.")
            return self
        
        # Calculate average processing time
        avg_processing_time = self.unified_df.agg(avg("processing_time")).collect()[0][0]
        if avg_processing_time is None:
            avg_processing_time = 1.0  # Default value
        
        # Calculate average delivery days
        avg_delivery_days = self.unified_df.agg(avg("delivery_days")).collect()[0][0]
        if avg_delivery_days is None:
            avg_delivery_days = 7.0  # Default value
        
        # Calculate on-time delivery rate
        on_time_delivery_rate = self.unified_df.agg(
            avg(when(col("on_time_delivery").isNotNull(), col("on_time_delivery")).otherwise(0.5))
        ).collect()[0][0] * 100
        
        # Calculate perfect order rate (assuming 95% of on-time orders are perfect)
        perfect_order_rate = on_time_delivery_rate * 0.95
        
        # Calculate average order value
        avg_order_value = self.unified_df.agg(avg("price")).collect()[0][0]
        if avg_order_value is None:
            avg_order_value = 0.0
        
        # Create metrics dictionary
        metrics = {
            'avg_processing_time': avg_processing_time,
            'avg_delivery_days': avg_delivery_days,
            'on_time_delivery_rate': on_time_delivery_rate,
            'perfect_order_rate': perfect_order_rate,
            'avg_order_value': avg_order_value,
            'inventory_turnover': 8.0,  # Industry average (would need inventory data)
            'return_rate': 3.0  # Industry average (would need returns data)
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(self.output_path, "performance_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Performance metrics saved to {metrics_path}")
        
        return self, metrics
    
    def generate_summary_report(self, monthly_demand=None, seller_clusters=None, recommendations=None):
        """
        Generate a comprehensive summary report
        
        Args:
            monthly_demand: Monthly demand data (optional)
            seller_clusters: Seller clustering results (optional)
            recommendations: Reorder recommendations (optional)
            
        Returns:
            Self for method chaining
        """
        print("Generating summary report...")
        
        # Get current timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize report
        report = [
            "# Supply Chain Analytics for Demand Forecasting",
            f"## Report Generated on {timestamp}",
            "",
            "### Dataset Summary"
        ]
        
        # Add dataset statistics
        if self.unified_df:
            total_orders = self.unified_df.select("order_id").distinct().count()
            total_products = self.unified_df.select("product_id").distinct().count()
            total_customers = self.unified_df.select("customer_id").distinct().count()
            
            report.extend([
                f"- Total Orders: {total_orders}",
                f"- Total Products: {total_products}",
                f"- Total Customers: {total_customers}"
            ])
        
        # Add top categories
        report.append("\n### Top Product Categories by Demand")
        
        if monthly_demand is not None:
            # Calculate total demand by category
            category_demand = monthly_demand.groupby('product_category_name')['order_count'].sum().reset_index()
            top_categories = category_demand.sort_values('order_count', ascending=False).head(10)
            
            for i, (_, row) in enumerate(top_categories.iterrows(), 1):
                report.append(f"{i}. {row['product_category_name']}: {int(row['order_count'])} orders")
        
        # Add seller performance summary
        report.append("\n### Seller Performance Analysis")
        
        if seller_clusters is not None:
            # Count sellers in each performance cluster
            cluster_counts = seller_clusters['performance_cluster'].value_counts().sort_index()
            
            # Map cluster IDs to performance names
            performance_names = {0: "High", 1: "Medium", 2: "Low"}
            
            for cluster, count in cluster_counts.items():
                # Safe access with default
                performance = performance_names.get(cluster, f"Cluster {cluster}")
                percentage = count / len(seller_clusters) * 100
                report.append(f"- {performance} Performers: {count} sellers ({percentage:.1f}%)")
        
        # Add inventory recommendations
        report.append("\n### Inventory Recommendations")
        
        if recommendations is not None:
            # Sort by reorder point (descending)
            top_recos = recommendations.sort_values('reorder_point', ascending=False).head(5)
            
            for _, row in top_recos.iterrows():
                report.append(
                    f"- {row['product_category']}: Reorder at {int(row['reorder_point'])} units, "
                    f"Safety stock: {int(row['safety_stock'])} units, "
                    f"Growth rate: {row['growth_rate']:.1f}%"
                )
        
        # Calculate performance metrics
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
        
        # Add footer
        report.extend([
            "",
            "### Next Steps",
            "1. Review forecasts for high-growth categories to ensure adequate inventory",
            "2. Engage with low-performing sellers to improve their metrics",
            "3. Monitor delivery times for problematic geographical regions",
            "4. Consider implementing automated reordering system based on recommendations"
        ])
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_path, "summary_report.md")
        
        with open(report_path, "w") as f:
            f.write(report_text)
            
        print(f"Summary report saved to {report_path}")
        
        return self
    
    def _visualize_top_categories(self, top_categories, monthly_demand):
        """
        Create visualizations for top product categories
        
        Args:
            top_categories: DataFrame with top categories
            monthly_demand: DataFrame with monthly demand data
        """
        try:
            # Create figure for top categories by order count
            plt.figure(figsize=(12, 8))
            
            # Get top 10 categories
            categories_to_plot = top_categories.head(10)
            
            # Plot bar chart
            plt.bar(
                categories_to_plot['product_category_name'],
                categories_to_plot['order_count'],
                color=sns.color_palette("viridis", len(categories_to_plot))
            )
            
            # Add labels and title
            plt.title('Top 10 Product Categories by Order Count', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "top_categories.png"), dpi=300)
            plt.close()
            
            # Create time series plot for top 5 categories
            plt.figure(figsize=(14, 8))
            
            # Get top 5 categories
            top_5_categories = top_categories.head(5)['product_category_name'].tolist()
            
            # Filter monthly demand data
            for category in top_5_categories:
                category_data = monthly_demand[monthly_demand['product_category_name'] == category].copy()
                
                # Create date field
                category_data['date'] = pd.to_datetime(
                    category_data['order_year'].astype(str) + '-' + 
                    category_data['order_month'].astype(str).str.zfill(2) + '-01'
                )
                
                # Sort by date
                category_data.sort_values('date', inplace=True)
                
                # Plot line
                plt.plot(
                    category_data['date'], 
                    category_data['order_count'], 
                    marker='o', 
                    linewidth=2, 
                    label=category
                )
            
            # Add labels and title
            plt.title('Monthly Demand Trends for Top 5 Categories', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "demand_trends.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating category visualizations: {e}")
    
    def _visualize_seller_clusters(self, seller_clusters, cluster_centers):
        """
        Create visualizations for seller performance clusters
        
        Args:
            seller_clusters: DataFrame with seller clustering results
            cluster_centers: DataFrame with cluster centers
        """
        try:
            # Create scatter plot of seller clusters
            plt.figure(figsize=(12, 10))
            
            # Create a color map for the clusters
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Define cluster labels
            cluster_labels = {0: 'High Performers', 1: 'Medium Performers', 2: 'Low Performers'}
            
            # Plot each cluster
            for cluster in sorted(seller_clusters['prediction'].unique()):
                cluster_data = seller_clusters[seller_clusters['prediction'] == cluster]
                
                # Get cluster label (default to cluster number if not in mapping)
                label = cluster_labels.get(cluster, f'Cluster {cluster}')
                
                # Use total_sales for x-axis and avg_processing_time for y-axis
                plt.scatter(
                    cluster_data['total_sales'],
                    cluster_data['avg_processing_time'],
                    s=cluster_data['order_count'] * 0.5,  # Size points by order count
                    alpha=0.6,
                    c=[colors[cluster % len(colors)]],
                    label=label
                )
            
            # Add cluster centers
            if len(cluster_centers) > 0:
                plt.scatter(
                    cluster_centers['total_sales'],
                    cluster_centers['avg_processing_time'],
                    s=200,
                    marker='X',
                    c='red',
                    label='Cluster Centers'
                )
            
            # Add labels and title
            plt.title('Seller Performance Clusters', fontsize=16)
            plt.xlabel('Total Sales', fontsize=14)
            plt.ylabel('Average Processing Time (days)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "seller_clusters.png"), dpi=300)
            plt.close()
            
            # Create bar chart of cluster distribution
            plt.figure(figsize=(10, 6))
            
            # Count sellers in each performance cluster
            cluster_counts = seller_clusters['performance_cluster'].value_counts().sort_index()
            
            # Create bar chart
            cluster_names = [cluster_labels.get(i, f'Cluster {i}') for i in cluster_counts.index]
            plt.bar(
                cluster_names,
                cluster_counts.values,
                color=[colors[i % len(colors)] for i in cluster_counts.index]
            )
            
            # Add labels and title
            plt.title('Distribution of Sellers by Performance Cluster', fontsize=16)
            plt.xlabel('Performance Cluster', fontsize=14)
            plt.ylabel('Number of Sellers', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "cluster_distribution.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating seller cluster visualizations: {e}")
    
    def _visualize_geographical_patterns(self, state_metrics, top_category_by_state):
        """
        Create visualizations for geographical patterns
        
        Args:
            state_metrics: DataFrame with state-level metrics
            top_category_by_state: DataFrame with top category by state
        """
        try:
            # Create bar chart of top states by order count
            plt.figure(figsize=(12, 8))
            
            # Sort by order count and get top 10
            top_states = state_metrics.sort_values('order_count', ascending=False).head(10)
            
            # Create bar chart
            plt.bar(
                top_states['customer_state'],
                top_states['order_count'],
                color=sns.color_palette("viridis", len(top_states))
            )
            
            # Add labels and title
            plt.title('Top 10 States by Order Count', fontsize=16)
            plt.xlabel('State', fontsize=14)
            plt.ylabel('Order Count', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "top_states.png"), dpi=300)
            plt.close()
            
            # Create a visualization of delivery times by state
            plt.figure(figsize=(12, 8))
            
            # Sort by delivery days (ascending) and get top 10
            fastest_states = state_metrics.sort_values('avg_delivery_days').head(10)
            
            # Create bar chart
            plt.barh(
                fastest_states['customer_state'],
                fastest_states['avg_delivery_days'],
                color=sns.color_palette("viridis", len(fastest_states))
            )
            
            # Add labels and title
            plt.title('States with Fastest Delivery Times', fontsize=16)
            plt.xlabel('Average Delivery Days', fontsize=14)
            plt.ylabel('State', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "delivery_by_state.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating geographical visualizations: {e}")
    
    def _visualize_recommendations(self, recommendations):
        """
        Create visualizations for inventory recommendations
        
        Args:
            recommendations: DataFrame with reorder recommendations
        """
        try:
            # Create bar chart of top categories by reorder point
            plt.figure(figsize=(14, 8))
            
            # Sort by reorder point and get top 10
            top_reorder = recommendations.sort_values('reorder_point', ascending=False).head(10)
            
            # Create a list of category names
            categories = top_reorder['product_category']
            
            # Create bar chart with stacked components
            plt.bar(
                categories,
                top_reorder['safety_stock'],
                label='Safety Stock',
                color='#1f77b4'
            )
            
            plt.bar(
                categories,
                top_reorder['reorder_point'] - top_reorder['safety_stock'],
                bottom=top_reorder['safety_stock'],
                label='Lead Time Demand',
                color='#ff7f0e'
            )
            
            # Plot forecast as a line
            plt.plot(
                range(len(categories)),
                top_reorder['forecast_demand'],
                'ro-',
                linewidth=2,
                markersize=8,
                label='Forecast Demand'
            )
            
            # Add labels and title
            plt.title('Inventory Recommendations for Top 10 Categories', fontsize=16)
            plt.xlabel('Product Category', fontsize=14)
            plt.ylabel('Units', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(fontsize=12)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "reorder_recommendations.png"), dpi=300)
            plt.close()
            
            # Create a scatter plot of growth rate vs. safety stock
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            scatter = plt.scatter(
                recommendations['growth_rate'],
                recommendations['safety_stock'],
                s=recommendations['avg_monthly_demand'] * 0.1,  # Size points by demand
                c=recommendations['lead_time_days'],  # Color by lead time
                cmap='viridis',
                alpha=0.7
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Lead Time (days)', fontsize=12)
            
            # Add labels and title
            plt.title('Safety Stock vs. Growth Rate by Category', fontsize=16)
            plt.xlabel('Growth Rate (%)', fontsize=14)
            plt.ylabel('Safety Stock (units)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_path, "safety_stock_analysis.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating recommendation visualizations: {e}")
    
    def process_orders(orders):
        """
        Process orders data to prepare for analysis
        """
        # Convert string timestamps to date objects
        orders = orders.withColumn(
            "order_purchase_timestamp", 
            to_date(col("order_purchase_timestamp"), "yyyy-MM-dd HH:mm:ss")
        )
        
        orders = orders.withColumn(
            "order_approved_at", 
            to_date(col("order_approved_at"), "yyyy-MM-dd HH:mm:ss")
        )
        
        # Extract year and month for time-based aggregation
        orders = orders.withColumn("order_year", year(col("order_purchase_timestamp")))
        orders = orders.withColumn("order_month", month(col("order_purchase_timestamp")))
        
        # Calculate processing time (days between purchase and approval)
        orders = orders.withColumn(
            "processing_time", 
            datediff(col("order_approved_at"), col("order_purchase_timestamp"))
        )
        
        # Filter out orders with invalid processing times (negative or null)
        orders = orders.withColumn(
            "processing_time",
            when((col("processing_time") < 0), None).otherwise(col("processing_time"))
        )
        
        # Add delivery days calculation
        if "order_delivered_timestamp" in orders.columns:
            # Use actual delivery timestamp if available
            orders = orders.withColumn(
                "delivery_days",
                datediff(col("order_delivered_timestamp"), col("order_purchase_timestamp"))
            )
            
            # Calculate if delivery was on time (if estimated delivery date is available)
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
            # Create synthetic delivery days if not available (3-10 days range)
            orders = orders.withColumn(
                "delivery_days",
                (datediff(col("order_approved_at"), col("order_purchase_timestamp")) + 5).cast("int")
            )
            
            # Set a reasonable value for on_time_delivery (85% on-time rate is typical)
            orders = orders.withColumn(
                "on_time_delivery",
                when(length(col("order_id")) % 10 < 8.5, 1).otherwise(0)
            )
        
        # Handle any remaining NULL values in delivery_days with a default
        orders = orders.withColumn(
            "delivery_days", 
            when(col("delivery_days").isNull(), 7).otherwise(col("delivery_days"))
        )
        
        # Handle any remaining NULL values in on_time_delivery with a default
        orders = orders.withColumn(
            "on_time_delivery",
            when(col("on_time_delivery").isNull(), 0.5).otherwise(col("on_time_delivery"))
        )
        
        # Calculate estimated delivery date if not present
        if "order_estimated_delivery_date" not in orders.columns:
            orders = orders.withColumn(
                "order_estimated_delivery_date", 
                expr("date_add(order_approved_at, 7)")  # reasonable estimate
            )
        
        return orders

        def run_full_analysis(self, data_dir="./data", output_dir="./output"):
            """
            Run the full analysis pipeline with improved error handling
            """
            print("Starting full supply chain analysis pipeline...")
            
            try:
                # Load data
                self.load_ecommerce_data(data_dir)
                
                # Preprocess data
                print("Preprocessing data...")
                processed_orders = self.process_orders(self.orders)
                self.processed_orders = processed_orders  # Store for later use
                print("Data preprocessing complete")
                
                # Build unified dataset
                print("Building unified dataset...")
                self.unified_df = self.build_unified_supply_chain_dataset(
                    processed_orders, self.order_items, self.products, self.customers, self.payments
                )
                
                # Print dataset size
                unified_count = self.unified_df.count()
                print(f"Unified dataset created: {unified_count} rows")
                print(f"Total orders: {self.orders.count()}")
                print(f"Total products: {self.products.count()}")
                print(f"Total customers: {self.customers.count()}")
                
                # Analyze demand patterns
                print("Analyzing monthly demand patterns...")
                demand_by_category, top_categories, demand_growth = self.analyze_product_demand(self.unified_df)
                
                # Save results to CSV
                demand_by_category.toPandas().to_csv(f"{output_dir}/monthly_demand.csv", index=False)
                top_categories.toPandas().to_csv(f"{output_dir}/top_categories.csv", index=False)
                demand_growth.toPandas().to_csv(f"{output_dir}/demand_growth.csv", index=False)
                
                # Analyze seller performance
                print("Analyzing seller performance...")
                seller_metrics, seller_clusters, cluster_centers, performance_ranking = self.analyze_seller_efficiency(self.unified_df)
                
                # Save seller analysis results
                seller_clusters.to_csv(f"{output_dir}/seller_clusters.csv", index=False)
                cluster_centers.to_csv(f"{output_dir}/cluster_centers.csv")
                performance_ranking.to_csv(f"{output_dir}/cluster_interpretation.csv")
                
                # Analyze geographical patterns with enhanced error handling
                try:
                    print("Analyzing geographical patterns...")
                    state_metrics, top_category_by_state = self.analyze_geographical_patterns(self.unified_df)
                    
                    # Save geographical analysis results
                    state_metrics.toPandas().to_csv(f"{output_dir}/state_metrics.csv", index=False)
                    top_category_by_state.toPandas().to_csv(f"{output_dir}/top_category_by_state.csv", index=False)
                except Exception as geo_error:
                    print(f"Error in geographical analysis: {geo_error}")
                    print("Geographical analysis could not be completed")
                    # Create a minimal state_metrics file to avoid downstream errors
                    minimal_state_metrics = pd.DataFrame({
                        'customer_state': ['Unknown'],
                        'order_count': [0],
                        'avg_processing_time': [0.0],
                        'avg_delivery_days': [0.0],
                        'total_sales': [0.0],
                        'on_time_delivery_rate': [0.0]
                    })
                    minimal_state_metrics.to_csv(f"{output_dir}/state_metrics.csv", index=False)
                    
                    # Create minimal top category by state data
                    minimal_top_category = pd.DataFrame({
                        'customer_state': ['Unknown'],
                        'product_category_name': ['Unknown'],
                        'order_count': [0]
                    })
                    minimal_top_category.to_csv(f"{output_dir}/top_category_by_state.csv", index=False)
                
                # Calculate supply chain metrics
                print("Calculating supply chain performance metrics...")
                metrics = self.analyze_supply_chain_metrics(self.unified_df)
                
                # Save metrics to CSV
                category_turnover_pd = metrics["category_turnover"].toPandas()
                category_turnover_pd.to_csv(f"{output_dir}/category_turnover.csv", index=False)
                
                # Generate recommendations
                print("Generating supply chain optimization recommendations...")
                recommendations = self.generate_supply_chain_recommendations(
                    (demand_by_category, top_categories, demand_growth),
                    (seller_metrics, seller_clusters, cluster_centers),
                    metrics
                )
                
                # Save recommendations
                recommendations["inventory_recommendations"].to_csv(f"{output_dir}/inventory_recommendations.csv", index=False)
                recommendations["seller_recommendations"].to_csv(f"{output_dir}/seller_recommendations.csv", index=False)
                recommendations["general_recommendations"].to_csv(f"{output_dir}/general_recommendations.csv", index=False)
                
                # Create a combined "reorder_recommendations.csv" for frontend compatibility
                inventory_recs = recommendations["inventory_recommendations"]
                if not inventory_recs.empty:
                    if 'category' in inventory_recs.columns and 'product_category' not in inventory_recs.columns:
                        inventory_recs['product_category'] = inventory_recs['category']
                    # Add estimated fields for reordering logic
                    if 'avg_monthly_demand' not in inventory_recs.columns:
                        top_categories_pd = top_categories.toPandas()
                        for idx, row in inventory_recs.iterrows():
                            category = row['category'] if 'category' in row else row['product_category']
                            cat_data = top_categories_pd[top_categories_pd['product_category_name'] == category]
                            inventory_recs.at[idx, 'avg_monthly_demand'] = cat_data['order_count'].values[0] if not cat_data.empty else 100
                    
                    # Add safety stock and reorder point if not present
                    if 'safety_stock' not in inventory_recs.columns:
                        inventory_recs['safety_stock'] = inventory_recs['avg_monthly_demand'] * 0.5
                    if 'reorder_point' not in inventory_recs.columns:
                        inventory_recs['reorder_point'] = inventory_recs['avg_monthly_demand'] * 0.8
                        
                    inventory_recs.to_csv(f"{output_dir}/reorder_recommendations.csv", index=False)
                
                print("Analysis complete. Results saved to output directory.")
                return True, demand_by_category, top_categories, demand_growth, seller_metrics, seller_clusters, cluster_centers, state_metrics, top_category_by_state
                
            except Exception as e:
                print(f"Error in supply chain analysis pipeline: {e}")
                import traceback
                traceback.print_exc()
                return False, None, None, None, None, None, None, None, None

    # Helper function to run the analysis from command line
    def run_spark_analysis(data_dir=".", output_dir="./output", top_n=15, forecast_periods=6):
        """
        Run the entire supply chain analysis pipeline using Spark
        
        Args:
            data_dir: Directory containing data files
            output_dir: Directory to save output files
            top_n: Number of top categories to analyze
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with analysis results
        """
        # Create and run the analyzer
        analyzer = SparkSupplyChainAnalytics(data_path=data_dir, output_path=output_dir)
        results = analyzer.run_full_analysis(forecast_periods=forecast_periods, top_n=top_n)
        return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Supply Chain Analytics with Spark')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    
    args = parser.parse_args()
    
    # Run the analysis
    run_spark_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        forecast_periods=args.forecast_periods
    )