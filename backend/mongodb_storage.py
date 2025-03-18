#!/usr/bin/env python3
"""
MongoDB Storage Module for Supply Chain Analytics

This module provides functions to store and retrieve supply chain analytics data
from MongoDB, enabling persistent storage of processed data, forecasts, and analysis results.
"""

import pymongo
import pandas as pd
import json
import logging
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.json_util import dumps, loads

# Configure logging
logger = logging.getLogger('supply_chain_analytics.mongodb')

class MongoDBStorage:
    """
    Class for storing and retrieving supply chain analytics data in MongoDB.
    """
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="supply_chain_analytics"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string. Default connects to local MongoDB.
            db_name: Name of the MongoDB database to use.
        """
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            logger.info(f"Connected to MongoDB database: {db_name}")
            
            # Create collections if they don't exist
            if "demand_data" not in self.db.list_collection_names():
                self.db.create_collection("demand_data")
            if "forecasts" not in self.db.list_collection_names():
                self.db.create_collection("forecasts")
            if "suppliers" not in self.db.list_collection_names():
                self.db.create_collection("suppliers")
            if "inventory" not in self.db.list_collection_names():
                self.db.create_collection("inventory")
            if "analysis_metadata" not in self.db.list_collection_names():
                self.db.create_collection("analysis_metadata")
                
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
            
    def store_monthly_demand(self, demand_data, run_id=None):
        """
        Store monthly demand data in MongoDB.
        
        Args:
            demand_data: DataFrame with monthly demand data
            run_id: Optional analysis run ID to associate the data with
            
        Returns:
            Number of records stored
        """
        if not isinstance(demand_data, pd.DataFrame):
            raise ValueError("demand_data must be a pandas DataFrame")
            
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
        records = []
        for _, row in demand_data.iterrows():
            record = row.to_dict()
            # Handle date objects for MongoDB storage
            if 'date' in record and isinstance(record['date'], datetime):
                record['date'] = record['date'].isoformat()
            record['run_id'] = run_id
            record['stored_at'] = datetime.now()
            records.append(record)
            
        if records:
            result = self.db.demand_data.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} demand records with run_id {run_id}")
            return len(result.inserted_ids)
        return 0
    
    def store_forecasts(self, forecast_data, run_id=None):
        """
        Store forecast results in MongoDB.
        
        Args:
            forecast_data: DataFrame with forecast data
            run_id: Optional analysis run ID to associate the data with
            
        Returns:
            Number of records stored
        """
        if not isinstance(forecast_data, pd.DataFrame):
            raise ValueError("forecast_data must be a pandas DataFrame")
            
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
        records = []
        for _, row in forecast_data.iterrows():
            record = row.to_dict()
            # Handle forecast values if they're in a complex structure
            if 'forecast_values' in record and isinstance(record['forecast_values'], list):
                record['forecast_values'] = json.dumps(record['forecast_values'])
            # Handle confidence intervals
            for field in ['lower_ci', 'upper_ci']:
                if field in record and isinstance(record[field], list):
                    record[field] = json.dumps(record[field])
            record['run_id'] = run_id
            record['stored_at'] = datetime.now()
            records.append(record)
            
        if records:
            result = self.db.forecasts.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} forecast records with run_id {run_id}")
            return len(result.inserted_ids)
        return 0
    
    def store_supplier_clusters(self, supplier_data, run_id=None):
        """
        Store supplier clustering data in MongoDB.
        
        Args:
            supplier_data: DataFrame with supplier clustering data
            run_id: Optional analysis run ID to associate the data with
            
        Returns:
            Number of records stored
        """
        if not isinstance(supplier_data, pd.DataFrame):
            raise ValueError("supplier_data must be a pandas DataFrame")
            
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
        records = []
        for _, row in supplier_data.iterrows():
            record = row.to_dict()
            record['run_id'] = run_id
            record['stored_at'] = datetime.now()
            records.append(record)
            
        if records:
            result = self.db.suppliers.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} supplier records with run_id {run_id}")
            return len(result.inserted_ids)
        return 0
    
    def store_inventory_recommendations(self, inventory_data, run_id=None):
        """
        Store inventory optimization recommendations in MongoDB.
        
        Args:
            inventory_data: DataFrame with inventory recommendations
            run_id: Optional analysis run ID to associate the data with
            
        Returns:
            Number of records stored
        """
        if not isinstance(inventory_data, pd.DataFrame):
            raise ValueError("inventory_data must be a pandas DataFrame")
            
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
        records = []
        for _, row in inventory_data.iterrows():
            record = row.to_dict()
            record['run_id'] = run_id
            record['stored_at'] = datetime.now()
            records.append(record)
            
        if records:
            result = self.db.inventory.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} inventory recommendation records with run_id {run_id}")
            return len(result.inserted_ids)
        return 0
    
    def store_analysis_metadata(self, metadata):
        """
        Store metadata about an analysis run in MongoDB.
        
        Args:
            metadata: Dictionary with analysis metadata
            
        Returns:
            ID of the stored metadata record
        """
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
            
        # Ensure run_id is present
        if 'run_id' not in metadata:
            metadata['run_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
            
        metadata['stored_at'] = datetime.now()
        result = self.db.analysis_metadata.insert_one(metadata)
        logger.info(f"Stored analysis metadata with ID {result.inserted_id}")
        return result.inserted_id
    
    def get_latest_demand_data(self, category=None, limit=1000):
        """
        Retrieve the latest monthly demand data from MongoDB.
        
        Args:
            category: Optional category filter
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with the latest demand data
        """
        query = {}
        if category:
            query['product_category_name'] = category
            
        # Sort by stored_at in descending order to get the latest data
        cursor = self.db.demand_data.find(query).sort('stored_at', pymongo.DESCENDING).limit(limit)
        records = list(cursor)
        
        # Convert MongoDB documents to DataFrame
        if records:
            df = pd.DataFrame(records)
            # Convert string dates back to datetime objects
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            # Remove MongoDB internal _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return pd.DataFrame()
    
    def get_latest_forecasts(self, category=None, run_id=None):
        """
        Retrieve the latest forecast data from MongoDB.
        
        Args:
            category: Optional category filter
            run_id: Optional analysis run ID filter
            
        Returns:
            DataFrame with the latest forecast data
        """
        query = {}
        if category:
            query['category'] = category
        if run_id:
            query['run_id'] = run_id
            
        # Sort by stored_at in descending order to get the latest data
        cursor = self.db.forecasts.find(query).sort('stored_at', pymongo.DESCENDING)
        records = list(cursor)
        
        # Convert MongoDB documents to DataFrame
        if records:
            df = pd.DataFrame(records)
            # Parse JSON fields back to lists
            for field in ['forecast_values', 'lower_ci', 'upper_ci']:
                if field in df.columns:
                    df[field] = df[field].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            # Remove MongoDB internal _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return pd.DataFrame()
    
    def get_supplier_clusters(self, cluster=None, run_id=None):
        """
        Retrieve supplier clustering data from MongoDB.
        
        Args:
            cluster: Optional cluster filter
            run_id: Optional analysis run ID filter
            
        Returns:
            DataFrame with supplier clustering data
        """
        query = {}
        if cluster is not None:  # Allow cluster 0
            query['prediction'] = cluster
        if run_id:
            query['run_id'] = run_id
            
        cursor = self.db.suppliers.find(query)
        records = list(cursor)
        
        # Convert MongoDB documents to DataFrame
        if records:
            df = pd.DataFrame(records)
            # Remove MongoDB internal _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return pd.DataFrame()
    
    def get_inventory_recommendations(self, category=None, run_id=None):
        """
        Retrieve inventory recommendations from MongoDB.
        
        Args:
            category: Optional category filter
            run_id: Optional analysis run ID filter
            
        Returns:
            DataFrame with inventory recommendations
        """
        query = {}
        if category:
            # Check both product_category and category fields
            query['$or'] = [{'product_category': category}, {'category': category}]
        if run_id:
            query['run_id'] = run_id
            
        cursor = self.db.inventory.find(query)
        records = list(cursor)
        
        # Convert MongoDB documents to DataFrame
        if records:
            df = pd.DataFrame(records)
            # Remove MongoDB internal _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return pd.DataFrame()
    
    def get_analysis_runs(self, limit=10):
        """
        Retrieve metadata for the most recent analysis runs.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            DataFrame with analysis run metadata
        """
        cursor = self.db.analysis_metadata.find().sort('stored_at', pymongo.DESCENDING).limit(limit)
        records = list(cursor)
        
        # Convert MongoDB documents to DataFrame
        if records:
            df = pd.DataFrame(records)
            # Remove MongoDB internal _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return pd.DataFrame()
    
    def delete_run_data(self, run_id):
        """
        Delete all data associated with a specific run ID.
        
        Args:
            run_id: Analysis run ID to delete
            
        Returns:
            Dictionary with deletion counts for each collection
        """
        if not run_id:
            raise ValueError("run_id must be provided")
            
        deletion_counts = {}
        collections = ["demand_data", "forecasts", "suppliers", "inventory", "analysis_metadata"]
        
        for collection in collections:
            result = self.db[collection].delete_many({"run_id": run_id})
            deletion_counts[collection] = result.deleted_count
            
        logger.info(f"Deleted data for run_id {run_id}: {deletion_counts}")
        return deletion_counts
    
    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a sample demand dataset
    dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
    categories = ['Electronics', 'Furniture', 'Clothing']
    
    data = []
    for category in categories:
        base_demand = np.random.randint(500, 2000)
        for date in dates:
            month = date.month
            # Add some seasonality
            seasonal_factor = 1.0 + 0.2 * np.sin(month / 12.0 * 2 * np.pi)
            # Add some noise
            noise = np.random.normal(0, 0.05)
            demand = int(base_demand * seasonal_factor * (1 + noise))
            
            data.append({
                'product_category_name': category,
                'date': date,
                'year': date.year,
                'month': date.month,
                'count': demand
            })
    
    demand_df = pd.DataFrame(data)
    
    # Initialize MongoDB storage
    storage = MongoDBStorage()
    
    # Store the sample data
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Store demand data
    storage.store_monthly_demand(demand_df, run_id)
    
    # Store analysis metadata
    metadata = {
        'run_id': run_id,
        'user': 'test_user',
        'parameters': {
            'seasonal': True,
            'forecast_periods': 6,
            'top_n': 5
        },
        'execution_time_seconds': 120,
        'data_sources': ['sample_data']
    }
    storage.store_analysis_metadata(metadata)
    
    # Retrieve and display the data
    retrieved_demand = storage.get_latest_demand_data()
    print(f"Retrieved {len(retrieved_demand)} demand records")
    
    # Close the connection
    storage.close()