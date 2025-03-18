#!/usr/bin/env python3
"""
Supply Chain Analytics for Demand Forecasting
Main entry point script that orchestrates the entire analytics pipeline.

This script provides a convenient way to run the complete analysis
with configurable parameters.
"""

import os
import sys
import argparse
import subprocess
import datetime
import shutil
import pandas as pd

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Supply Chain Analytics for Demand Forecasting')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top categories to analyze')
    parser.add_argument('--forecast-periods', type=int, default=6, help='Number of periods to forecast')
    parser.add_argument('--use-auto-arima', action='store_true', help='Use auto ARIMA parameter selection')
    parser.add_argument('--seasonal', action='store_true', help='Include seasonality in the model')
    parser.add_argument('--supplier-clusters', type=int, default=3, help='Number of supplier clusters to create')
    parser.add_argument('--clean', action='store_true', help='Clean output directory before running')
    parser.add_argument('--skip-frontend', action='store_true', help='Skip frontend setup')
    parser.add_argument('--use-mongodb', action='store_true', help='Store results in MongoDB')
    parser.add_argument('--mongodb-uri', type=str, default='mongodb://localhost:27017/', 
                        help='MongoDB connection URI')
    parser.add_argument('--mongodb-db', type=str, default='supply_chain_analytics',
                        help='MongoDB database name')
    parser.add_argument('--use-spark', action='store_true', help='Use Apache Spark for data processing')

    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def clean_directory(directory):
    """Clean output directory if it exists"""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print(f"Cleaned directory: {directory}")
        except Exception as e:
            print(f"Error cleaning directory: {e}")
    ensure_directory(directory)

def check_dependencies():
    """Check if required packages are installed."""
    # Map display names to their actual importable module names
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scikit-learn': 'sklearn',  
        'statsmodels': 'statsmodels',
        'pmdarima': 'pmdarima'
    }
    
    missing_packages = []
    
    for package_display, module_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_display)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install required packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_required_files(output_dir):
    """
    Check that all required CSV files exist in the output directory.
    If any are missing, log an error and return False.
    """
    required_files = [
        "monthly_demand.csv",
        "forecast_report.csv",
        "seller_clusters.csv",
        "reorder_recommendations.csv"
    ]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            missing_files.append(file)
    if missing_files:
        print("Error: The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    return True

def run_analysis(args):
    """Run the main analysis using parsed arguments"""

    if args.use_spark:
        script_to_run = os.path.join("backend", "spark_implementation.py")
        print("Running analysis with Spark...")
        
        # For Spark, only include the arguments it recognizes
        cmd = [
            "python", 
            script_to_run,
            f"--data-dir={args.data_dir}",
            f"--output-dir={args.output_dir}",
            f"--top-n={args.top_n}",
            f"--forecast-periods={args.forecast_periods}"
        ]
        
        # Avoid adding MongoDB args to Spark implementation
        use_mongodb_after_spark = args.use_mongodb
    else:
        script_to_run = os.path.join("backend", "main.py")
        print("Running analysis with pandas...")
        
        # For standard analysis, include all arguments
        cmd = [
            "python", 
            script_to_run,
            f"--data-dir={args.data_dir}",
            f"--output-dir={args.output_dir}",
            f"--top-n={args.top_n}",
            f"--forecast-periods={args.forecast_periods}"
        ]
        
        # Add MongoDB options if specified
        if args.use_mongodb:
            cmd.append("--use-mongodb")
            cmd.append(f"--mongodb-uri={args.mongodb_uri}")
            cmd.append(f"--mongodb-db={args.mongodb_db}")
        
        use_mongodb_after_spark = False
    
    # Add common arguments
    if args.use_auto_arima:
        cmd.append("--use-auto-arima")
        
    if args.seasonal:
        cmd.append("--seasonal")
        
    if args.supplier_clusters != 3:
        cmd.append(f"--supplier-clusters={args.supplier_clusters}")
    
    # Run the command
    print("Running supply chain analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Set environment variables for MongoDB if needed
        if args.use_mongodb:
            os.environ['MONGODB_URI'] = args.mongodb_uri
            os.environ['MONGODB_DB'] = args.mongodb_db
        
        # Run analysis
        subprocess.run(cmd, check=True)
        
        # If using Spark with MongoDB, store the results in MongoDB after Spark processing
        if use_mongodb_after_spark:
            # Import after Spark processing completes
            sys.path.append('./backend')
            try:
                from mongodb_storage import MongoDBStorage
                
                # Generate a run ID
                from datetime import datetime
                run_id = datetime.now().strftime("%Y%m%d%H%M%S")
                
                # Initialize MongoDB storage
                storage = MongoDBStorage(args.mongodb_uri, args.mongodb_db)
                
                # Store metadata
                metadata = {
                    'run_id': run_id,
                    'timestamp': datetime.now(),
                    'parameters': {
                        'data_dir': args.data_dir,
                        'output_dir': args.output_dir,
                        'top_n': args.top_n,
                        'forecast_periods': args.forecast_periods,
                        'use_spark': True,
                        'use_auto_arima': args.use_auto_arima,
                        'seasonal': args.seasonal,
                        'supplier_clusters': args.supplier_clusters
                    },
                    'engine': 'spark'
                }
                storage.store_analysis_metadata(metadata)
                
                print("Storing Spark results in MongoDB...")
                
                # Load and store monthly demand
                try:
                    demand_path = os.path.join(args.output_dir, 'monthly_demand.csv')
                    if os.path.exists(demand_path):
                        demand_df = pd.read_csv(demand_path)
                        storage.store_monthly_demand(demand_df, run_id)
                        print("Stored monthly demand data in MongoDB")
                    else:
                        print(f"Warning: {demand_path} not found")
                        
                    # Load and store forecast report
                    forecast_path = os.path.join(args.output_dir, 'forecast_report.csv')
                    if os.path.exists(forecast_path):
                        forecast_df = pd.read_csv(forecast_path)
                        storage.store_forecasts(forecast_df, run_id)
                        print("Stored forecast data in MongoDB")
                    else:
                        print(f"Warning: {forecast_path} not found")
                        
                    # Load and store seller clusters
                    seller_path = os.path.join(args.output_dir, 'seller_clusters.csv')
                    if os.path.exists(seller_path):
                        seller_df = pd.read_csv(seller_path)
                        storage.store_supplier_clusters(seller_df, run_id)
                        print("Stored supplier clustering data in MongoDB")
                    else:
                        print(f"Warning: {seller_path} not found")
                        
                    # Load and store recommendations
                    recom_path = os.path.join(args.output_dir, 'reorder_recommendations.csv')
                    if os.path.exists(recom_path):
                        recom_df = pd.read_csv(recom_path)
                        storage.store_inventory_recommendations(recom_df, run_id)
                        print("Stored inventory recommendations in MongoDB")
                    else:
                        print(f"Warning: {recom_path} not found")
                    
                    print(f"Spark analysis results stored in MongoDB with run_id: {run_id}")
                except Exception as e:
                    print(f"Error storing output files in MongoDB: {e}")
                
                storage.close()
            except ImportError as e:
                print(f"MongoDB integration not available. Error: {e}")
            except Exception as e:
                print(f"Error storing Spark results in MongoDB: {e}")
                
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        return False
    
    return True
def run_supplier_analysis(args):
    """Run additional supplier analysis"""
    # Check if seller_clusters.csv exists
    seller_clusters_path = os.path.join(args.output_dir, "seller_clusters.csv")
    if not os.path.exists(seller_clusters_path):
        print("Seller clusters data not found. Skipping detailed supplier analysis.")
        return False
    
    print("Running enhanced supplier analysis...")
    
    # Create a Python script to run on the fly
    script = f"""
import sys
sys.path.append('backend')
from supplier_analyzer import SupplierAnalyzer

analyzer = SupplierAnalyzer("{seller_clusters_path}")
analyzer.n_clusters = {args.supplier_clusters}
results = analyzer.run_analysis("{args.output_dir}")

print("Supplier analysis complete.")
"""
    
    try:
        subprocess.run([sys.executable, "-c", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running supplier analysis: {e}")
        return False
    
    return True

def setup_frontend(args):
    """Set up the frontend for visualization by copying output files"""
    if args.skip_frontend:
        print("Skipping frontend setup as requested.")
        return True
    
    # Create frontend data directory if it doesn't exist
    frontend_data_dir = os.path.join("frontend", "public", "data")
    ensure_directory(frontend_data_dir)
    
    # Copy all CSV and image files from output to frontend data directory
    try:
        print(f"Copying output files to {frontend_data_dir}...")
        for filename in os.listdir(args.output_dir):
            if filename.endswith(('.csv', '.png', '.jpg', '.md')):
                source = os.path.join(args.output_dir, filename)
                destination = os.path.join(frontend_data_dir, filename)
                shutil.copy2(source, destination)
                print(f"  Copied {filename}")
        
        print("All output files copied to frontend data directory")
    except Exception as e:
        print(f"Error setting up frontend: {e}")
        return False
    
    return True

def create_summary(args):
    """Create and print a summary of the analysis"""
    print("\n" + "="*80)
    print("SUPPLY CHAIN ANALYTICS SUMMARY")
    print("="*80)
    
    # Get timestamp for the report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Analysis completed: {timestamp}")
    
    # Count generated files
    csv_files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv')]
    image_files = [f for f in os.listdir(args.output_dir) if f.endswith(('.png', '.jpg'))]
    
    print(f"Data files generated: {len(csv_files)}")
    print(f"Visualization files generated: {len(image_files)}")
    
    # List key files
    key_files = [
        "forecast_report.csv",
        "seller_clusters.csv",
        "reorder_recommendations.csv",
        "state_metrics.csv",
        "supplier_recommendations.csv",
        "summary_report.md"
    ]
    
    print("\nKey output files:")
    for file in key_files:
        path = os.path.join(args.output_dir, file)
        if os.path.exists(path):
            print(f"  - {file} ({os.path.getsize(path)} bytes)")
        else:
            print(f"  - {file} [NOT FOUND]")
    
    print("\nOutput directory:")
    print(f"  {os.path.abspath(args.output_dir)}")
    
    # Add MongoDB info if used
    if args.use_mongodb:
        print("\nMongoDB Storage:")
        print(f"  Database: {args.mongodb_db}")
        print(f"  Connection URI: {args.mongodb_uri.split('@')[-1] if '@' in args.mongodb_uri else args.mongodb_uri}")
        print("  Collections: demand_data, forecasts, suppliers, inventory, analysis_metadata")
    
    print("\nNext steps:")
    print("  1. Review the summary report: output/summary_report.md")
    print("  2. Explore visualizations in the output directory")
    print("  3. Use the frontend for interactive analysis:")
    print("     cd frontend && npm start")
    if args.use_mongodb:
        print("  4. Access your data in MongoDB Atlas dashboard")
    
    print("="*80)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Clean output directory if requested
    if args.clean:
        clean_directory(args.output_dir)
    else:
        ensure_directory(args.output_dir)
    
    # Run main analysis
    if not run_analysis(args):
        return 1
    
    # Check that all required output files were generated
    if not check_required_files(args.output_dir):
        print("Error: Required output files are missing. Please check the analysis logs.")
        return 1
    
    # Run additional supplier analysis
    run_supplier_analysis(args)
    
    # Set up frontend
    setup_frontend(args)
    
    # Create and print summary
    create_summary(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
