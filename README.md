# Supply Chain Analytics for Demand Forecasting

This project uses big data analytics techniques to optimize supply chain operations through demand forecasting, supplier performance analysis, and inventory optimization recommendations.

## Features

- **Demand Forecasting**: Advanced time-series forecasting using ARIMA/SARIMA models
- **Supplier Analysis**: Performance clustering and recommendations
- **Inventory Optimization**: Data-driven safety stock and reorder point calculations
- **Geographical Insights**: Regional analysis of order patterns
- **Interactive Dashboard**: Visualize key metrics and trends
- **Big Data Support**: Spark integration for large-scale datasets

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ (for frontend)
- Required Python packages (see `requirements.txt`)
- Apache Spark (optional, for large datasets)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nhiennh/supply-chain-analytics.git
cd supply-chain-analytics
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up the frontend (optional):
```bash
cd frontend
npm install
cd ..
```

### Running the Analysis

Use the main analysis script with your data:

```bash
python run_analysis.py --data-dir ./data --output-dir ./output
```

Additional options:
```
--top-n 15               # Number of top categories to analyze
--forecast-periods 6     # Number of periods to forecast
--use-auto-arima         # Enable automatic ARIMA parameter selection
--seasonal               # Include seasonality in forecasting models
--supplier-clusters 3    # Number of supplier clusters to create
--clean                  # Clean output directory before running
```

For large datasets, use the Spark implementation:

```bash
python -c "from backend.spark_implementation import run_spark_analysis; run_spark_analysis(data_dir='./data', output_dir='./output')"
```

### Viewing Results

1. Check the summary report:
```bash
cat output/summary_report.md
```

2. Start the frontend dashboard (optional):
```bash
cd frontend
npm start
```

3. Open `http://localhost:3000` in your browser to access the interactive dashboard

## Data Requirements

The analysis expects the following CSV files in the data directory:

- `df_Orders.csv` - Order transactions
- `df_OrderItems.csv` - Items within orders
- `df_Customers.csv` - Customer information
- `df_Products.csv` - Product details
- `df_Payments.csv` - Payment information (optional)

## Output Files

The analysis generates various outputs in the specified output directory:

- `forecast_report.csv` - Demand forecasts with accuracy metrics
- `reorder_recommendations.csv` - Inventory optimization recommendations
- `seller_clusters.csv` - Supplier performance clustering results
- `state_metrics.csv` - Geographical analysis
- `performance_metrics.csv` - Overall supply chain performance
- `summary_report.md` - Executive summary of findings
- Multiple visualization files (PNG format)

## Analysis Methodology

### Demand Forecasting

The ARIMA forecasting process:
1. Time series preprocessing and stationarity testing
2. Automatic parameter selection to find optimal (p,d,q) values
3. Model validation using train/test splitting
4. Forecast generation with confidence intervals
5. Performance evaluation (MAPE, RMSE, MAE)

### Supplier Clustering

The supplier analysis process:
1. Feature engineering from order and delivery metrics
2. Standardization and dimensionality reduction
3. K-means clustering with optimal cluster determination
4. Performance labeling (High, Medium, Low)
5. Recommendation generation based on cluster characteristics

### Inventory Optimization

The inventory recommendation process:
1. Demand aggregation and variability analysis
2. Safety stock calculation based on service level
3. Lead time estimation from historical data
4. Reorder point determination
5. Order frequency optimization using EOQ principles

## Architecture

The project follows a modular architecture:

- **Data Processing Layer**: Handles data loading, cleaning, and integration
- **Analytics Engine**: Performs core analytical operations
- **Forecasting Engine**: Implements time series forecasting algorithms
- **Visualization Layer**: Creates visual representations of insights
- **Web Dashboard**: Provides interactive access to analytics results

For more details, see the `ARCHITECTURE.md` document.

## Extending the Project

### Adding New Data Sources

To incorporate additional data:
1. Add data preprocessing in `data_preprocessing.py`
2. Update the unified dataset creation in `main.py`
3. Enhance analysis and visualization as needed

### Customizing Forecasting Models

To modify forecasting approach:
1. Update `arima_forecasting.py` with new models or parameters
2. Add logic for model selection based on data characteristics
3. Extend validation methods as needed

### Adding New Visualizations

To create additional visualizations:
1. Add new methods to `visualization_module.py`
2. Update `main.py` to call these methods
3. Enhance frontend components if needed

## Big Data Processing

For large datasets, the project includes Spark integration:

1. Use `spark_implementation.py` instead of the standard processing pipeline
2. Configure Spark settings for your environment in the script
3. Run the analysis as shown above with the Spark implementation

## Error Handling

The project includes robust error handling:

1. Input validation with comprehensive data checks
2. Graceful degradation with fallback strategies
3. Detailed logging for troubleshooting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of the Big Data Analytics course
- Uses various open-source libraries including Pandas, NumPy, Scikit-learn, and pmdarima


## MongoDB Integration

This project stores analysis results in MongoDB for persistent storage. To view and explore your data:

1. **Log into your MongoDB Atlas account**
2. Navigate to your cluster
3. Click "Browse Collections" to view stored analysis results
4. Use MongoDB Compass for more advanced exploration and querying


## Prerequisites

1. **MongoDB Server**: You'll need either:
   - A local MongoDB server installed on your machine (recommended for development)
   - A MongoDB Atlas account for cloud-based storage
   - Another MongoDB-compatible database service

2. **MongoDB Python Driver and Node.js Driver**:
   ```bash
   # For Python backend
   pip install pymongo

   # For Node.js server
   npm install mongodb mongoose --save
   ```

## Configuration

### MongoDB Connection String

The application uses a MongoDB connection string in the format:
```
mongodb://[username:password@]host[:port]/[database]
```

You can specify this connection string in several ways:
1. Command line argument: `--mongodb-uri`
2. Environment variable: `MONGODB_URI`
3. Default: `mongodb://localhost:27017/` (local MongoDB server)


### MongoDB Collections
- `demand_data` - Monthly demand data by product category
- `forecasts` - Time series forecasting results
- `suppliers` - Supplier clustering and performance analysis
- `inventory` - Inventory optimization recommendations
- `analysis_metadata` - Metadata about each analysis ru

## Running Analysis with MongoDB Storage

To store analysis results in MongoDB, use the `--use-mongodb` flag when running the analysis:

```bash
python run_analysis.py --data-dir=./data --output-dir=./output --use-mongodb
```

Additional MongoDB options:
```bash
python run_analysis.py --use-mongodb --mongodb-uri="mongodb+srv://username:password@cluster.mongodb.net/" --mongodb-db="my_supply_chain"
```

## Accessing MongoDB Data from the Frontend

The application includes a MongoDB Explorer page that allows you to:
- View all analysis runs stored in MongoDB
- Inspect detailed data for each run
- Compare data across multiple runs
- Delete outdated analysis runs

To access the MongoDB Explorer, run the frontend application and navigate to the "MongoDB Explorer" tab in the sidebar.

## MongoDB REST API Endpoints

The application provides the following REST API endpoints for MongoDB data:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mongo/runs` | GET | Get list of all analysis runs |
| `/api/mongo/run/:runId` | GET | Get detailed data for a specific run |
| `/api/mongo/run/:runId` | DELETE | Delete a specific run and all associated data |
| `/api/mongo/latest` | GET | Get the most recent analysis data |
| `/api/mongo/forecasts/:category` | GET | Get forecasts for a specific product category |

## Data Model

### Analysis Metadata
```json
{
  "run_id": "20231215123045",
  "timestamp": "2023-12-15T12:30:45.123Z",
  "parameters": {
    "data_dir": "./data",
    "output_dir": "./output",
    "top_n": 15,
    "forecast_periods": 6,
    "use_auto_arima": true,
    "seasonal": true,
    "supplier_clusters": 3
  },
  "execution_environment": {
    "platform": "linux",
    "python_version": "3.8.10"
  }
}
```

### Demand Data
```json
{
  "product_category_name": "Electronics",
  "date": "2023-01-01T00:00:00Z",
  "year": 2023,
  "month": 1,
  "count": 1250,
  "run_id": "20231215123045",
  "stored_at": "2023-12-15T12:30:45.123Z"
}
```

### Forecasts
```json
{
  "category": "Electronics",
  "forecast_values": [1350, 1400, 1450, 1500, 1550, 1600],
  "lower_ci": [1200, 1220, 1240, 1260, 1280, 1300],
  "upper_ci": [1500, 1580, 1660, 1740, 1820, 1900],
  "growth_rate": 12.5,
  "run_id": "20231215123045",
  "stored_at": "2023-12-15T12:30:45.123Z"
}
```

## Troubleshooting

### Connection Issues
If you have trouble connecting to MongoDB:
1. Ensure MongoDB server is running
2. Check connection string format
3. Verify network connectivity and firewall settings
4. For Atlas, ensure your IP is in the allowlist

### Data Issues
If data is not appearing in MongoDB:
1. Check MongoDB logs for errors
2. Verify database and collection names
3. Ensure data is not being filtered out during processing

## Design Considerations

The MongoDB integration follows these design principles:

1. **Fail Gracefully**: The application will work without MongoDB if unavailable
2. **Run-Based Organization**: All data is grouped by analysis run for easy comparison
3. **Minimal Schema Enforcement**: The data model is flexible to accommodate changing requirements
4. **Comprehensive Metadata**: Each run includes detailed metadata for reproducibility
5. **Efficient Queries**: Data is structured for common query patterns

## Future Enhancements

Potential improvements to the MongoDB integration:

1. **Aggregation Pipelines**: Add advanced MongoDB aggregation for cross-run analytics
2. **Change Streams**: Implement real-time updates via MongoDB change streams
3. **Time Series Collections**: Use MongoDB 5.0+ time series collections for time-based data
4. **Schema Validation**: Add JSON schema validation for data integrity
5. **Data Expiration**: Implement TTL indexes for automatic data cleanup