# Supply Chain Analytics for Demand Forecasting

This project uses big data analytics techniques to optimize supply chain operations through demand forecasting, supplier performance analysis, and inventory optimization recommendations.

## Features

- **Demand Forecasting**: Advanced time-series forecasting using ARIMA/SARIMA models
- **Supplier Analysis**: Performance clustering and recommendations
- **Inventory Optimization**: Data-driven safety stock and reorder point calculations
- **Geographical Insights**: Regional analysis of order patterns
- **Interactive Dashboard**: Visualize key metrics and trends

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ (for frontend)
- Required Python packages (see `requirements.txt`)

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
- `df_Payments.csv` - Payment information

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

## Project Structure

```
supply-chain-analytics/
├── README.md                    # Project documentation
├── requirements.txt             # Required Python packages
├── run_analysis.py              # Main entry point script
├── backend/                     # Backend code
│   ├── arima_forecasting.py     # Enhanced ARIMA forecasting module
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── main.py                  # Analysis orchestration
│   ├── supplier_analyzer.py     # Supplier clustering and analysis
│   ├── visualization_module.py  # Visualization utilities
│   └── spark_implementation.py  # Optional Spark-based analysis
├── frontend/                    # Frontend application
│