# Supply Chain Analytics for Demand Forecasting

This project uses big data to analyze and optimize supply chain operations. It processes order histories, supplier information, and logistics tracking data to forecast demand and identify bottlenecks, using predictive modeling to suggest optimal reorder strategies.

## Project Structure

The project consists of the following components:

### Backend (Python)
- **main.py**: Main application script that orchestrates the entire analytics pipeline
- **arima_forecasting.py**: Implementation of time series forecasting using ARIMA models
- **spark_implementation.py**: Distributed processing using Apache Spark for large datasets
- **visualization_module.py**: Creates visualizations of supply chain data

### Frontend (React)
- React-based dashboard for visualizing supply chain analytics
- Components for time series visualization, product category analysis, seller performance, and recommendations

### API Server (Node.js/Express)
- Serves data processed by the Python backend to the React frontend
- Provides endpoints for detailed analysis and forecasts

## Getting Started

### Prerequisites

#### Backend
- Python 3.7+
- Required Python packages: pandas, numpy, matplotlib, scikit-learn, statsmodels, seaborn
- Optional: Apache Spark for large dataset processing

#### Frontend
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/supply-chain-analytics.git
cd supply-chain-analytics
```

2. Install Python dependencies
```
pip install -r requirements.txt
```

3. Install Node.js dependencies
```
cd frontend
npm install
```

4. Install API server dependencies
```
cd ../server
npm install
```

### Running the Application

1. Process the supply chain data using the Python backend
```
python main.py --data-dir=./data --output-dir=./output --top-n=5 --forecast-periods=6
```

2. Start the API server
```
cd server
node server.js
```

3. Start the React frontend (in a new terminal)
```
cd frontend
npm start
```

4. Access the dashboard at http://localhost:3000

## Project Implementation Details

### Data Processing

The system can process supply chain data using either pandas (for smaller datasets) or Apache Spark (for larger datasets), determined by the `--use-spark` flag.

### Time Series Forecasting

The ARIMA forecasting module:
- Preprocesses time series data
- Tests data stationarity
- Determines optimal ARIMA parameters
- Trains models and generates forecasts
- Evaluates model performance using MAE, RMSE, and MAPE metrics

### Supply Chain Analysis

The system performs various analyses:
- Demand patterns by product category
- Seller performance clustering
- Geographical order patterns
- Calculation of supply chain KPIs (processing time, delivery rates, etc.)
- Generation of reorder recommendations

### Visualization

The visualization module creates:
- Demand trend charts
- Category performance visualizations
- Seller performance clusters
- Reorder recommendation charts
- Forecast accuracy evaluations

## Dashboard Features

The React-based dashboard provides:
- **Overview**: KPIs and high-level metrics
- **Demand Forecasting**: Time series forecasts with confidence intervals
- **Product Categories**: Analysis of top-performing categories
- **Seller Performance**: Clustering and analysis of seller efficiency
- **Geographical Analysis**: Order patterns by location
- **Recommendations**: Inventory optimization suggestions

## Deployment

For production deployment:

1. Build the React frontend
```
cd frontend
npm run build
```

2. Configure the server for production (update API endpoints, enable compression, etc.)

3. Serve the built frontend files and run the API server

## Contributors

- Your Name

## License

This project is licensed under the MIT License - see the LICENSE file for details.