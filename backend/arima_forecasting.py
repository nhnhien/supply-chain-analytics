import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class DemandForecaster:
    """
    Class for demand forecasting using ARIMA models
    """
    def __init__(self, data_path="demand_by_category.csv"):
        """
        Initialize the forecaster with data
        
        Args:
            data_path: Path to CSV file with demand data
        """
        self.data = pd.read_csv(data_path)
        self.categories = self.data['product_category_name'].unique()
        self.models = {}
        self.forecasts = {}
        self.performance = {}
        self.best_params = {}
        self.count_column = None
        
    def _get_count_column(self, category):
        """Helper to find the appropriate count column for a category"""
        # Find the count column
        count_column = None
        possible_count_columns = ['order_count', 'count', 'order_id_count', 'count(order_id)']
        
        for col in possible_count_columns:
            if col in self.category_data[category].columns:
                count_column = col
                break
        
        if count_column is None:
            # If no specific count column is found, use the first numeric column
            numeric_cols = self.category_data[category].select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                count_column = numeric_cols[0]
        
        return count_column
        
    def preprocess_data(self):
        """
        Preprocess data for time series analysis
        """
        # Check if we need to extract year and month
        if 'year' not in self.data.columns or 'month' not in self.data.columns:
            # Convert timestamps to datetime if they exist
            date_columns = [col for col in self.data.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
            
            if date_columns:
                # Use the first date column found
                date_col = date_columns[0]
                print(f"Using {date_col} to extract year and month")
                
                # Convert to datetime
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                
                # Extract year and month
                self.data['year'] = self.data[date_col].dt.year
                self.data['month'] = self.data[date_col].dt.month
                
                # Create proper date column
                self.data['date'] = pd.to_datetime(
                    self.data['year'].astype(str) + '-' + 
                    self.data['month'].astype(str).str.zfill(2) + '-01'
                )
            else:
                # If no date column found, attempt to use year and month columns directly
                print("No timestamp columns found, using order_year and order_month if available")
                year_col = next((col for col in self.data.columns if 'year' in col.lower()), None)
                month_col = next((col for col in self.data.columns if 'month' in col.lower()), None)
                
                if year_col and month_col:
                    self.data['year'] = self.data[year_col]
                    self.data['month'] = self.data[month_col]
                    self.data['date'] = pd.to_datetime(
                        self.data['year'].astype(str) + '-' + 
                        self.data['month'].astype(str).str.zfill(2) + '-01'
                    )
                else:
                    # Last resort - create date from row index
                    print("WARNING: No date columns found. Using index as date reference.")
                    self.data['date'] = pd.date_range(start='2021-01-01', periods=len(self.data), freq='M')
                    self.data['year'] = self.data['date'].dt.year
                    self.data['month'] = self.data['date'].dt.month
        else:
            # Create date from existing year and month columns
            self.data['date'] = pd.to_datetime(
                self.data['year'].astype(str) + '-' + 
                self.data['month'].astype(str).str.zfill(2) + '-01'
            )
        
        # Create category-specific datasets
        self.category_data = {}
        # Include all categories with at least 1 data point
        for category in self.categories:
            if pd.isna(category):
                continue
                
            category_df = self.data[self.data['product_category_name'] == category].copy()
            category_df.sort_values('date', inplace=True)
            category_df.set_index('date', inplace=True)
            
            # Add all categories to the data dictionary
            self.category_data[category] = category_df
            
        # Determine the count column for the dataset
        possible_count_columns = ['order_count', 'count', 'order_id_count', 'count(order_id)']
        
        for category in self.category_data:
            for col in possible_count_columns:
                if col in self.category_data[category].columns:
                    self.count_column = col
                    print(f"Using '{self.count_column}' as count column")
                    break
            if self.count_column:
                break
                
        if not self.count_column:
            # If no specific count column is found, use the first numeric column
            for category in self.category_data:
                numeric_cols = self.category_data[category].select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    self.count_column = numeric_cols[0]
                    print(f"No standard count column found. Using '{self.count_column}' as count column")
                    break
                    
        print(f"Processed {len(self.category_data)} categories")
        
    def test_stationarity(self, category):
        """
        Test if a time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            category: Product category to test
            
        Returns:
            Dictionary with test results
        """
        if category not in self.category_data:
            return {"error": "Category not found"}
            
        count_column = self._get_count_column(category)
        if not count_column:
            return {"error": "No suitable numeric column found"}
            
        ts = self.category_data[category][count_column]
        
        # Check if there's enough data for the test
        if len(ts) < 4:  # ADF test requires at least 4 data points
            return {
                "adf_statistic": None,
                "p_value": None,
                "critical_values": None,
                "is_stationary": False,
                "error": "Insufficient data for stationarity test"
            }
        
        try:
            # Run ADF test
            result = adfuller(ts.dropna())
            
            return {
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < 0.05
            }
        except Exception as e:
            print(f"Error in stationarity test for {category}: {e}")
            return {
                "adf_statistic": None,
                "p_value": None,
                "critical_values": None,
                "is_stationary": False,
                "error": str(e)
            }
        
    def plot_time_series(self, category, figsize=(12, 8)):
        """
        Plot time series data with trend and seasonality analysis
        
        Args:
            category: Product category to plot
            figsize: Figure size tuple (width, height)
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return
            
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
            
        ts = self.category_data[category][count_column]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
        fig.suptitle(f'Time Series Analysis: {category}', fontsize=16)
        
        # Original time series
        axes[0].plot(ts.index, ts.values)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Order Count')
        axes[0].grid(True)
        
        # Try to decompose the time series
        try:
            # We need at least 2 full seasonal cycles for decomposition
            if len(ts) >= 24:  # 2 years of monthly data
                decomposition = seasonal_decompose(ts, model='additive', period=12)
                
                # Plot trend
                axes[1].plot(decomposition.trend.index, decomposition.trend.values)
                axes[1].set_title('Trend Component')
                axes[1].set_ylabel('Trend')
                axes[1].grid(True)
                
                # Plot seasonality
                axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
                axes[2].set_title('Seasonal Component')
                axes[2].set_ylabel('Seasonality')
                axes[2].grid(True)
            else:
                # Not enough data for seasonal decomposition
                axes[1].text(0.5, 0.5, 'Insufficient data for trend analysis', 
                           horizontalalignment='center', verticalalignment='center')
                axes[2].text(0.5, 0.5, 'Insufficient data for seasonality analysis', 
                           horizontalalignment='center', verticalalignment='center')
                
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
            axes[1].text(0.5, 0.5, f'Error in trend analysis: {e}', 
                       horizontalalignment='center', verticalalignment='center')
            axes[2].text(0.5, 0.5, f'Error in seasonality analysis: {e}', 
                       horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'{category}_time_series_analysis.png')
        plt.close()
    
    def plot_acf_pacf(self, category, figsize=(12, 8)):
        """
        Plot ACF and PACF to determine ARIMA parameters
        
        Args:
            category: Product category to analyze
            figsize: Figure size tuple (width, height)
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return
            
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
            
        ts = self.category_data[category][count_column]
        
        # Check if we have enough data
        if len(ts) < 4:
            print(f"Insufficient data for ACF/PACF analysis for {category}")
            return
        
        # Check stationarity
        stationarity_result = self.test_stationarity(category)
        
        # Apply differencing if not stationary
        if not stationarity_result.get("is_stationary", False):
            ts_diff = ts.diff().dropna()
            title_suffix = " (After First Differencing)"
        else:
            ts_diff = ts
            title_suffix = " (Original Series)"
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'ACF and PACF for {category}{title_suffix}', fontsize=16)
        
        try:
            # Plot ACF
            max_lags = min(24, len(ts_diff) // 2 - 1)  # Max lags = 50% of series length - 1
            if max_lags > 0:  # Ensure we have at least 1 lag
                plot_acf(ts_diff, ax=ax1, lags=max_lags)
                ax1.set_title('Autocorrelation Function')
                
                # Plot PACF
                plot_pacf(ts_diff, ax=ax2, lags=max_lags)
                ax2.set_title('Partial Autocorrelation Function')
            else:
                ax1.text(0.5, 0.5, 'Insufficient data for ACF', 
                       horizontalalignment='center', verticalalignment='center')
                ax2.text(0.5, 0.5, 'Insufficient data for PACF', 
                       horizontalalignment='center', verticalalignment='center')
        except Exception as e:
            print(f"Error plotting ACF/PACF for {category}: {e}")
            ax1.text(0.5, 0.5, f'Error plotting ACF: {e}', 
                   horizontalalignment='center', verticalalignment='center')
            ax2.text(0.5, 0.5, f'Error plotting PACF: {e}', 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'{category}_acf_pacf.png')
        plt.close()

    def find_best_parameters(self, category, max_p=3, max_d=2, max_q=3):
        """
        Find the best ARIMA parameters using AIC criterion
        
        Args:
            category: Product category to analyze
            max_p: Maximum value for p (AR parameter)
            max_d: Maximum value for d (differencing parameter)
            max_q: Maximum value for q (MA parameter)
            
        Returns:
            Tuple with best parameters (p, d, q)
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return None
        
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return (1, 1, 1)  # Default parameters
            
        ts = self.category_data[category][count_column]
        
        best_aic = float('inf')
        best_params = None
        
        # Check if we have enough data for modeling
        if len(ts) < 6:  # Minimum data points for reliable parameter search
            print(f"Insufficient data for {category} to find optimal parameters, using default")
            return (1, 1, 1)  # Default parameters
        
        # Grid search for best parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    # Skip if p=0 and q=0 (meaningless model)
                    if p == 0 and q == 0:
                        continue
                        
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        results = model.fit()
                        
                        # Get AIC
                        aic = results.aic
                        
                        # Update best parameters if current model is better
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                            
                    except Exception as e:
                        # Skip if model fails to converge
                        continue
        
        if best_params is None:
            print(f"Could not find optimal parameters for {category}, using default")
            best_params = (1, 1, 1)
            
        print(f"Best ARIMA parameters for {category}: {best_params} (AIC: {best_aic:.2f})")
        self.best_params[category] = best_params
        return best_params
    
    def train_and_forecast(self, category, periods=6, use_best_params=True):
        """
        Train ARIMA model and generate forecasts with improved visualization data
        
        Args:
            category: Product category to forecast
            periods: Number of periods to forecast
            use_best_params: Whether to use grid search for best parameters
            
        Returns:
            Dictionary with forecast results, including visualization data
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return None
        
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return None
                
        ts = self.category_data[category][count_column]
        
        # Prepare historical data for visualization
        historical_data = []
        for date, value in zip(ts.index, ts.values):
            historical_data.append({
                'date': date,
                'value': float(value),
                'type': 'historical'
            })
        
        # Check if we have enough data for forecasting (at least 4 points)
        if len(ts) < 4:
            print(f"Insufficient data points for {category}, need at least 4 data points")
            result = self._create_empty_forecast_results(category, ts)
            # Add visualization data
            result['visualization_data'] = historical_data
            return result
        
        # Determine ARIMA parameters
        if use_best_params:
            if category not in self.best_params:
                params = self.find_best_parameters(category)
            else:
                params = self.best_params[category]
        else:
            # Default parameters
            params = (1, 1, 1)
            
        # If d=2, we need at least 3+d data points, so adjust if necessary
        if params[1] > 1 and len(ts) < (3 + params[1]):
            print(f"Insufficient data for differencing with d={params[1]}, reducing d parameter")
            params = (params[0], 1, params[2])  # Reduce d to 1
        
        # Train-test split for validation (80-20 split)
        train_size = max(int(len(ts) * 0.8), 3)  # Ensure at least 3 data points for training
        test_size = len(ts) - train_size
        
        # Initialize metrics
        mae, rmse, mape = None, None, None
        
        # Only perform validation if we have enough data
        if test_size > 0:
            train, test = ts[:train_size], ts[train_size:]
            
            try:
                # Train model on training data
                model = ARIMA(train, order=params)
                model_fit = model.fit()
                
                # Store the model
                self.models[category] = model_fit
                
                # Validate on test data
                forecast = model_fit.forecast(steps=len(test))
                
                # Calculate error metrics with robust handling
                try:
                    # Make sure forecast and test have the same index before calculation
                    test_values = test.values if hasattr(test, 'values') else test
                    forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
                    
                    if len(test_values) == len(forecast_values):
                        mae = mean_absolute_error(test_values, forecast_values)
                        rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
                        
                        # Robust MAPE calculation that handles zeros
                        # Calculate MAPE only on non-zero actual values
                        non_zero_mask = np.array(test_values) > 0
                        
                        if np.any(non_zero_mask):
                            # Only calculate MAPE where actual values are non-zero
                            actual_non_zero = np.array(test_values)[non_zero_mask]
                            forecast_non_zero = np.array(forecast_values)[non_zero_mask]
                            
                            # Calculate percentage errors
                            percent_errors = np.abs((actual_non_zero - forecast_non_zero) / actual_non_zero)
                            
                            # Mean of percentage errors
                            mape = np.mean(percent_errors) * 100
                        else:
                            print(f"Warning: Cannot calculate MAPE for {category} as actual values contain zeros")
                            mape = None
                    else:
                        print(f"Warning: Length mismatch between test and forecast for {category}")
                        mae, rmse, mape = None, None, None
                except Exception as e:
                    print(f"Error calculating metrics for {category}: {e}")
                    mae, rmse, mape = None, None, None
            except Exception as e:
                print(f"Error in model validation for {category}: {e}")
                # Continue with forecasting despite validation error
        else:
            print(f"Warning: Insufficient data for validation split for {category}, proceeding with full data training")
        
        # Store performance metrics
        self.performance[category] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        # Train final model on all data
        try:
            final_model = ARIMA(ts, order=params)
            final_model_fit = final_model.fit()
            
            # Generate forecast
            forecast_values = final_model_fit.forecast(steps=periods)
            
            # Ensure forecast values are positive (replace negative values with 0)
            forecast_values = np.maximum(forecast_values, 0)
            
            # Carefully create forecast index
            try:
                if isinstance(ts.index[-1], pd.Timestamp):
                    # Handle timestamp index - ensure proper frequency
                    last_date = ts.index[-1]
                    # Create a proper date range with monthly frequency
                    # Use MS (month start) to ensure proper monthly intervals
                    forecast_index = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
                else:
                    # Handle numeric index
                    try:
                        last_idx = int(ts.index[-1])
                        forecast_index = range(last_idx + 1, last_idx + periods + 1)
                    except:
                        # If conversion fails, use simple range
                        forecast_index = range(periods)
            except Exception as e:
                print(f"Error creating forecast index for {category}: {e}")
                # Fallback to simple numeric index
                forecast_index = range(periods)
            
            # Create forecast DataFrame with safe values
            try:
                # Ensure forecast values are valid numbers
                forecast_values = pd.Series(forecast_values).fillna(ts.mean()).values
                
                forecast_df = pd.DataFrame({
                    'forecast': forecast_values
                }, index=forecast_index)
                
                # Add confidence intervals
                try:
                    forecast_with_ci = final_model_fit.get_forecast(periods)
                    conf_int = forecast_with_ci.conf_int()
                    # Ensure lower CI is positive
                    forecast_df['lower_ci'] = np.maximum(conf_int.iloc[:, 0].values, 0)
                    forecast_df['upper_ci'] = conf_int.iloc[:, 1].values
                except Exception as e:
                    print(f"Warning: Could not calculate confidence intervals for {category}: {e}")
                    # Use a simple estimate based on RMSE if available
                    if rmse is not None:
                        forecast_df['lower_ci'] = np.maximum(forecast_values - 1.96 * rmse, 0)
                        forecast_df['upper_ci'] = forecast_values + 1.96 * rmse
                    else:
                        # Use a percentage of the forecast values (30% interval)
                        forecast_df['lower_ci'] = np.maximum(forecast_values * 0.7, 0)
                        forecast_df['upper_ci'] = forecast_values * 1.3
                
                # Store forecast
                self.forecasts[category] = forecast_df
                
                # Create visualization data for forecasts
                forecast_viz_data = []
                for date, value, lower, upper in zip(forecast_index, 
                                                forecast_values, 
                                                forecast_df['lower_ci'], 
                                                forecast_df['upper_ci']):
                    forecast_viz_data.append({
                        'date': date,
                        'value': float(value),
                        'lowerBound': float(lower),
                        'upperBound': float(upper),
                        'type': 'forecast'
                    })
                
                # Combine historical and forecast data for visualization
                visualization_data = historical_data + forecast_viz_data
                
                return {
                    'model': final_model_fit,
                    'forecast': forecast_df,
                    'params': params,
                    'performance': self.performance[category],
                    'visualization_data': visualization_data
                }
            except Exception as e:
                print(f"Error creating forecast DataFrame for {category}: {e}")
                result = self._create_empty_forecast_results(category, ts)
                # Add visualization data
                result['visualization_data'] = historical_data
                return result
                
        except Exception as e:
            print(f"Error in final forecasting for {category}: {e}")
            result = self._create_empty_forecast_results(category, ts)
            # Add visualization data
            result['visualization_data'] = historical_data
            return result
    
    def _create_empty_forecast_results(self, category, ts):
        """
        Create empty forecast results when forecasting fails, with improved interpolation
        
        Args:
            category: Product category
            ts: Time series data
        
        Returns:
            Dictionary with minimal forecast results and interpolated forecast
        """
        # Get basic statistics from the time series
        # Use absolute value to ensure positive value
        avg_value = abs(ts.mean()) if not ts.empty else max(ts.max(), 1) if not ts.empty else 1
        
        # Create an empty forecast with at least the average value
        periods = 6  # Default forecast horizon
        
        # Create forecast index carefully
        try:
            if isinstance(ts.index[-1], pd.Timestamp) and not ts.empty:
                # Handle timestamp index - ensure proper frequency
                last_date = ts.index[-1]
                # Create a proper date range with monthly frequency starting after the last date
                # MS = month start frequency
                forecast_index = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
            else:
                # Handle numeric index or empty time series
                if not ts.empty:
                    try:
                        last_idx = int(ts.index[-1])
                        forecast_index = range(last_idx + 1, last_idx + periods + 1)
                    except:
                        # If conversion fails, use simple range
                        forecast_index = range(periods)
                else:
                    forecast_index = range(periods)
        except Exception as e:
            print(f"Error creating fallback forecast index for {category}: {e}")
            # Ultimate fallback - simple range
            forecast_index = range(periods)
        
        # Attempt to generate a simple trend-based forecast if we have enough data
        if len(ts) >= 2:
            # Calculate a simple trend from the last few points
            # Take up to the last 6 points or all available points
            last_n = min(6, len(ts))
            last_points = ts[-last_n:]
            
            # Calculate slope using simple linear regression
            x = np.arange(len(last_points))
            y = np.array(last_points)
            
            # Calculate the trend slope
            try:
                slope, _ = np.polyfit(x, y, 1)
            except:
                # If regression fails, assume flat trend
                slope = 0
            
            # Generate forecast values based on trend
            last_value = ts.iloc[-1] if hasattr(ts, 'iloc') else ts[-1]
            forecast_values = [max(last_value + slope * (i+1), 0.1) for i in range(periods)]
        else:
            # If we don't have enough data, use a simple average-based forecast
            # Use a slightly reduced avg_value to avoid projecting growth with limited data
            forecast_values = [max(avg_value * 0.95, 1)] * periods  # Slight reduction, minimum 1
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'lower_ci': [max(value * 0.7, 0) for value in forecast_values],  # Simple 30% lower bound
            'upper_ci': [value * 1.3 for value in forecast_values]   # Simple 30% upper bound
        }, index=forecast_index)
        
        # Store the forecast
        self.forecasts[category] = forecast_df
        
        # Calculate a simple RMSE based on the standard deviation
        rmse = ts.std() if len(ts) > 1 else None
        
        # Store minimal performance metrics
        self.performance[category] = {
            'mae': None,
            'rmse': rmse,
            'mape': None
        }
        
        return {
            'model': None,
            'forecast': forecast_df,
            'params': self.best_params.get(category, (1, 1, 1)),
            'performance': self.performance[category],
            'is_fallback': True  # Flag that this is a fallback forecast
        }
    
    def run_all_forecasts(self, top_n=5, periods=6):
        """
        Run forecasts for top N categories
        
        Args:
            top_n: Number of top categories to forecast
            periods: Number of periods to forecast
            
        Returns:
            Dictionary of forecasts for each category
        """
        # Preprocess data if not already done
        if not hasattr(self, 'category_data'):
            self.preprocess_data()
            
        # Get top categories by total demand
        all_categories = {}
        for category, df in self.category_data.items():
            count_column = self._get_count_column(category)
            if count_column:
                all_categories[category] = 0  # Default if no count column found
                
        top_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Run forecasts for top categories
        for category, _ in top_categories:
            print(f"\nProcessing category: {category}")
            
            # Train model and generate forecast
            self.train_and_forecast(category, periods=periods)
            
        return {cat: self.forecasts[cat] for cat, _ in top_categories if cat in self.forecasts}
    
    def plot_forecast(self, category, figsize=(12, 6)):
        """
        Plot historical data and forecast
        
        Args:
            category: Product category to plot
            figsize: Figure size tuple
        """
        if category not in self.category_data or category not in self.forecasts:
            print(f"No forecast available for category '{category}'")
            return
        
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
            
        # Get historical data and forecast
        historical = self.category_data[category][count_column]
        forecast = self.forecasts[category]
        
        # Performance metrics
        perf = self.performance[category]
        params = self.best_params.get(category, (1, 1, 1))
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(historical.index, historical.values, label='Historical Demand', color='blue')
        
        # Plot forecast
        plt.plot(forecast.index, forecast['forecast'], label='Forecast', color='red')
        
        # Plot confidence intervals
        plt.fill_between(forecast.index, 
                        forecast['lower_ci'], 
                        forecast['upper_ci'], 
                        color='pink', alpha=0.3, label='95% Confidence Interval')
        
        # Add labels and title
        title = f'Demand Forecast for {category}\n'
        if perf['mape'] is not None:
            title += f'ARIMA{params} (MAPE: {perf["mape"]:.2f}%, RMSE: {perf["rmse"]:.2f})'
        else:
            title += f'ARIMA{params} (Limited data)'
            
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Order Count')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plt.savefig(f'{category}_demand_forecast.png')
        plt.close()
    
    def generate_forecast_report(self, output_file="forecast_report.csv"):
        """
        Generate a summary report of all forecasts
        
        Args:
            output_file: Path to save the CSV report
        """
        if not self.forecasts:
            print("No forecasts available. Run forecasts first.")
            return
            
        # Prepare report data
        report_data = []
        
        for category, forecast in self.forecasts.items():
            # Get performance metrics
            perf = self.performance[category]
            params = self.best_params.get(category, (1, 1, 1))
            
            # Get average historical demand
            count_column = self._get_count_column(category)
            if not count_column:
                continue
                
            historical = self.category_data[category][count_column]
            avg_demand = max(historical.mean(), 1)  # Ensure at least 1
            
            # Get future demand statistics
            try:
                future_demand = forecast['forecast'].mean()
                
                # Handle case when forecast doesn't have confidence intervals
                if 'lower_ci' in forecast.columns and 'upper_ci' in forecast.columns:
                    min_demand = forecast['lower_ci'].min()
                    max_demand = forecast['upper_ci'].max()
                else:
                    # Use 30% below and above the forecast as an estimate
                    min_demand = max(future_demand * 0.7, 0)
                    max_demand = future_demand * 1.3
                
                # Calculate growth rate with robust handling
                if avg_demand > 0:
                    growth_rate = ((future_demand - avg_demand) / avg_demand) * 100
                else:
                    # If historical demand is zero, we can't calculate percentage growth
                    growth_rate = 0 if future_demand == 0 else 100  # Assume 100% growth if going from 0 to non-zero
                
                # Bound growth rate to reasonable values (-100% to +100%)
                growth_rate = max(min(growth_rate, 100), -100)
                
            except Exception as e:
                print(f"Error calculating forecast statistics for {category}: {e}")
                # Use default values
                future_demand = avg_demand
                min_demand = max(avg_demand * 0.7, 0)
                max_demand = avg_demand * 1.3
                growth_rate = 0
            
            # Add to report
            report_data.append({
                'category': category,
                'arima_params': f"({params[0]},{params[1]},{params[2]})",
                'avg_historical_demand': avg_demand,
                'forecast_demand': future_demand,
                'min_forecast': min_demand,
                'max_forecast': max_demand,
                'growth_rate': growth_rate,
                'mae': perf['mae'],
                'rmse': perf['rmse'],
                'mape': perf['mape'],
                # Add a flag for forecasts with limited data
                'data_quality': 'Limited' if len(historical) < 8 else 'Sufficient',
                # Include visualization_data flag
                'has_visualization': True
            })
            
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)
        
        # Ensure numeric columns have appropriate data types
        numeric_cols = ['avg_historical_demand', 'forecast_demand', 'min_forecast', 
                    'max_forecast', 'growth_rate', 'mae', 'rmse', 'mape']
        
        for col in numeric_cols:
            if col in report_df.columns:
                # Convert to float but keep NaN values
                report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
        
        report_df.to_csv(output_file, index=False)
        
        print(f"Forecast report saved to {output_file}")
        return report_df

# If run as a script
if __name__ == "__main__":
    # Example usage
    forecaster = DemandForecaster()
    forecaster.preprocess_data()
    
    # Run forecasts for top 5 categories
    forecasts = forecaster.run_all_forecasts(top_n=5, periods=6)
    
    # Generate report
    report = forecaster.generate_forecast_report()
    
    # Plot forecasts for each category
    for category in forecasts:
        forecaster.plot_time_series(category)
        forecaster.plot_acf_pacf(category)
        forecaster.plot_forecast(category)
        
    print("Forecasting complete. Check the generated report and visualization files.")