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
        for category in self.categories:
            if pd.isna(category):
                continue
                
            category_df = self.data[self.data['product_category_name'] == category].copy()
            category_df.sort_values('date', inplace=True)
            category_df.set_index('date', inplace=True)
            
            # Only keep categories with enough data
            if len(category_df) >= 12:  # At least 12 months of data
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
                    
        print(f"Processed {len(self.category_data)} categories with sufficient data")
        
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
        
        # Run ADF test
        result = adfuller(ts.dropna())
        
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05
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
            plot_acf(ts_diff, ax=ax1, lags=max_lags)
            ax1.set_title('Autocorrelation Function')
            
            # Plot PACF
            plot_pacf(ts_diff, ax=ax2, lags=max_lags)
            ax2.set_title('Partial Autocorrelation Function')
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
        if len(ts) < 12:
            print(f"Insufficient data for {category} to find optimal parameters")
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
        Train ARIMA model and generate forecasts
        
        Args:
            category: Product category to forecast
            periods: Number of periods to forecast
            use_best_params: Whether to use grid search for best parameters
            
        Returns:
            Dictionary with forecast results
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return None
        
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return None
            
        ts = self.category_data[category][count_column]
        
        # Determine ARIMA parameters
        if use_best_params:
            if category not in self.best_params:
                params = self.find_best_parameters(category)
            else:
                params = self.best_params[category]
        else:
            # Default parameters
            params = (1, 1, 1)
        
        # Train-test split for validation (80-20 split)
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        # Train model on training data
        model = ARIMA(train, order=params)
        model_fit = model.fit()
        
        # Store the model
        self.models[category] = model_fit
        
        # Validate on test data
        forecast = model_fit.forecast(steps=len(test))
        
        # Calculate error metrics
        if len(test) > 0:
            try:
                mae = mean_absolute_error(test, forecast)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                mape = np.mean(np.abs((test - forecast) / test)) * 100
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                mae, rmse, mape = None, None, None
        else:
            mae, rmse, mape = None, None, None
        
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
            forecast_index = pd.date_range(start=ts.index[-1], periods=periods+1, freq='M')[1:]
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower_ci': final_model_fit.get_forecast(periods).conf_int().iloc[:, 0],
                'upper_ci': final_model_fit.get_forecast(periods).conf_int().iloc[:, 1]
            }, index=forecast_index)
            
            # Store forecast
            self.forecasts[category] = forecast_df
            
            return {
                'model': final_model_fit,
                'forecast': forecast_df,
                'params': params,
                'performance': self.performance[category]
            }
        except Exception as e:
            print(f"Error in final forecasting for {category}: {e}")
            return None
    
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
            title += f'ARIMA{params}'
            
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
    
    def run_all_forecasts(self, top_n=5, periods=6):
        """
        Run forecasts for top N categories
        
        Args:
            top_n: Number of top categories to forecast
            periods: Number of periods to forecast
        """
        # Preprocess data if not already done
        if not hasattr(self, 'category_data'):
            self.preprocess_data()
            
        # Get top categories by total demand
        all_categories = {}
        for category, df in self.category_data.items():
            count_column = self._get_count_column(category)
            if count_column:
                all_categories[category] = df[count_column].sum()
            else:
                all_categories[category] = 0  # Default if no count column found
            
        top_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Run forecasts for top categories
        for category, _ in top_categories:
            print(f"\nProcessing category: {category}")
            
            # Plot time series analysis
            self.plot_time_series(category)
            
            # Plot ACF and PACF
            self.plot_acf_pacf(category)
            
            # Train model and generate forecast
            self.train_and_forecast(category, periods=periods)
            
            # Plot forecast
            self.plot_forecast(category)
            
        return {cat: self.forecasts[cat] for cat, _ in top_categories if cat in self.forecasts}
    
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
            avg_demand = historical.mean()
            
            # Get future demand statistics
            future_demand = forecast['forecast'].mean()
            min_demand = forecast['lower_ci'].min()
            max_demand = forecast['upper_ci'].max()
            
            # Calculate growth rate
            growth_rate = ((future_demand - avg_demand) / avg_demand) * 100 if avg_demand > 0 else 0
            
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
                'mape': perf['mape']
            })
            
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)
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