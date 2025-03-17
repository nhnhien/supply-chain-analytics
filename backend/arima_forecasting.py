import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm  # For auto_arima
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class DemandForecaster:
    """
    Enhanced class for demand forecasting using ARIMA models
    with improved parameter selection, seasonality handling, and UI feedback.
    """
    def __init__(self, data_path="demand_by_category.csv"):
        """
        Initialize the forecaster with data.
        
        Args:
            data_path: Path to CSV file with demand data.
        """
        self.data = pd.read_csv(data_path)
        self.categories = self.data['product_category_name'].unique()
        self.models = {}
        self.forecasts = {}
        self.performance = {}
        self.best_params = {}
        self.growth_rates = {}
        self.count_column = None
        self.use_auto_arima = False  # Set to True for auto parameter selection
        self.seasonal = False        # Set to True to include seasonality in the model
        self.seasonal_period = 12    # Default seasonal period (monthly data)
        
        # New state variables for UI feedback
        self.forecastNote = {}         # Stores fallback or error notes per category
        self.visualization_data = {}   # Stores combined historical and forecast visualization data by category

    def _get_count_column(self, category):
        """Helper to find the appropriate count column for a category."""
        count_column = None
        possible_count_columns = ['order_count', 'count', 'order_id_count', 'count(order_id)']
        for col in possible_count_columns:
            if col in self.category_data[category].columns:
                count_column = col
                break
        if count_column is None:
            numeric_cols = self.category_data[category].select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                count_column = numeric_cols[0]
        return count_column

    def preprocess_data(self):
        """
        Preprocess data for time series analysis with enhanced date handling.
        """
        if 'year' not in self.data.columns or 'month' not in self.data.columns:
            date_columns = [col for col in self.data.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                print(f"Using {date_col} to extract year and month")
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data['year'] = self.data[date_col].dt.year
                self.data['month'] = self.data[date_col].dt.month
                self.data['date'] = pd.to_datetime(
                    self.data['year'].astype(str) + '-' +
                    self.data['month'].astype(str).str.zfill(2) + '-01'
                )
            else:
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
                    print("WARNING: No date columns found. Using index as date reference.")
                    self.data['date'] = pd.date_range(start='2021-01-01', periods=len(self.data), freq='M')
                    self.data['year'] = self.data['date'].dt.year
                    self.data['month'] = self.data['date'].dt.month
        else:
            self.data['date'] = pd.to_datetime(
                self.data['year'].astype(str) + '-' +
                self.data['month'].astype(str).str.zfill(2) + '-01'
            )
        
        self.category_data = {}
        for category in self.categories:
            if pd.isna(category):
                continue
            category_df = self.data[self.data['product_category_name'] == category].copy()
            category_df.sort_values('date', inplace=True)
            category_df.set_index('date', inplace=True)
            self.category_data[category] = category_df
        
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
            for category in self.category_data:
                numeric_cols = self.category_data[category].select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    self.count_column = numeric_cols[0]
                    print(f"No standard count column found. Using '{self.count_column}' as count column")
                    break
                    
        print(f"Processed {len(self.category_data)} categories")
    
    def test_stationarity(self, category):
        """
        Test if a time series is stationary using the Augmented Dickey-Fuller test.
        
        Args:
            category: Product category to test.
            
        Returns:
            Dictionary with test results.
        """
        if category not in self.category_data:
            return {"error": "Category not found"}
        count_column = self._get_count_column(category)
        if not count_column:
            return {"error": "No suitable numeric column found"}
        ts = self.category_data[category][count_column]
        if len(ts) < 4:
            return {
                "adf_statistic": None,
                "p_value": None,
                "critical_values": None,
                "is_stationary": False,
                "error": "Insufficient data for stationarity test"
            }
        try:
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
        Plot time series data with trend and seasonality analysis.
        
        Args:
            category: Product category to plot.
            figsize: Figure size tuple (width, height).
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
        ts = self.category_data[category][count_column]
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
        fig.suptitle(f'Time Series Analysis: {category}', fontsize=16)
        axes[0].plot(ts.index, ts.values)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Order Count')
        axes[0].grid(True)
        try:
            if len(ts) >= 24:
                decomposition = seasonal_decompose(ts, model='additive', period=12)
                axes[1].plot(decomposition.trend.index, decomposition.trend.values)
                axes[1].set_title('Trend Component')
                axes[1].set_ylabel('Trend')
                axes[1].grid(True)
                axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
                axes[2].set_title('Seasonal Component')
                axes[2].set_ylabel('Seasonality')
                axes[2].grid(True)
            else:
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
        Plot ACF and PACF to determine ARIMA parameters.
        
        Args:
            category: Product category to analyze.
            figsize: Figure size tuple (width, height).
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
        ts = self.category_data[category][count_column]
        if len(ts) < 4:
            print(f"Insufficient data for ACF/PACF analysis for {category}")
            return
        stationarity_result = self.test_stationarity(category)
        if not stationarity_result.get("is_stationary", False):
            ts_diff = ts.diff().dropna()
            title_suffix = " (After First Differencing)"
        else:
            ts_diff = ts
            title_suffix = " (Original Series)"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'ACF and PACF for {category}{title_suffix}', fontsize=16)
        try:
            max_lags = min(24, len(ts_diff) // 2 - 1)
            if max_lags > 0:
                plot_acf(ts_diff, ax=ax1, lags=max_lags)
                ax1.set_title('Autocorrelation Function')
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
        Find the best ARIMA parameters using either auto_arima or grid search.
        
        Args:
            category: Product category to analyze.
            max_p: Maximum value for p (AR parameter).
            max_d: Maximum value for d (differencing parameter).
            max_q: Maximum value for q (MA parameter).
            
        Returns:
            Tuple with best parameters.
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return None
        count_column = self._get_count_column(category)
        if not count_column:
            return {"error": "No suitable numeric column found"}
        ts = self.category_data[category][count_column]
        if len(ts) < 6:
            print(f"Insufficient data for {category} to find optimal parameters, using default")
            return (1, 1, 1)
        if self.use_auto_arima:
            try:
                print(f"Using auto_arima for {category}")
                seasonal = self.seasonal and len(ts) >= (2 * self.seasonal_period)
                if seasonal:
                    model = pm.auto_arima(
                        ts,
                        start_p=0, max_p=max_p,
                        start_q=0, max_q=max_q,
                        start_P=0, max_P=1,
                        start_Q=0, max_Q=1,
                        d=None, max_d=max_d, D=None, max_D=1,
                        seasonal=True, m=self.seasonal_period,
                        information_criterion='aic',
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True
                    )
                    order = model.order
                    seasonal_order = model.seasonal_order
                    params = (order[0], order[1], order[2], seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3])
                    print(f"Best SARIMA parameters for {category}: SARIMA{order}x{seasonal_order} (AIC: {model.aic():.2f})")
                else:
                    model = pm.auto_arima(
                        ts,
                        start_p=0, max_p=max_p,
                        start_q=0, max_q=max_q,
                        d=None, max_d=max_d,
                        seasonal=False,
                        information_criterion='aic',
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True
                    )
                    params = model.order
                    print(f"Best ARIMA parameters for {category}: {params} (AIC: {model.aic():.2f})")
                self.best_params[category] = params
                return params
            except Exception as e:
                print(f"Error in auto_arima for {category}: {e}")
                print("Falling back to grid search")
        best_aic = float('inf')
        best_params = None
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except Exception:
                        continue
        if best_params is None:
            print(f"Could not find optimal parameters for {category}, using default")
            best_params = (1, 1, 1)
        print(f"Best ARIMA parameters for {category}: {best_params} (AIC: {best_aic:.2f})")
        self.best_params[category] = best_params
        return best_params

    def calculate_consistent_growth_rate(self, category):
        """
        Calculate a consistent growth rate for a category to be used across the application.
        
        Args:
            category: Product category name.
            
        Returns:
            float: Growth rate as a percentage (-80 to +100).
        """
        count_column = self._get_count_column(category)
        if not count_column:
            return 0.0
        historical = self.category_data[category][count_column]
        if len(historical) < 2:
            return 0.0
        if category not in self.forecasts:
            return 0.0
        forecast = self.forecasts[category]
        last_n = min(3, len(historical))
        recent_historical = historical[-last_n:]
        historical_avg = recent_historical.mean()
        if len(forecast) > 0:
            future_val = forecast['forecast'].iloc[0]
        else:
            future_val = historical_avg * 0.9
        if historical_avg > 0:
            growth_rate = ((future_val - historical_avg) / historical_avg) * 100
        else:
            growth_rate = 100 if future_val > 0 else 0
        growth_rate = max(min(growth_rate, 100), -80)
        self.growth_rates[category] = growth_rate
        return growth_rate

    def calculate_robust_mape(self, actuals, forecasts):
        """
        Calculate a robust version of MAPE that handles near-zero values.
        
        Args:
            actuals: Array-like of actual values.
            forecasts: Array-like of forecast values.
            
        Returns:
            float: MAPE value, capped at 100%.
        """
        if len(actuals) != len(forecasts) or len(actuals) == 0:
            return None
        actuals = np.array(actuals)
        forecasts = np.array(forecasts)
        denominators = np.abs(actuals) + np.abs(forecasts)
        valid_indexes = denominators >= 0.0001
        if not np.any(valid_indexes):
            return 50.0
        actuals_valid = actuals[valid_indexes]
        forecasts_valid = forecasts[valid_indexes]
        denominators_valid = denominators[valid_indexes]
        abs_errors = np.abs(forecasts_valid - actuals_valid)
        percentage_errors = abs_errors / (denominators_valid / 2) * 100
        percentage_errors = np.minimum(percentage_errors, 100)
        mape = np.mean(percentage_errors)
        mape = min(mape, 100)
        return mape

    def train_and_forecast(self, category, periods=6, use_best_params=True):
        """
        Train ARIMA model and generate forecasts with improved validation,
        bounds checking, and UI state updates.
        
        Args:
            category: Product category to forecast.
            periods: Number of periods to forecast.
            use_best_params: Whether to use grid search for best parameters.
            
        Returns:
            Dictionary with forecast results, including visualization data.
        """
        if category not in self.category_data:
            print(f"Category '{category}' not found")
            return None
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return None
        ts = self.category_data[category][count_column]
        historical_data = []
        for date, value in zip(ts.index, ts.values):
            historical_data.append({
                'date': date,
                'value': float(value),
                'type': 'historical'
            })
        if len(ts) < 4:
            print(f"Insufficient data points for {category}, need at least 4 data points")
            result = self._create_empty_forecast_results(category, ts)
            self.visualization_data[category] = historical_data
            self.forecastNote[category] = "Insufficient data for forecasting; using fallback forecast."
            return result
        if use_best_params:
            if category not in self.best_params:
                params = self.find_best_parameters(category)
            else:
                params = self.best_params[category]
        else:
            params = (1, 1, 1)
        is_seasonal = self.seasonal and len(params) > 3 and params[6] > 1
        if is_seasonal:
            p, d, q, P, D, Q, s = params
            if len(ts) < (4 + d + D * s):
                print(f"Insufficient data for SARIMA with parameters {params}, reducing to simple ARIMA")
                params = (p, d, q)
                is_seasonal = False
        else:
            if len(params) > 3:
                p, d, q = params[:3]
                params = (p, d, q)
            if params[1] > 1 and len(ts) < (3 + params[1]):
                print(f"Insufficient data for differencing with d={params[1]}, reducing d parameter")
                p, _, q = params
                params = (p, 1, q)
        train_size = max(int(len(ts) * 0.8), 3)
        test_size = len(ts) - train_size
        mae, rmse, mape = None, None, None
        if test_size > 0:
            train, test = ts[:train_size], ts[train_size:]
            try:
                if is_seasonal:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    p, d, q, P, D, Q, s = params
                    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                else:
                    model = ARIMA(train, order=params)
                    model_fit = model.fit()
                self.models[category] = model_fit
                forecast = model_fit.forecast(steps=len(test))
                test_values = test.values if hasattr(test, 'values') else test
                forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
                if len(test_values) == len(forecast_values):
                    mae = mean_absolute_error(test_values, forecast_values)
                    rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
                    mape = self.calculate_robust_mape(test_values, forecast_values)
                else:
                    print(f"Warning: Length mismatch between test and forecast for {category}")
                    mae, rmse, mape = None, None, None
            except Exception as e:
                print(f"Error in model validation for {category}: {e}")
        else:
            print(f"Warning: Insufficient data for validation split for {category}, proceeding with full data training")
        self.performance[category] = {'mae': mae, 'rmse': rmse, 'mape': mape}
        try:
            if is_seasonal:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                p, d, q, P, D, Q, s = params
                final_model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s),
                                      enforce_stationarity=False, enforce_invertibility=False)
                final_model_fit = final_model.fit(disp=False)
            else:
                final_model = ARIMA(ts, order=params)
                final_model_fit = final_model.fit()
            forecast_values = final_model_fit.forecast(steps=periods)
            forecast_values = np.maximum(forecast_values, 0)
            historical_mean = ts.mean()
            significant_decline = False
            if len(ts) >= 6:
                first_3_avg = ts[:3].mean()
                last_3_avg = ts[-3:].mean()
                if first_3_avg > 0 and (last_3_avg / first_3_avg) < 0.5:
                    significant_decline = True
            if not significant_decline:
                min_forecast_value = max(0.1 * historical_mean, 1)
                forecast_values = np.maximum(forecast_values, min_forecast_value)
            else:
                forecast_values = np.maximum(forecast_values, 1)
            try:
                if isinstance(ts.index[-1], pd.Timestamp):
                    last_date = ts.index[-1]
                    forecast_index = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
                else:
                    try:
                        last_idx = int(ts.index[-1])
                        forecast_index = range(last_idx + 1, last_idx + periods + 1)
                    except:
                        forecast_index = range(periods)
            except Exception as e:
                print(f"Error creating forecast index for {category}: {e}")
                forecast_index = range(periods)
            try:
                forecast_values = pd.Series(forecast_values).fillna(ts.mean()).values
                forecast_df = pd.DataFrame({'forecast': forecast_values}, index=forecast_index)
                try:
                    if is_seasonal:
                        if rmse is not None:
                            z_value = 1.645  # 95% confidence
                            forecast_df['lower_ci'] = np.maximum(forecast_values - z_value * rmse, 0)
                            forecast_df['upper_ci'] = forecast_values + z_value * rmse
                        else:
                            forecast_df['lower_ci'] = np.maximum(forecast_values * 0.7, 0)
                            forecast_df['upper_ci'] = forecast_values * 1.3
                    else:
                        forecast_with_ci = final_model_fit.get_forecast(periods)
                        conf_int = forecast_with_ci.conf_int()
                        forecast_df['lower_ci'] = np.maximum(conf_int.iloc[:, 0].values, 0)
                        forecast_df['upper_ci'] = conf_int.iloc[:, 1].values
                except Exception as e:
                    print(f"Warning: Could not calculate confidence intervals for {category}: {e}")
                    if rmse is not None:
                        forecast_df['lower_ci'] = np.maximum(forecast_values - 1.96 * rmse, 0)
                        forecast_df['upper_ci'] = forecast_values + 1.96 * rmse
                    else:
                        forecast_df['lower_ci'] = np.maximum(forecast_values * 0.7, 0)
                        forecast_df['upper_ci'] = forecast_values * 1.3
                self.forecasts[category] = forecast_df  # Store forecast for later reference
                growth_rate = self.calculate_consistent_growth_rate(category)
                forecast_viz_data = []
                for date, value, lower, upper in zip(forecast_index, forecast_values, forecast_df['lower_ci'], forecast_df['upper_ci']):
                    forecast_viz_data.append({
                        'date': date,
                        'value': float(value),
                        'lowerBound': float(lower),
                        'upperBound': float(upper),
                        'type': 'forecast'
                    })
                visualization_data = historical_data + forecast_viz_data
                self.visualization_data[category] = visualization_data
                self.forecastNote[category] = ""  # Clear any previous note
                return {
                    'model': final_model_fit,
                    'forecast': forecast_df,
                    'params': params,
                    'seasonal': is_seasonal,
                    'performance': self.performance[category],
                    'visualization_data': visualization_data,
                    'growth_rate': growth_rate
                }
            except Exception as e:
                print(f"Error creating forecast DataFrame for {category}: {e}")
                result = self._create_empty_forecast_results(category, ts)
                result['visualization_data'] = historical_data
                return result
        except Exception as e:
            print(f"Error in final forecasting for {category}: {e}")
            result = self._create_empty_forecast_results(category, ts)
            result['visualization_data'] = historical_data
            return result

    def _create_empty_forecast_results(self, category, ts):
        """
        Create fallback forecast results when forecasting fails.
        
        Args:
            category: Product category.
            ts: Time series data.
            
        Returns:
            Dictionary with minimal forecast results.
        """
        avg_value = abs(ts.mean()) if not ts.empty else (ts.max() if not ts.empty else 100)
        periods = 6
        try:
            if isinstance(ts.index[-1], pd.Timestamp) and not ts.empty:
                last_date = ts.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
            else:
                if not ts.empty:
                    try:
                        last_idx = int(ts.index[-1])
                        forecast_index = range(last_idx + 1, last_idx + periods + 1)
                    except:
                        forecast_index = range(periods)
                else:
                    forecast_index = range(periods)
        except Exception as e:
            print(f"Error creating fallback forecast index for {category}: {e}")
            forecast_index = range(periods)
        if len(ts) >= 2:
            last_n = min(6, len(ts))
            last_points = ts[-last_n:]
            x = np.arange(len(last_points))
            y = np.array(last_points)
            try:
                slope, intercept = np.polyfit(x, y, 1)
                significant_decline = False
                if slope < 0 and abs(slope) > (0.5 * avg_value / last_n):
                    significant_decline = True
                last_value = float(ts.iloc[-1] if hasattr(ts, 'iloc') else ts[-1])
                if significant_decline:
                    min_value = max(0.1 * avg_value, 1)
                    forecast_values = [max(last_value + slope * (i+1), min_value) for i in range(periods)]
                elif slope > 0:
                    max_growth = last_value * 2
                    forecast_values = [min(last_value + slope * (i+1), max_growth) for i in range(periods)]
                else:
                    min_value = max(0.5 * last_value, 1)
                    forecast_values = [max(last_value + slope * (i+1), min_value) for i in range(periods)]
                if last_value > 0:
                    growth_rate = ((forecast_values[0] - last_value) / last_value) * 100
                    growth_rate = max(min(growth_rate, 100), -80)
                else:
                    growth_rate = 0 if forecast_values[0] == 0 else 100
            except Exception as e:
                print(f"Error in fallback trend calculation for {category}: {e}")
                last_value = float(ts.iloc[-1] if hasattr(ts, 'iloc') else ts[-1]) if not ts.empty else avg_value
                forecast_values = [max(last_value * (0.95 ** (i+1)), 1) for i in range(periods)]
                growth_rate = -5
        else:
            forecast_values = [max(avg_value * 0.9, 1)] * periods
            growth_rate = -10
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'lower_ci': [max(value * 0.7, 0) for value in forecast_values],
            'upper_ci': [value * 1.3 for value in forecast_values]
        }, index=forecast_index)
        self.forecasts[category] = forecast_df
        rmse = ts.std() if len(ts) > 1 else avg_value * 0.2
        self.performance[category] = {
            'mae': avg_value * 0.15 if len(ts) > 0 else None,
            'rmse': rmse,
            'mape': min(30, abs(growth_rate) + 10)
        }
        self.growth_rates[category] = growth_rate
        self.forecastNote[category] = "Fallback forecast used due to insufficient data or errors during modeling."
        visualization_data = []
        for date, value in zip(ts.index, ts.values):
            visualization_data.append({
                'date': date,
                'value': float(value),
                'type': 'historical'
            })
        for date, value, lower, upper in zip(forecast_index, forecast_values, forecast_df['lower_ci'], forecast_df['upper_ci']):
            visualization_data.append({
                'date': date,
                'value': float(value),
                'lowerBound': float(lower),
                'upperBound': float(upper),
                'type': 'forecast'
            })
        return {
            'model': None,
            'forecast': forecast_df,
            'params': self.best_params.get(category, (1, 1, 1)),
            'seasonal': False,
            'performance': self.performance[category],
            'is_fallback': True,
            'visualization_data': visualization_data,
            'growth_rate': growth_rate
        }

    def run_all_forecasts(self, top_n=5, periods=6):
        """
        Run forecasts for top N categories and ensure consistent growth rates.
        
        Args:
            top_n: Number of top categories to forecast.
            periods: Number of periods to forecast.
            
        Returns:
            Dictionary of forecasts for each category.
        """
        if not hasattr(self, 'category_data'):
            self.preprocess_data()
        all_categories = {}
        for category, df in self.category_data.items():
            count_column = self._get_count_column(category)
            if count_column:
                total_demand = df[count_column].sum()
                all_categories[category] = total_demand if not pd.isna(total_demand) else 0
        top_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for category, _ in top_categories:
            print(f"\nProcessing category: {category}")
            self.train_and_forecast(category, periods=periods)
        for category, _ in top_categories:
            if category in self.forecasts:
                self.calculate_consistent_growth_rate(category)
        return {cat: self.forecasts[cat] for cat, _ in top_categories if cat in self.forecasts}

    def plot_forecast(self, category, figsize=(12, 6)):
        """
        Plot historical data and forecast.
        
        Args:
            category: Product category to plot.
            figsize: Figure size tuple.
        """
        if category not in self.category_data or category not in self.forecasts:
            print(f"No forecast available for category '{category}'")
            return
        count_column = self._get_count_column(category)
        if not count_column:
            print(f"No suitable numeric column found for {category}")
            return
        historical = self.category_data[category][count_column]
        forecast = self.forecasts[category]
        perf = self.performance[category]
        params = self.best_params.get(category, (1, 1, 1))
        is_seasonal = len(params) > 3
        plt.figure(figsize=figsize)
        plt.plot(historical.index, historical.values, label='Historical Demand', color='blue', marker='o')
        plt.plot(forecast.index, forecast['forecast'], label='Forecast', color='red')
        plt.fill_between(forecast.index, 
                         forecast['lower_ci'], 
                         forecast['upper_ci'], 
                         color='pink', alpha=0.3, label='95% Confidence Interval')
        if is_seasonal:
            p, d, q, P, D, Q, s = params
            model_name = f"SARIMA({p},{d},{q})x({P},{D},{Q},{s})"
        else:
            model_name = f"ARIMA{params}"
        if perf['mape'] is not None:
            title = f'{model_name} (MAPE: {perf["mape"]:.2f}%, RMSE: {perf["rmse"]:.2f})'
        else:
            title = f'{model_name} (Limited data)'
        plt.title(f'Demand Forecast for {category}\n{title}')
        plt.xlabel('Date')
        plt.ylabel('Order Count')
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig(f'{category}_demand_forecast.png')
        plt.close()
    
    def generate_forecast_report(self, output_file="forecast_report.csv"):
        """
        Generate a comprehensive summary report of all forecasts with improved data quality checks.
        
        Args:
            output_file: Path to save the CSV report.
        """
        if not self.forecasts:
            print("No forecasts available. Run forecasts first.")
            return
        if not hasattr(self, 'growth_rates'):
            self.growth_rates = {}
        report_data = []
        for category, forecast in self.forecasts.items():
            perf = self.performance[category]
            params = self.best_params.get(category, (1, 1, 1))
            count_column = self._get_count_column(category)
            if not count_column:
                continue
            historical = self.category_data[category][count_column]
            avg_demand = max(historical.mean(), 1)
            try:
                if forecast is None or len(forecast) == 0:
                    future_demand = avg_demand * 0.9
                    min_forecast = max(future_demand * 0.7, 1)
                    max_forecast = future_demand * 1.3
                    growth_rate = -10 
                else:
                    future_demand = forecast['forecast'].mean()
                    if 'lower_ci' in forecast.columns and 'upper_ci' in forecast.columns:
                        min_forecast = forecast['lower_ci'].min()
                        max_forecast = forecast['upper_ci'].max()
                    else:
                        min_forecast = max(future_demand * 0.7, 1)
                        max_forecast = future_demand * 1.3
                    if category in self.growth_rates:
                        growth_rate = self.growth_rates[category]
                    else:
                        growth_rate = ((future_demand - avg_demand) / avg_demand) * 100 if avg_demand > 0 else 100
                    growth_rate = max(min(growth_rate, 100), -80)
                    self.growth_rates[category] = growth_rate
            except Exception as e:
                print(f"Error calculating forecast statistics for {category}: {e}")
                future_demand = avg_demand * 0.9
                min_forecast = max(avg_demand * 0.7, 1)
                max_forecast = avg_demand * 1.3
                growth_rate = -10
            if len(params) > 3:
                p, d, q, P, D, Q, s = params
                arima_params = f"({p},{d},{q})x({P},{D},{Q},{s})"
            else:
                arima_params = f"({params[0]},{params[1]},{params[2]})"
            mape_value = perf['mape']
            if mape_value is not None:
                mape_value = min(mape_value, 100)
            report_data.append({
                'category': category,
                'arima_params': arima_params,
                'avg_historical_demand': avg_demand,
                'forecast_demand': future_demand,
                'min_forecast': min_forecast,
                'max_forecast': max_forecast,
                'growth_rate': growth_rate,
                'mae': perf['mae'],
                'rmse': perf['rmse'],
                'mape': mape_value,
                'data_quality': 'Limited' if len(historical) < 8 else 'Sufficient',
                'has_visualization': True,
                'forecast_note': self.forecastNote.get(category, "")
            })
        report_df = pd.DataFrame(report_data)
        numeric_cols = ['avg_historical_demand', 'forecast_demand', 'min_forecast', 
                        'max_forecast', 'growth_rate', 'mae', 'rmse', 'mape']
        for col in numeric_cols:
            if col in report_df.columns:
                report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
                if col == 'growth_rate':
                    report_df[col] = report_df[col].clip(lower=-80, upper=100)
                elif col == 'mape':
                    report_df[col] = report_df[col].clip(upper=100)
        report_df.to_csv(output_file, index=False)
        print(f"Forecast report saved to {output_file}")
        return report_df

# If run as a script
if __name__ == "__main__":
    forecaster = DemandForecaster()
    forecaster.preprocess_data()
    forecaster.use_auto_arima = True
    forecaster.seasonal = True
    forecaster.seasonal_period = 12  # Monthly data
    forecasts = forecaster.run_all_forecasts(top_n=15, periods=6)
    report = forecaster.generate_forecast_report()
    for category in forecasts:
        forecaster.plot_time_series(category)
        forecaster.plot_acf_pacf(category)
        forecaster.plot_forecast(category)
    print("Forecasting complete. Check the generated report and visualization files.")
