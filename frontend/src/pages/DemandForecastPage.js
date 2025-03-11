import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  CardHeader, Divider, List, ListItem, ListItemText,
  FormControl, InputLabel, Select, MenuItem,
  Alert, AlertTitle
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';

/**
 * Forecast Interpretation Component that shows explanations of forecast quality
 * 
 * @param {Object} props Component props
 * @param {Object} props.forecast Selected category forecast data
 */
const ForecastInterpretation = ({ forecast }) => {
  if (!forecast) return null;
  
  // Check if we have meaningful data
  const hasGrowthRate = forecast.growth_rate != null;
  const hasMape = forecast.mape != null;
  const hasRmse = forecast.rmse != null;
  const hasHistoricalDemand = forecast.avg_historical_demand != null;
  const dataQuality = forecast.data_quality || 'Unknown';
  
  // Generate interpretation text based on data quality
  let interpretationText = '';
  
  if (dataQuality === 'Limited') {
    interpretationText = `The ${forecast.category} category has limited historical data, which affects forecast reliability. `;
    
    if (hasHistoricalDemand && forecast.avg_historical_demand > 0) {
      interpretationText += `Historical average demand is ${Math.round(forecast.avg_historical_demand)} units. `;
      
      if (hasGrowthRate) {
        // Use absolute value for messaging but keep the direction
        const growthDirection = forecast.growth_rate > 0 ? 'growth' : 'decline';
        interpretationText += `The forecast suggests a ${Math.abs(forecast.growth_rate).toFixed(1)}% ${growthDirection}, but this should be treated as an estimate. `;
      } else {
        interpretationText += `Growth trends could not be reliably determined. `;
      }
      
      interpretationText += 'Consider collecting more historical data to improve forecast accuracy.';
    } else {
      interpretationText += 'Insufficient historical data is available for meaningful analysis.';
    }
  } else {
    if (hasGrowthRate) {
      if (forecast.growth_rate > 5) {
        interpretationText = `The ${forecast.category} category shows strong growth (${forecast.growth_rate.toFixed(1)}%). Consider increasing inventory.`;
      } else if (forecast.growth_rate > 0) {
        interpretationText = `The ${forecast.category} category shows moderate growth (${forecast.growth_rate.toFixed(1)}%). Maintain current inventory levels.`;
      } else {
        interpretationText = `The ${forecast.category} category shows a decline (${Math.abs(forecast.growth_rate).toFixed(1)}%). Consider reducing inventory.`;
      }
      
      // Add a note about forecast reliability if MAPE is high
      if (hasMape && forecast.mape > 30) {
        interpretationText += ' Note: High forecast error indicates uncertainty in these recommendations.';
      }
    } else {
      interpretationText = `The ${forecast.category} category has insufficient data for growth analysis.`;
    }
  }
  
  // Add RMSE information if available
  if (hasRmse) {
    interpretationText += ` RMSE of ${forecast.rmse.toFixed(2)} indicates the average error magnitude in units.`;
  }
  
  return (
    <Box sx={{ mt: 'auto' }}>
      <Typography variant="subtitle2" gutterBottom>
        Forecast Interpretation
      </Typography>
      <Typography variant="body2" sx={{ mb: 2 }}>
        {interpretationText}
      </Typography>
    </Box>
  );
};

/**
 * Demand Forecast Page Component with enhanced data handling
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Forecast data
 */
const DemandForecastPage = ({ data }) => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [forecastData, setForecastData] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState([]);
  const [forecastNote, setForecastNote] = useState('');
  
  // Initialize data and select first category
  useEffect(() => {
    if (data && data.forecastReport && data.forecastReport.length > 0) {
      // Set the initial category selection
      const validForecasts = data.forecastReport.filter(f => f && f.category);
      if (validForecasts.length > 0) {
        setSelectedCategory(validForecasts[0].category);
      }
      
      // Process performance metrics with null checks and validations
      if (data.performanceMetrics) {
        setPerformanceMetrics(
          data.performanceMetrics
            .filter(metric => metric != null && metric.category != null) // Filter out null metrics
            .map(metric => {
              // Ensure numeric values
              const mape = parseNumericValue(metric.mape);
              const normalizedMape = mape > 100 ? 100 : mape; // Cap MAPE at 100% for coloring
              
              return {
                ...metric,
                mape: mape,
                mapeColor: (mape != null) 
                  ? (normalizedMape < 10 ? '#4caf50' : normalizedMape < 20 ? '#ff9800' : '#f44336')
                  : '#cccccc' // Default color when mape is null
              }
            })
        );
      }
    }
  }, [data]);
  
  // Update forecast data when category changes
  useEffect(() => {
    if (selectedCategory && data) {
      // Get forecast data and category data with proper validations
      const forecastReport = getValidArray(data.forecastReport);
      const categoryData = getCategoryData(data, selectedCategory);
      
      // Find the selected category forecast
      const categoryForecast = forecastReport.find(
        forecast => forecast.category === selectedCategory
      );
      
      if (categoryForecast) {
        // Process historical data
        const historicalPoints = processHistoricalData(categoryData);
        
        // Sort by date
        historicalPoints.sort((a, b) => a.date - b.date);
        
        // Process forecast data
        const forecastPoints = [];
        
        if (historicalPoints.length > 0) {
          // Get the last historical data point
          const lastHistoricalPoint = historicalPoints[historicalPoints.length - 1];
          const lastDate = new Date(lastHistoricalPoint.date);
          const lastValue = lastHistoricalPoint.value;
          
          // Get forecasted demand (use absolute value to ensure positive)
          const forecastDemand = Math.abs(parseNumericValue(categoryForecast.forecast_demand, lastValue));
          
          // Calculate average monthly growth over forecast period (6 months)
          // Calculate growth rate safely, keeping it within reasonable bounds
          let growthRate = parseNumericValue(categoryForecast.growth_rate, 0);
          growthRate = Math.max(Math.min(growthRate, 100), -100); // Limit to ±100%
          
          const monthlyGrowthRate = growthRate / 6; // Distribute growth over 6 months
          
          // Generate forecast points with safety checks
          for (let i = 1; i <= 6; i++) {
            const forecastDate = new Date(lastDate);
            forecastDate.setMonth(forecastDate.getMonth() + i);
            
            // Start from last historical value and apply monthly growth
            // With safety check to ensure positive values
            const forecastValue = Math.max(lastValue * (1 + (monthlyGrowthRate / 100) * i), 0);
            
            // Get MAPE or use a default error percentage
            const mape = parseNumericValue(categoryForecast.mape, 15);
            
            // Calculate confidence intervals with increasing uncertainty
            const errorPercentage = mape / 100 * (1 + (i-1)/6); // Increasing uncertainty with time
            const marginOfError = forecastValue * errorPercentage;
            
            forecastPoints.push({
              date: forecastDate,
              value: forecastValue,
              lowerBound: Math.max(0, forecastValue - marginOfError),
              upperBound: forecastValue + marginOfError,
              type: 'forecast'
            });
          }
        }
        
        // Combine historical and forecast data
        const combinedData = [...historicalPoints, ...forecastPoints];
        
        // Add a note about estimation method if we had to generate the forecast
        if (forecastPoints.length > 0) {
          const hasForecastValues = categoryForecast.forecast_values && 
                                   Array.isArray(categoryForecast.forecast_values) && 
                                   categoryForecast.forecast_values.length > 0;
                                   
          if (!hasForecastValues) {
            setForecastNote('Note: Forecast visualization is based on growth rate estimation. For precise values, please refer to the statistics below.');
          } else {
            setForecastNote('');
          }
        }
        
        setForecastData(combinedData);
      }
    }
  }, [selectedCategory, data]);
  
  // Helper function to process historical data
  const processHistoricalData = (categoryData) => {
    return (categoryData || [])
      .filter(point => point && (parseNumericValue(point.count) > 0 || parseNumericValue(point.order_count) > 0))
      .map(point => {
        // Handle date - ensure it's a valid Date object
        let pointDate = point.date;
        if (typeof pointDate === 'string') {
          pointDate = new Date(pointDate);
        } else if (!pointDate && (point.order_year || point.year) && (point.order_month || point.month)) {
          pointDate = new Date(
            parseInt(point.order_year || point.year), 
            parseInt(point.order_month || point.month) - 1, 
            1
          );
        }
        
        if (!pointDate || isNaN(pointDate.getTime())) {
          // If date is still invalid, use current date as fallback
          pointDate = new Date();
        }
        
        return {
          date: pointDate,
          value: parseNumericValue(point.count || point.order_count, 0),
          type: 'historical'
        };
      });
  };
  
  // Helper function to get category data
  const getCategoryData = (data, category) => {
    if (data && data.categoryData && data.categoryData[category]) {
      return data.categoryData[category];
    }
    return [];
  };
  
  // Helper function to ensure we have an array
  const getValidArray = (arr) => {
    return Array.isArray(arr) ? arr.filter(item => item != null) : [];
  };
  
  // Helper function to parse numeric values safely
  const parseNumericValue = (value, defaultValue = null) => {
    if (value === undefined || value === null || value === '' || value === 'N/A') {
      return defaultValue;
    }
    
    const parsed = parseFloat(value);
    return isNaN(parsed) ? defaultValue : parsed;
  };
  
  // Show loading message if no data is available
  if (!data) {
    return <Typography>No forecast data available</Typography>;
  }
  
  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
  };
  
  // Format date for display
  const formatDate = (date) => {
    if (!date) return '';
    if (typeof date === 'string') date = new Date(date);
    
    if (isNaN(date.getTime())) return 'Invalid Date';
    
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };
  
  // Get current category forecast for statistics panel
  const getCurrentCategoryForecast = () => {
    if (!data || !data.forecastReport) return null;
    
    return getValidArray(data.forecastReport)
      .filter(f => f && f.category === selectedCategory)
      .map(forecast => {
        // Ensure all values are properly formatted and non-negative
        return {
          ...forecast,
          avg_historical_demand: Math.max(parseNumericValue(forecast.avg_historical_demand, 0), 0),
          forecast_demand: Math.max(parseNumericValue(forecast.forecast_demand, 0), 0),
          growth_rate: parseNumericValue(forecast.growth_rate, 0),
          mape: parseNumericValue(forecast.mape),
          rmse: parseNumericValue(forecast.rmse),
          mae: parseNumericValue(forecast.mae),
        };
      });
  };
  
  const categoryForecasts = getCurrentCategoryForecast();
  const dataQuality = categoryForecasts && categoryForecasts[0] ? categoryForecasts[0].data_quality : null;
  const hasNoForecastData = forecastData.length === 0;
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Demand Forecasting
      </Typography>
      
      <Grid container spacing={3}>
        {/* Category Selector */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <FormControl fullWidth>
              <InputLabel id="category-select-label">Product Category</InputLabel>
              <Select
                labelId="category-select-label"
                id="category-select"
                value={selectedCategory}
                label="Product Category"
                onChange={handleCategoryChange}
              >
                {getValidArray(data.forecastReport).map(forecast => (
                  <MenuItem key={forecast.category} value={forecast.category}>
                    {forecast.category}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>
        </Grid>
        
        {/* Data Quality Alert - Show when data is limited */}
        {dataQuality === 'Limited' && (
          <Grid item xs={12}>
            <Alert severity="warning">
              <AlertTitle>Limited Historical Data</AlertTitle>
              The '{selectedCategory}' category has limited historical data points, which may affect forecast accuracy. The system will generate a basic forecast based on available data.
            </Alert>
          </Grid>
        )}
        
        {/* Forecast Chart */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 500 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Demand Forecast for {selectedCategory}
            </Typography>
            
            {forecastNote && (
              <Typography variant="caption" color="text.secondary" sx={{ mb: 2 }}>
                {forecastNote}
              </Typography>
            )}
            
            <ResponsiveContainer width="100%" height="90%">
              {!hasNoForecastData ? (
                <LineChart
                  data={forecastData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                    label={{ value: 'Month', position: 'bottom', offset: 0 }}
                  />
                  <YAxis 
                    label={{ value: 'Order Count', angle: -90, position: 'insideLeft' }}
                    domain={[0, dataMax => Math.max(dataMax * 1.1, 10)]} // 10% buffer, minimum 10
                  />
                  <Tooltip 
                    formatter={(value) => new Intl.NumberFormat().format(value)}
                    labelFormatter={formatDate}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    data={forecastData.filter(d => d.type === 'historical')}
                    name="Historical"
                    stroke="#8884d8"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 8 }}
                    connectNulls
                    isAnimationActive={true}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    data={forecastData.filter(d => d.type === 'forecast')}
                    name="Forecast"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ r: 4 }}
                    connectNulls
                    isAnimationActive={true}
                  />
                  <Line
                    type="monotone"
                    dataKey="upperBound"
                    data={forecastData.filter(d => d.type === 'forecast')}
                    name="Upper Bound"
                    stroke="#ffc658"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                    activeDot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="lowerBound"
                    data={forecastData.filter(d => d.type === 'forecast')}
                    name="Lower Bound"
                    stroke="#ff8042"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                    activeDot={false}
                  />
                </LineChart>
              ) : (
                <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" height="100%">
                  <Typography color="text.secondary" gutterBottom>
                    No detailed forecast visualization available for {selectedCategory}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Basic forecast statistics are available based on historical data
                  </Typography>
                </Box>
              )}
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Forecast Details */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, height: 500, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Statistics
            </Typography>
            
            {categoryForecasts && categoryForecasts.map(forecast => (
              <Box key={forecast.category} sx={{ mb: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Average Historical Demand
                        </Typography>
                        <Typography variant="h6">
                          {forecast.avg_historical_demand != null 
                            ? Math.round(forecast.avg_historical_demand) 
                            : 'N/A'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Forecasted Demand
                        </Typography>
                        <Typography variant="h6">
                          {forecast.forecast_demand != null 
                            ? Math.round(forecast.forecast_demand) 
                            : 'N/A'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Growth Rate
                        </Typography>
                        <Typography variant="h6" color={
                          forecast.growth_rate > 0 ? 'success.main' : 
                          forecast.growth_rate < 0 ? 'error.main' : 
                          'inherit'
                        }>
                          {forecast.growth_rate != null 
                            ? forecast.growth_rate.toFixed(2) 
                            : 'N/A'}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          ARIMA Parameters
                        </Typography>
                        <Typography variant="h6">
                          {forecast.arima_params || 'N/A'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          MAPE
                        </Typography>
                        <Typography variant="h6" color={
                          (forecast.mape != null)
                            ? (forecast.mape < 10 ? 'success.main' : 
                               forecast.mape < 20 ? 'warning.main' : 
                               'error.main')
                            : 'text.secondary'
                        }>
                          {forecast.mape != null
                            ? (forecast.mape > 100 ? '> 100' : forecast.mape.toFixed(2)) 
                            : 'N/A'}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          RMSE
                        </Typography>
                        <Typography variant="h6">
                          {forecast.rmse != null 
                            ? forecast.rmse.toFixed(2) 
                            : 'N/A'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            ))}
            
            {/* Forecast Interpretation */}
            <ForecastInterpretation forecast={categoryForecasts && categoryForecasts[0]} />
          </Paper>
        </Grid>
        
        {/* Model Performance */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Model Performance by Category
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              {performanceMetrics.length > 0 ? (
                <BarChart
                  data={performanceMetrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="category" 
                    angle={-45} 
                    textAnchor="end"
                    height={70}
                  />
                  <YAxis 
                    label={{ value: 'MAPE (%)', angle: -90, position: 'insideLeft' }}
                    domain={[0, dataMax => Math.min(dataMax, 100)]} // Cap at 100% for readability
                  />
                  <Tooltip 
                    formatter={(value) => {
                      if (value == null) return 'N/A';
                      return value > 100 ? '> 100%' : `${value.toFixed(2)}%`;
                    }}
                  />
                  <Legend />
                  <Bar dataKey="mape" name="Mean Absolute Percentage Error (MAPE)">
                    {performanceMetrics.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry && entry.mapeColor ? entry.mapeColor : '#cccccc'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              ) : (
                <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                  <Typography color="text.secondary">
                    No model performance metrics available
                  </Typography>
                </Box>
              )}
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DemandForecastPage;