import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  CardHeader, Divider, List, ListItem, ListItemText,
  FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';

/**
 * Demand Forecast Page Component
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Forecast data
 */
const DemandForecastPage = ({ data }) => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [forecastData, setForecastData] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState([]);
  
  useEffect(() => {
    if (data && data.forecastReport && data.forecastReport.length > 0) {
      // Set the initial category selection
      setSelectedCategory(data.forecastReport[0].category);
      
      // Process performance metrics with null checks
      if (data.performanceMetrics) {
        setPerformanceMetrics(
          data.performanceMetrics
            .filter(metric => metric != null) // Filter out null metrics
            .map(metric => ({
              ...metric,
              mapeColor: (metric.mape != null) 
                ? (metric.mape < 10 ? '#4caf50' : metric.mape < 20 ? '#ff9800' : '#f44336')
                : '#cccccc' // Default color when mape is null
            }))
        );
      }
    }
  }, [data]);
  
  // Update forecast data when category changes
  useEffect(() => {
    if (selectedCategory && data && data.forecastReport) {
      // Find the selected category forecast
      const categoryForecast = data.forecastReport.find(
        forecast => forecast.category === selectedCategory
      );
      
      if (categoryForecast) {
        // Create forecast data points (mock data - would come from real API)
        const currentDate = new Date();
        const forecastPoints = [];
        
        // Historical data (last 6 months)
        for (let i = 6; i > 0; i--) {
          const date = new Date(currentDate);
          date.setMonth(date.getMonth() - i);
          
          forecastPoints.push({
            date: date,
            value: Math.round((categoryForecast.avg_historical_demand || 100) * (0.9 + Math.random() * 0.2)),
            type: 'historical'
          });
        }
        
        // Current month
        forecastPoints.push({
          date: currentDate,
          value: Math.round(categoryForecast.avg_historical_demand || 100),
          type: 'historical'
        });
        
        // Forecast data (next 6 months)
        for (let i = 1; i <= 6; i++) {
          const date = new Date(currentDate);
          date.setMonth(date.getMonth() + i);
          
          // Calculate forecasted value with growth rate with null checks
          const growthRate = categoryForecast.growth_rate || 0;
          const forecastDemand = categoryForecast.forecast_demand || 100;
          const growthFactor = 1 + (growthRate / 100) * (i / 6);
          const forecastedValue = Math.round(forecastDemand * growthFactor);
          
          // Calculate confidence intervals with null checks
          const mape = categoryForecast.mape || 10;
          const margin = forecastedValue * (mape / 100) * (i / 3);
          
          forecastPoints.push({
            date: date,
            value: forecastedValue,
            lowerBound: Math.max(0, Math.round(forecastedValue - margin)),
            upperBound: Math.round(forecastedValue + margin),
            type: 'forecast'
          });
        }
        
        setForecastData(forecastPoints);
      }
    }
  }, [selectedCategory, data]);
  
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
    
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };
  
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
                {(data.forecastReport || []).map(forecast => (
                  <MenuItem key={forecast.category} value={forecast.category}>
                    {forecast.category}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>
        </Grid>
        
        {/* Forecast Chart */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 500 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Demand Forecast for {selectedCategory}
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
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
                />
                <Tooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  labelFormatter={formatDate}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="value"
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
                  name="Lower Bound"
                  stroke="#ff8042"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  activeDot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Forecast Details */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, height: 500, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Statistics
            </Typography>
            
            {(data.forecastReport || []).filter(f => f && f.category === selectedCategory).map(forecast => (
              <Box key={forecast.category} sx={{ mb: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Average Historical Demand
                        </Typography>
                        <Typography variant="h6">
                          {(forecast.avg_historical_demand != null) 
                            ? forecast.avg_historical_demand.toFixed(0) 
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
                          {(forecast.forecast_demand != null) 
                            ? forecast.forecast_demand.toFixed(0) 
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
                        <Typography variant="h6">
                          {(forecast.growth_rate != null) 
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
                          {(forecast.mape != null) 
                            ? forecast.mape.toFixed(2) 
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
                          {(forecast.rmse != null) 
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
            <Box sx={{ mt: 'auto' }}>
              <Typography variant="subtitle2" gutterBottom>
                Forecast Interpretation
              </Typography>
              {(data.forecastReport || [])
                .filter(f => f && f.category === selectedCategory)
                .map(forecast => (
                <Typography key={forecast.category} variant="body2" sx={{ mb: 2 }}>
                  {(forecast.growth_rate != null)
                    ? (forecast.growth_rate > 5 
                        ? `The ${forecast.category} category shows strong growth (${forecast.growth_rate.toFixed(1)}%). Consider increasing inventory.`
                        : forecast.growth_rate > 0
                        ? `The ${forecast.category} category shows moderate growth (${forecast.growth_rate.toFixed(1)}%). Maintain current inventory levels.`
                        : `The ${forecast.category} category shows a decline (${forecast.growth_rate.toFixed(1)}%). Consider reducing inventory.`)
                    : `The ${forecast.category} category has insufficient data for growth analysis.`
                  }
                </Typography>
              ))}
            </Box>
          </Paper>
        </Grid>
        
        {/* Model Performance */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Model Performance by Category
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
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
                />
                <Tooltip formatter={(value) => (value != null) ? `${value.toFixed(2)}%` : 'N/A'} />
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
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DemandForecastPage;