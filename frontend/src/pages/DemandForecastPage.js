import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  FormControl, InputLabel, Select, MenuItem,
  Alert, AlertTitle
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';

/**
 * Improved Forecast Interpretation Component that handles extreme values gracefully
 * 
 * @param {Object} props Component props
 * @param {Object} props.forecast Selected category forecast data
 */
/**
 * Enhanced Forecast Interpretation Component with more informative guidance
 * 
 * @param {Object} props Component props
 * @param {Object} props.forecast Selected category forecast data
 */
const ForecastInterpretation = ({ forecast }) => {
  if (!forecast) return null;
  
  // Get normalized values with better handling of invalid values
  const hasGrowthRate = forecast.growth_rate != null && !isNaN(parseFloat(forecast.growth_rate));
  const hasMape = forecast.mape != null && !isNaN(parseFloat(forecast.mape));
  const hasRmse = forecast.rmse != null && !isNaN(parseFloat(forecast.rmse));
  const hasHistoricalDemand = forecast.avg_historical_demand != null && 
                             !isNaN(parseFloat(forecast.avg_historical_demand));
  const dataQuality = forecast.data_quality || 'Unknown';
  
  // Normalize and cap extreme values
  const growthRate = hasGrowthRate ? 
    Math.max(Math.min(parseFloat(forecast.growth_rate), 100), -80) : 0;
  
  const mape = hasMape ? 
    Math.min(parseFloat(forecast.mape), 100) : 30;
  
  // Generate interpretation text based on data quality and values
  let interpretationText = '';
  
  if (dataQuality === 'Limited') {
    interpretationText = `The ${forecast.category} category has limited historical data, which affects forecast reliability. `;
    
    if (hasHistoricalDemand && parseFloat(forecast.avg_historical_demand) > 0) {
      interpretationText += `Historical average demand is ${Math.round(forecast.avg_historical_demand)} units. `;
      
      if (hasGrowthRate) {
        // Use absolute value for messaging but keep the direction
        const growthDirection = growthRate > 0 ? 'growth' : 'decline';
        interpretationText += `The forecast suggests a ${Math.abs(growthRate).toFixed(1)}% ${growthDirection}, but this should be treated as an estimate given the limited data. `;
      } else {
        interpretationText += `Growth trends could not be reliably determined with limited data. `;
      }
      
      interpretationText += 'Consider collecting more historical data to improve forecast accuracy.';
    } else {
      interpretationText += 'Insufficient historical data is available for meaningful analysis.';
    }
  } else {
    // More sophisticated interpretation based on growth rate and accuracy
    let forecastConfidence = "";
    
    // First determine confidence level based on MAPE
    if (mape > 60) {
      forecastConfidence = "Low confidence: ";
      interpretationText += "This forecast has high uncertainty due to volatile historical patterns. ";
    } else if (mape > 30) {
      forecastConfidence = "Moderate confidence: ";
      interpretationText += "This forecast has moderate uncertainty. ";
    } else {
      forecastConfidence = "High confidence: ";
    }
    
    // Now add growth-specific advice
    if (growthRate <= -50) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows a significant decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider substantially reducing inventory and investigating potential market shifts.`;
    } else if (growthRate < -20) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows a strong decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider reducing inventory and exploring product/marketing improvements.`;
    } else if (growthRate < 0) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows a moderate decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider slight inventory reductions while monitoring trends.`;
    } else if (growthRate > 50) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows exceptional growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Consider significantly increasing inventory and expanding supplier capacity.`;
    } else if (growthRate > 20) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows strong growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Consider increasing inventory and securing additional supply.`;
    } else if (growthRate > 5) {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows moderate growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Maintain current inventory levels with slight increases.`;
    } else {
      interpretationText += `${forecastConfidence}The ${forecast.category} category shows stable demand (${growthRate.toFixed(1)}%). `;
      interpretationText += `Maintain current inventory levels.`;
    }
    
    // Add quantitative context
    if (hasHistoricalDemand && hasRmse) {
      const histDemand = parseFloat(forecast.avg_historical_demand);
      const rmseValue = parseFloat(forecast.rmse);
      const rmsePercentage = (rmseValue / histDemand * 100).toFixed(1);
      
      interpretationText += ` Typical forecast error is about ${rmsePercentage}% (${Math.round(rmseValue)} units).`;
    }
  }
  
  // Add actionable guidance based on data quality and forecast
  let actionableGuidance = "";
  
  if (mape > 60) {
    actionableGuidance = "Recommendation: Consider using qualitative methods (expert opinion, market research) alongside this quantitative forecast for decision-making.";
  } else if (growthRate > 20) {
    actionableGuidance = "Recommendation: Develop a scaling plan to handle the projected growth, including supplier and logistics capacity.";
  } else if (growthRate < -20) {
    actionableGuidance = "Recommendation: Develop a plan to gradually reduce inventory while maintaining service levels.";
  } else {
    actionableGuidance = "Recommendation: Review inventory policies quarterly to ensure alignment with demand patterns.";
  }
  
  return (
    <Box sx={{ mt: 'auto' }}>
      <Typography variant="subtitle2" gutterBottom>
        Forecast Interpretation
      </Typography>
      <Typography variant="body2" sx={{ mb: 2 }}>
        {interpretationText}
      </Typography>
      <Typography variant="body2" fontWeight="medium" color="primary.main">
        {actionableGuidance}
      </Typography>
    </Box>
  );
};

/**
 * Demand Forecast Page Component with enhanced data handling and visualization
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Forecast data
 */
const DemandForecastPage = ({ data }) => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [forecastData, setForecastData] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState([]);
  const [forecastNote, setForecastNote] = useState('');
  const [hasVisualizationData, setHasVisualizationData] = useState(false);
  
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
  
  // Initialize data and select first category
  useEffect(() => {
    if (data && data.forecastReport && data.forecastReport.length > 0) {
      // Set the initial category selection
      const validForecasts = getValidArray(data.forecastReport).filter(f => f && f.category);
      if (validForecasts.length > 0) {
        setSelectedCategory(validForecasts[0].category);
      }
      
      // Process performance metrics with null checks and validations
      if (data.performanceMetrics) {
        setPerformanceMetrics(
          getValidArray(data.performanceMetrics)
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
              };
            })
        );
      }
    }
  }, [data]);
  
  // Get category data from data structure
  const getCategoryData = (data, category) => {
    if (data && data.categories && data.categories.categoryData && data.categories.categoryData[category]) {
      return data.categories.categoryData[category];
    }
    return [];
  };
  
  // Update forecast data when category changes
// Update forecast data when category changes
// Update the visualization handling in DemandForecastPage.js

// Improve the useEffect that handles forecast data changes
useEffect(() => {
  if (selectedCategory && data) {
    // Always create visualization data, even for uncertain forecasts
    const forecastReport = getValidArray(data.forecastReport);
    const categoryData = getCategoryData(data, selectedCategory);
    const historicalData = processHistoricalData(categoryData);
    
    // Find the selected category forecast
    const categoryForecast = forecastReport.find(
      forecast => forecast.category === selectedCategory
    );
    
    // Always create visualizations, even if uncertainty is high
    if (categoryForecast && historicalData.length > 0) {
      const lastPoint = historicalData[historicalData.length - 1];
      
      // Get safe forecast values
      const forecastValue = parseFloat(categoryForecast.forecast_demand || 
                                      categoryForecast.next_month_forecast || 0);
      const growthRate = categoryForecast.growth_rate !== undefined && 
                        categoryForecast.growth_rate !== null
                        ? Math.max(Math.min(parseFloat(categoryForecast.growth_rate), 100), -80) / 100
                        : 0;
      
      // Generate forecast points
      const forecastPoints = [];
      for (let i = 1; i <= 6; i++) {
        const forecastDate = new Date(lastPoint.date);
        forecastDate.setMonth(forecastDate.getMonth() + i);
        
        // Calculate point value - for first point use actual forecast if available
        let pointValue;
        if (i === 1 && forecastValue > 0) {
          pointValue = forecastValue;
        } else {
          // For subsequent points, apply growth pattern based on type
          if (growthRate < -0.5) {
            // Severe decline case - asymptotically approach minimum
            const decayFactor = Math.pow(1 + growthRate, i);
            pointValue = Math.max(lastPoint.value * decayFactor, 1);
          } else if (growthRate > 0) {
            // Growth case - dampen growth rate for future periods
            const adjustedGrowth = growthRate / (1 + 0.2 * (i - 1));
            pointValue = lastPoint.value * (1 + adjustedGrowth * i);
          } else {
            // Moderate decline or flat
            pointValue = Math.max(lastPoint.value * Math.pow(1 + growthRate, i), 1);
          }
        }
        
        // Calculate uncertainty bounds based on MAPE
        const mape = categoryForecast.mape !== undefined && 
                   categoryForecast.mape !== null
                   ? Math.min(parseFloat(categoryForecast.mape), 100) / 100 
                   : 0.2;
                   
        // Uncertainty increases with time horizon
        const errorFactor = mape * (1 + (i-1) * 0.2);
        const lowerBound = Math.max(pointValue * (1 - errorFactor), 0);
        const upperBound = pointValue * (1 + errorFactor);
        
        forecastPoints.push({
          date: forecastDate,
          value: pointValue,
          lowerBound: lowerBound,
          upperBound: upperBound,
          type: 'forecast'
        });
      }
      
      // Combine historical and forecast data
      const combinedData = [...historicalData, ...forecastPoints];
      setForecastData(combinedData);
      setHasVisualizationData(true);
      
      // Indicate forecast quality in note
      const mapeValue = categoryForecast.mape;
      if (mapeValue && mapeValue > 50) {
        setForecastNote('Note: High forecast uncertainty. Consider this visualization as indicative only.');
      } else if (mapeValue && mapeValue > 25) {
        setForecastNote('Note: Moderate forecast uncertainty. Treat projections as directional estimates.');
      } else {
        setForecastNote('');
      }
    } else {
      // Empty forecast with no data
      setForecastData(historicalData);
      setHasVisualizationData(false);
      setForecastNote('No forecast data available for this category.');
    }
  }
}, [selectedCategory, data]);
  
  // Helper function to process historical data
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
    })
    .sort((a, b) => a.date - b.date); // Sort by date
};
  
// Helper function to generate forecast points from historical data and forecast statistics
const generateForecastPoints = (historicalData, categoryForecast) => {
  if (historicalData.length < 2) return [];
  
  // Get the last historical point
  const lastPoint = historicalData[historicalData.length - 1];
  
  // Get growth rate from forecast
  const growthRate = parseNumericValue(categoryForecast.growth_rate, 0) / 100; // Convert to decimal
  
  // Get forecast statistics for bounds calculation
  const mape = parseNumericValue(categoryForecast.mape, 20) / 100; // Default to 20% if not available
  
  // Generate 6 forecast points
  const forecastPoints = [];
  for (let i = 1; i <= 6; i++) {
    // Create date for this forecast point (add i months to last date)
    const forecastDate = new Date(lastPoint.date);
    forecastDate.setMonth(forecastDate.getMonth() + i);
    
    // Calculate forecast value based on growth rate
    // With limit to ensure no negative values
    const forecastValue = Math.max(lastPoint.value * (1 + (growthRate * i/6)), 0);
    
    // Calculate error bounds with increasing uncertainty over time
    const errorFactor = mape * (1 + (i-1)/6); // Increasing uncertainty with time
    const lowerBound = Math.max(forecastValue * (1 - errorFactor), 0);
    const upperBound = forecastValue * (1 + errorFactor);
    
    forecastPoints.push({
      date: forecastDate,
      value: forecastValue,
      lowerBound: lowerBound,
      upperBound: upperBound,
      type: 'forecast'
    });
  }
  
  return forecastPoints;
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
  
  const categoryForecasts = getCurrentCategoryForecast();
  const dataQuality = categoryForecasts && categoryForecasts[0] ? categoryForecasts[0].data_quality : null;
  
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
            
            <ResponsiveContainer width="100%" height="100%">
  {hasVisualizationData ? (
    <LineChart
      data={forecastData}
      margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
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
      {forecastData.some(d => d.type === 'forecast') && (
        <>
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
        </>
      )}
    </LineChart>
  ) : (
    <Box display="flex" justifyContent="center" alignItems="center" height="100%">
      <Typography variant="body1" color="text.secondary">
        {forecastNote || "No historical data available for this category"}
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