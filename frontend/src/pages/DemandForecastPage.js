import React, { useState, useEffect, useMemo } from 'react';
import { Grid, Paper, Typography, Box, Card, CardContent, FormControl, InputLabel, Select, MenuItem, Alert, AlertTitle } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

// Helper functions
const getValidArray = (arr) => Array.isArray(arr) ? arr.filter(item => item != null) : [];
const parseNumericValue = (value, defaultValue = null) => {
  if (value === undefined || value === null || value === '' || value === 'N/A') return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

const getCategoryData = (data, category) => {
  if (data && data.categories && data.categories.categoryData && data.categories.categoryData[category]) {
    return data.categories.categoryData[category];
  }
  return [];
};

const processHistoricalData = (categoryData) => {
  return (categoryData || [])
    .filter(point => point && (parseNumericValue(point.count) > 0 || parseNumericValue(point.order_count) > 0))
    .map(point => {
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
        pointDate = new Date();
      }
      return {
        date: pointDate,
        value: parseNumericValue(point.count || point.order_count, 0),
        type: 'historical'
      };
    })
    .sort((a, b) => a.date - b.date);
};

const ForecastInterpretation = ({ forecast }) => {
  if (!forecast) return null;

  const hasGrowthRate = forecast.growth_rate != null && !isNaN(parseFloat(forecast.growth_rate));
  const hasMape = forecast.mape != null && !isNaN(parseFloat(forecast.mape));
  const hasRmse = forecast.rmse != null && !isNaN(parseFloat(forecast.rmse));
  const hasHistoricalDemand = forecast.avg_historical_demand != null && !isNaN(parseFloat(forecast.avg_historical_demand));
  const dataQuality = forecast.data_quality || 'Unknown';

  const growthRate = hasGrowthRate ? Math.max(Math.min(parseFloat(forecast.growth_rate), 100), -80) : 0;
  const mape = hasMape ? Math.min(parseFloat(forecast.mape), 100) : 30;

  let interpretationText = '';

  if (dataQuality === 'Limited') {
    interpretationText = `The ${forecast.category} category has limited historical data, which affects forecast reliability. `;
    if (hasHistoricalDemand && parseFloat(forecast.avg_historical_demand) > 0) {
      interpretationText += `Historical average demand is ${Math.round(forecast.avg_historical_demand)} units. `;
      if (hasGrowthRate) {
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
    let forecastConfidence = "";
    if (mape > 60) {
      forecastConfidence = "Low confidence: ";
      interpretationText += "This forecast has high uncertainty due to volatile historical patterns. ";
    } else if (mape > 30) {
      forecastConfidence = "Moderate confidence: ";
      interpretationText += "This forecast has moderate uncertainty. ";
    } else {
      forecastConfidence = "High confidence: ";
    }
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
    if (hasHistoricalDemand && hasRmse) {
      const histDemand = parseFloat(forecast.avg_historical_demand);
      const rmseValue = parseFloat(forecast.rmse);
      const rmsePercentage = (rmseValue / histDemand * 100).toFixed(1);
      interpretationText += ` Typical forecast error is about ${rmsePercentage}% (${Math.round(rmseValue)} units).`;
    }
  }
  
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

const DemandForecastPage = ({ data }) => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [forecastData, setForecastData] = useState([]);
  const [forecastNote, setForecastNote] = useState('');
  const [hasVisualizationData, setHasVisualizationData] = useState(false);

  // Memoize valid forecast report and performance metrics to avoid re-computation
  const validForecastReport = useMemo(() => getValidArray(data?.forecastReport) || [], [data]);
  const performanceMetrics = useMemo(() => {
    if (!data?.performanceMetrics) return [];
    return getValidArray(data.performanceMetrics)
      .filter(metric => metric && metric.category)
      .map(metric => {
        const mape = parseNumericValue(metric.mape) || 30;
        return {
          ...metric,
          mape,
          mapeColor: (mape < 10 ? '#4caf50' : mape < 20 ? '#ff9800' : '#f44336')
        };
      });
  }, [data]);

  // Set initial category if not already set
  useEffect(() => {
    if (validForecastReport.length > 0 && !selectedCategory) {
      setSelectedCategory(validForecastReport[0].category);
    }
  }, [validForecastReport, selectedCategory]);

  // Update forecast data when selectedCategory or data changes
  useEffect(() => {
    if (!selectedCategory || !data) return;
    const forecastReport = validForecastReport;
    const categoryData = getCategoryData(data, selectedCategory);
    const historicalData = processHistoricalData(categoryData);
    const categoryForecast = forecastReport.find(forecast => forecast.category === selectedCategory);
    if (categoryForecast && historicalData.length > 0) {
      const lastPoint = historicalData[historicalData.length - 1];
      const forecastValue = parseFloat(categoryForecast.forecast_demand || categoryForecast.next_month_forecast || 0);
      const growthRate = categoryForecast.growth_rate != null
        ? Math.max(Math.min(parseFloat(categoryForecast.growth_rate), 100), -80) / 100
        : 0;
      const forecastPoints = [];
      for (let i = 1; i <= 6; i++) {
        const forecastDate = new Date(lastPoint.date);
        forecastDate.setMonth(forecastDate.getMonth() + i);
        let pointValue;
        if (i === 1 && forecastValue > 0) {
          pointValue = forecastValue;
        } else {
          if (growthRate < -0.5) {
            const decayFactor = Math.pow(1 + growthRate, i);
            pointValue = Math.max(lastPoint.value * decayFactor, 1);
          } else if (growthRate > 0) {
            const adjustedGrowth = growthRate / (1 + 0.2 * (i - 1));
            pointValue = lastPoint.value * (1 + adjustedGrowth * i);
          } else {
            pointValue = Math.max(lastPoint.value * Math.pow(1 + growthRate, i), 1);
          }
        }
        const mape = categoryForecast.mape != null
          ? Math.min(parseFloat(categoryForecast.mape), 100) / 100
          : 0.2;
        const errorFactor = mape * (1 + (i - 1) * 0.2);
        const lowerBound = Math.max(pointValue * (1 - errorFactor), 0);
        const upperBound = pointValue * (1 + errorFactor);
        forecastPoints.push({
          date: forecastDate,
          value: pointValue,
          lowerBound,
          upperBound,
          type: 'forecast'
        });
      }
      const combinedData = [...historicalData, ...forecastPoints];
      setForecastData(combinedData);
      setHasVisualizationData(true);
      const mapeValue = categoryForecast.mape;
      if (mapeValue && mapeValue > 50) {
        setForecastNote('Note: High forecast uncertainty. Consider this visualization as indicative only.');
      } else if (mapeValue && mapeValue > 25) {
        setForecastNote('Note: Moderate forecast uncertainty. Treat projections as directional estimates.');
      } else {
        setForecastNote('');
      }
    } else {
      setForecastData(historicalData);
      setHasVisualizationData(false);
      setForecastNote('No forecast data available for this category.');
    }
  }, [selectedCategory, data, validForecastReport]);

  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
  };

  const formatDate = (date) => {
    if (!date) return '';
    const d = typeof date === 'string' ? new Date(date) : date;
    return isNaN(d.getTime()) ? 'Invalid Date' : d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  const getCurrentCategoryForecast = () => {
    if (!data || !data.forecastReport) return null;
    return getValidArray(data.forecastReport)
      .filter(f => f && f.category === selectedCategory)
      .map(forecast => ({
        ...forecast,
        avg_historical_demand: Math.max(parseNumericValue(forecast.avg_historical_demand, 0), 0),
        forecast_demand: Math.max(parseNumericValue(forecast.forecast_demand, 0), 0),
        growth_rate: parseNumericValue(forecast.growth_rate, 0),
        mape: parseNumericValue(forecast.mape),
        rmse: parseNumericValue(forecast.rmse),
        mae: parseNumericValue(forecast.mae)
      }));
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

        {/* Data Quality Alert */}
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
                <LineChart data={forecastData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={formatDate} label={{ value: 'Month', position: 'bottom', offset: 0 }} />
                  <YAxis label={{ value: 'Order Count', angle: -90, position: 'insideLeft' }} domain={[0, (dataMax) => Math.max(dataMax * 1.1, 10)]} />
                  <Tooltip formatter={(value) => new Intl.NumberFormat().format(value)} labelFormatter={formatDate} />
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
            {getValidArray(getCurrentCategoryForecast()) && getCurrentCategoryForecast().map(forecast => (
              <Box key={forecast.category} sx={{ mb: 2 }}>
                {/* Render cards with forecast details (omitted for brevity) */}
              </Box>
            ))}
            {/* Forecast Interpretation */}
            <ForecastInterpretation forecast={getCurrentCategoryForecast() && getCurrentCategoryForecast()[0]} />
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
                <BarChart data={performanceMetrics} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" angle={-45} textAnchor="end" height={70} />
                  <YAxis label={{ value: 'MAPE (%)', angle: -90, position: 'insideLeft' }} domain={[0, (dataMax) => Math.min(dataMax, 100)]} />
                  <Tooltip formatter={(value) => (value == null ? 'N/A' : value > 100 ? '> 100%' : `${value.toFixed(2)}%`)} />
                  <Legend />
                  <Bar dataKey="mape" name="Mean Absolute Percentage Error (MAPE)">
                    {performanceMetrics.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry && entry.mapeColor ? entry.mapeColor : '#cccccc'} />
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
