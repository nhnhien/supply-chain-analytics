import React, { useState, useEffect, useMemo } from 'react';
import { Grid, Paper, Typography, Box, FormControl, InputLabel, Select, MenuItem, Alert, AlertTitle } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

// Helper functions
const getValidArray = (arr) =>
  Array.isArray(arr) ? arr.filter(item => item != null) : [];

const parseNumericValue = (value, defaultValue = null) => {
  if (value === undefined || value === null || value === '' || value === 'N/A') return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

// Safely get category data
const getCategoryData = (data, category) => {
  if (!data || !category) return [];
  return data.categories && data.categories.categoryData && data.categories.categoryData[category]
    ? data.categories.categoryData[category]
    : [];
};

const ForecastInterpretation = ({ forecast }) => {
  if (!forecast) return null;

  const growthRateRaw = parseNumericValue(forecast.growth_rate, 0);
  const mapeRaw = parseNumericValue(forecast.mape, 30);
  const rmseRaw = parseNumericValue(forecast.rmse, null);
  const histDemand = parseNumericValue(forecast.avg_historical_demand, null);
  const dataQuality = forecast.data_quality || 'Unknown';

  const growthRate = growthRateRaw !== null ? Math.max(Math.min(growthRateRaw, 100), -80) : 0;
  const mape = mapeRaw !== null ? Math.min(mapeRaw, 100) : 30;

  let interpretationText = '';

  if (dataQuality === 'Limited') {
    interpretationText = `The ${forecast.category || 'this'} category has limited historical data, which affects forecast reliability. `;
    if (histDemand !== null && histDemand > 0) {
      interpretationText += `Historical average demand is ${Math.round(histDemand)} units. `;
      if (growthRateRaw !== null) {
        const growthDirection = growthRate > 0 ? 'growth' : 'decline';
        interpretationText += `The forecast suggests a ${Math.abs(growthRate).toFixed(1)}% ${growthDirection}, but treat this as an estimate given the limited data. `;
      } else {
        interpretationText += `Growth trends could not be reliably determined. `;
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
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows a significant decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider substantially reducing inventory and investigating market shifts.`;
    } else if (growthRate < -20) {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows a strong decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider reducing inventory and exploring product/marketing improvements.`;
    } else if (growthRate < 0) {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows a moderate decline (${Math.abs(growthRate).toFixed(1)}%). `;
      interpretationText += `Consider slight inventory reductions while monitoring trends.`;
    } else if (growthRate > 50) {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows exceptional growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Consider significantly increasing inventory and expanding supplier capacity.`;
    } else if (growthRate > 20) {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows strong growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Consider increasing inventory and securing additional supply.`;
    } else if (growthRate > 5) {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows moderate growth (${growthRate.toFixed(1)}%). `;
      interpretationText += `Maintain current inventory levels with slight increases.`;
    } else {
      interpretationText += `${forecastConfidence}The ${forecast.category || 'selected'} category shows stable demand (${growthRate.toFixed(1)}%). `;
      interpretationText += `Maintain current inventory levels.`;
    }
    if (histDemand !== null && rmseRaw !== null && histDemand > 0) {
      const rmsePercentage = (rmseRaw / histDemand * 100).toFixed(1);
      interpretationText += ` Typical forecast error is about ${rmsePercentage}% (${Math.round(rmseRaw)} units).`;
    }
  }

  let actionableGuidance = "";
  if (mape > 60) {
    actionableGuidance = "Recommendation: Consider using qualitative methods (expert opinion, market research) alongside this quantitative forecast.";
  } else if (growthRate > 20) {
    actionableGuidance = "Recommendation: Develop a scaling plan to handle the projected growth.";
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
  // Memoize safeData so that it only updates when data changes.
  const safeData = useMemo(() => data || {}, [data]);

  const [selectedCategory, setSelectedCategory] = useState('');
  const [forecastData, setForecastData] = useState([]);
  const [forecastNote, setForecastNote] = useState('');
  const [hasVisualizationData, setHasVisualizationData] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  // Memoize valid forecast report
  const validForecastReport = useMemo(() => {
    const forecastReport = safeData.forecastReport || [];
    return getValidArray(forecastReport);
  }, [safeData.forecastReport]);
  
  const performanceMetrics = useMemo(() => {
    if (!safeData.performanceMetrics) return [];
    return getValidArray(safeData.performanceMetrics)
      .filter(metric => metric && metric.category)
      .map(metric => {
        const mape = parseNumericValue(metric.mape) || 30;
        return {
          ...metric,
          mape,
          mapeColor: mape < 10 ? '#4caf50' : mape < 20 ? '#ff9800' : '#f44336'
        };
      });
  }, [safeData.performanceMetrics]);

  // Process historical data with robust date handling
  const processHistoricalData = (categoryData) => {
    if (!Array.isArray(categoryData)) return [];
    
    console.log(`Processing historical data: ${categoryData.length} data points`);
    if (categoryData.length > 0) {
      console.log('Sample data point:', categoryData[0]);
    }
    
    return categoryData.reduce((acc, point) => {
      if (!point) return acc;
      const count = parseNumericValue(point.count) || parseNumericValue(point.order_count, 0);
      
      let pointDate = null;
      
      // Handle direct date objects
      if (point.date instanceof Date) {
        pointDate = point.date;
      } 
      // Handle string dates
      else if (typeof point.date === 'string') {
        try {
          pointDate = new Date(point.date);
        } catch (err) {
          console.warn(`Error parsing date string: ${point.date}.`);
        }
      }
      
      // If no valid date and year/month fields exist, try to construct the date.
      if ((!pointDate || isNaN(pointDate.getTime())) && 
          (point.order_year || point.year) && 
          (point.order_month || point.month)) {
        try {
          pointDate = new Date(
            parseInt(point.order_year || point.year),
            parseInt(point.order_month || point.month) - 1,
            1
          );
        } catch (err) {
          console.warn(`Error constructing date from year/month for point: ${JSON.stringify(point)}.`);
          pointDate = null;
        }
      }
      
      // Last resort: generate date from index position
      if (!pointDate || isNaN(pointDate.getTime())) {
        if (acc.length > 0) {
          // If we have previous points, add a month to the last one
          const lastDate = new Date(acc[acc.length-1].date);
          lastDate.setMonth(lastDate.getMonth() + 1);
          pointDate = lastDate;
        } else {
          // Start with the current month minus 6 months
          const date = new Date();
          date.setMonth(date.getMonth() - 6);
          pointDate = date;
        }
        console.warn(`Generated synthetic date for data point with count ${count}`);
      }
      
      acc.push({
        date: pointDate,
        value: count,
        type: 'historical'
      });
      return acc;
    }, []).sort((a, b) => a.date - b.date);
  };

  // Set initial category if not already set
  useEffect(() => {
    if (validForecastReport.length > 0 && !selectedCategory && !isInitialized) {
      const firstValid = validForecastReport.find(f => f && (f.category || f.product_category || f.product_category_name));
      if (firstValid) {
        const categoryName = firstValid.category || firstValid.product_category || firstValid.product_category_name;
        setSelectedCategory(categoryName);
        setIsInitialized(true);
      }
    }
  }, [validForecastReport, selectedCategory, isInitialized]);

  // Process forecast data when dependencies change.
  useEffect(() => {
    const processForecastData = () => {
      if (!selectedCategory) {
        setForecastData([]);
        setHasVisualizationData(false);
        setForecastNote('Please select a category');
        return;
      }
      
      const forecastReport = validForecastReport;
      const categoryData = getCategoryData(safeData, selectedCategory);
      const historicalData = processHistoricalData(categoryData);
      
      console.log(`Processing forecast data for category: ${selectedCategory}`);
      console.log(`Historical data points: ${historicalData.length}`);
      
      const categoryForecast = forecastReport.find(forecast =>
        forecast &&
        (forecast.category === selectedCategory ||
         forecast.product_category === selectedCategory ||
         forecast.product_category_name === selectedCategory)
      );
      
      console.log(`Found category forecast: ${categoryForecast ? 'Yes' : 'No'}`);
      if (categoryForecast) {
        console.log(`Forecast details for ${selectedCategory}:`, {
          avg_historical_demand: categoryForecast.avg_historical_demand,
          forecast_demand: categoryForecast.forecast_demand || categoryForecast.next_month_forecast,
          growth_rate: categoryForecast.growth_rate
        });
      }
      
      // Define fallback function (within the correct scope)
      const fallbackToHistorical = (note) => {
        if (historicalData.length > 0) {
          setForecastData(historicalData);
          setHasVisualizationData(true);
          setForecastNote(note);
        } else if (categoryForecast) {
          // If we have forecast data but no historical data, we can still show something
          const baseDate = new Date();
          baseDate.setMonth(baseDate.getMonth() - 1);
          
          const baseValue = parseNumericValue(categoryForecast.avg_historical_demand, 100);
          const basePoint = { 
            date: baseDate, 
            value: baseValue, 
            type: 'historical' 
          };
          
          const syntheticHistoricalData = [basePoint];
          setForecastData(syntheticHistoricalData);
          setHasVisualizationData(true);
          setForecastNote("Limited historical data available. Showing synthesized forecast visualization.");
        } else {
          setForecastData([]);
          setHasVisualizationData(false);
          setForecastNote(note);
        }
      };
      
      // Even with limited data, we'll try to provide visualization
      // The backend code may fall back to alternative methods
      if (categoryForecast) {
        const forecasts = categoryForecast?.forecast_values || [];
        const visualizationThreshold = historicalData.length === 0 ? 0 : 4;
        
        if (historicalData.length < visualizationThreshold && forecasts.length === 0) {
          console.log(`Limited data for ${selectedCategory}, visualizationThreshold=${visualizationThreshold}`);
          fallbackToHistorical("Limited historical data. Showing basic forecast interpretation.");
          return;
        }
        
        try {
          const lastPoint = historicalData.length > 0 ? 
            historicalData[historicalData.length - 1] : null;
            
          const forecastValue = parseNumericValue(categoryForecast.forecast_demand || 
                                                 categoryForecast.next_month_forecast, 0);
          const growthRate = categoryForecast.growth_rate != null
            ? Math.max(Math.min(parseNumericValue(categoryForecast.growth_rate, 0), 100), -80) / 100
            : 0;
          const forecastPoints = [];
          
          // If we have no historical data but have forecast info, create a synthetic historical point
          if (!lastPoint && forecastValue > 0) {
            const baseDate = new Date();
            baseDate.setMonth(baseDate.getMonth() - 1);
            
            const baseValue = parseNumericValue(categoryForecast.avg_historical_demand, 
                                                forecastValue * 0.9);
            
            historicalData.push({
              date: baseDate,
              value: baseValue,
              type: 'historical'
            });
            
            console.log("Created synthetic historical data point:", historicalData[0]);
          }
          
          // Generate forecast points for the next 6 months.
          for (let i = 1; i <= 6; i++) {
            // When we have historical data, base forecast on that
            const forecastDate = historicalData.length > 0 ? 
              new Date(historicalData[historicalData.length - 1].date) : new Date();
              
            forecastDate.setMonth(forecastDate.getMonth() + i);
            
            // Get a base value to work from
            let baseValue;
            if (historicalData.length > 0) {
              baseValue = historicalData[historicalData.length - 1].value;
            } else if (forecastValue > 0) {
              baseValue = forecastValue / (1 + growthRate);
            } else {
              baseValue = parseNumericValue(categoryForecast.avg_historical_demand, 100);
            }
            
            let pointValue;
            if (i === 1 && forecastValue > 0) {
              pointValue = forecastValue;
            } else {
              if (growthRate < -0.5) {
                const decayFactor = Math.pow(1 + growthRate, i);
                pointValue = Math.max(baseValue * decayFactor, 1);
              } else if (growthRate > 0) {
                const adjustedGrowth = growthRate / (1 + 0.2 * (i - 1));
                pointValue = baseValue * (1 + adjustedGrowth * i);
              } else {
                pointValue = Math.max(baseValue * Math.pow(1 + growthRate, i), 1);
              }
            }
            
            const mape = categoryForecast.mape != null
              ? Math.min(parseNumericValue(categoryForecast.mape, 20), 100) / 100
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
          
          const mapeValue = parseNumericValue(categoryForecast.mape, 0);
          if (mapeValue > 50) {
            setForecastNote('Note: High forecast uncertainty. Consider this visualization as indicative only.');
          } else if (mapeValue > 25) {
            setForecastNote('Note: Moderate forecast uncertainty. Treat projections as directional estimates.');
          } else if (historicalData.length === 0 || historicalData.length === 1) {
            setForecastNote('Note: Limited historical data. Forecast is based on available category metrics.');
          } else {
            setForecastNote('');
          }
        } catch (error) {
          console.error("Error calculating forecast data:", error);
          fallbackToHistorical('Error generating forecast. Showing available data only.');
        }
      } else {
        if (historicalData.length > 0) {
          fallbackToHistorical('No forecast data available for this category. Showing historical data only.');
        } else {
          fallbackToHistorical('No historical data available for this category.');
        }
      }
    };

    processForecastData();
  }, [selectedCategory, safeData, validForecastReport]);

  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
  };

  const formatDate = (date) => {
    if (!date) return '';
    let d;
    try {
      d = typeof date === 'string' ? new Date(date) : date;
    } catch (error) {
      console.warn(`Error converting date: ${date}`, error);
      return 'Invalid Date';
    }
    return isNaN(d.getTime()) ? 'Invalid Date' : d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  // Moved the getCurrentCategoryForecast function inside the scope where it's used.
  const currentCategoryForecast = useMemo(() => {
    if (!safeData || !safeData.forecastReport) return [];
    return getValidArray(safeData.forecastReport)
      .filter(f => f && (
        f.category === selectedCategory ||
        f.product_category === selectedCategory ||
        f.product_category_name === selectedCategory
      ))
      .map(forecast => ({
        ...forecast,
        avg_historical_demand: Math.max(parseNumericValue(forecast.avg_historical_demand, 0), 0),
        forecast_demand: Math.max(parseNumericValue(forecast.forecast_demand, 0), 0),
        growth_rate: parseNumericValue(forecast.growth_rate, 0),
        mape: parseNumericValue(forecast.mape),
        rmse: parseNumericValue(forecast.rmse),
        mae: parseNumericValue(forecast.mae)
      }));
  }, [safeData, selectedCategory]);

  const dataQuality = currentCategoryForecast && currentCategoryForecast[0] ? currentCategoryForecast[0].data_quality : null;
  
  // If no forecast or performance metrics exist, display an alert.
  if (!safeData.forecastReport && !safeData.performanceMetrics) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Demand Forecasting
        </Typography>
        <Alert severity="info">
          <AlertTitle>No forecast data available</AlertTitle>
          No forecast data has been generated yet. Please run the analysis to generate forecasts.
        </Alert>
      </Box>
    );
  }

  // Render the forecast chart.
  const renderForecastChart = () => {
    if (!hasVisualizationData || forecastData.length === 0) {
      // Even without data, let's check if we have forecast information to show
      const categoryForecast = currentCategoryForecast && currentCategoryForecast.length > 0 
          ? currentCategoryForecast[0] : null;
          
      if (categoryForecast) {
        return (
          <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" height="100%">
            <Typography variant="body1" sx={{ mb: 2 }}>
              {forecastNote || "No historical data available for visualization"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Forecast growth rate: {categoryForecast.growth_rate?.toFixed(2)}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Average historical demand: {Math.round(categoryForecast.avg_historical_demand || 0)} units
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Next month forecast: {Math.round(categoryForecast.forecast_demand || categoryForecast.next_month_forecast || 0)} units
            </Typography>
          </Box>
        );
      }
      
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <Typography variant="body1" color="text.secondary">
            {forecastNote || "No historical data available for this category"}
          </Typography>
        </Box>
      );
    }
    
    return (
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
    );
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
                {getValidArray(safeData.forecastReport)
                  .filter(forecast => forecast && (forecast.category || forecast.product_category || forecast.product_category_name))
                  .map(forecast => {
                    const categoryName = forecast.category || forecast.product_category || forecast.product_category_name;
                    return (
                      <MenuItem key={categoryName} value={categoryName}>
                        {categoryName}
                      </MenuItem>
                    );
                  })}
              </Select>
            </FormControl>
          </Paper>
        </Grid>

        {/* Data Quality Alert */}
        {dataQuality === 'Limited' && (
          <Grid item xs={12}>
            <Alert severity="warning">
              <AlertTitle>Limited Historical Data</AlertTitle>
              The '{selectedCategory}' category has limited historical data points, which may affect forecast accuracy. A basic forecast will be generated based on available data.
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
              {renderForecastChart()}
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Forecast Details */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, height: 500, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Statistics
            </Typography>
            {getValidArray(currentCategoryForecast).map((forecast, index) => (
              <Box key={`${forecast.category}-${index}`} sx={{ mb: 2 }}>
                {/* Render forecast detail cards (omitted for brevity) */}
              </Box>
            ))}
            <ForecastInterpretation forecast={getValidArray(currentCategoryForecast)[0]} />
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