import React, { useMemo } from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';

/**
 * Top Categories Chart Component
 * 
 * @param {Object} props Component props
 * @param {Array} props.categories Array of category names
 * @param {Object} props.categoryData Object with category data, where each key corresponds to a category name and its value is an array of data rows.
 * @param {String} props.chartType Type of chart to display ('pie' or 'bar', defaults to 'bar')
 * @param {Array} props.chartColors Optional array of colors to use for the chart
 */
const TopCategoriesChart = ({ categories, categoryData, chartType = 'bar', chartColors }) => {
  const theme = useTheme();
  
  // Define default colors based on theme if not provided
  const COLORS = chartColors || [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
    theme.palette.grey[500]
  ];
  
  // Calculate total demand for each category with robust error handling.
  const chartData = useMemo(() => {
    // Ensure categories is a valid array and categoryData is a non-null object.
    if (!Array.isArray(categories) || typeof categoryData !== 'object' || categoryData === null) return [];
    
    return categories.map(category => {
      // Safely access category data.
      const categoryRows = Array.isArray(categoryData?.[category]) ? categoryData[category] : [];
      
      // Sum up demand values (using parseFloat to ensure numeric addition)
      const totalDemand = categoryRows.reduce((sum, row) => {
        const count = parseFloat(row.count) || parseFloat(row.order_count) || 0;
        return sum + count;
      }, 0);
      
      return {
        name: category,
        value: totalDemand
      };
    }).sort((a, b) => b.value - a.value); // Sort by value in descending order for bar chart
  }, [categories, categoryData]);

  // Compute the sum of all demand values.
  const totalDemandSum = chartData.reduce((sum, d) => sum + d.value, 0);

  // If total demand is zero, render a friendly message.
  if (totalDemandSum === 0) {
    return (
      <Box 
        sx={{ 
          width: '100%', 
          height: '100%', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 2,
          color: theme.palette.text.secondary,
          p: 3
        }}
      >
        <Box
          component="span"
          sx={{
            display: 'inline-block',
            p: 1,
            borderRadius: '50%',
            bgcolor: theme.palette.action.hover,
            width: 60,
            height: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <Box
            component="span"
            sx={{
              fontSize: '2rem',
              lineHeight: 1
            }}
          >
            📊
          </Box>
        </Box>
        <Typography variant="h6" color="text.secondary" align="center">
          No demand data available
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          Run the analysis to generate category data
        </Typography>
      </Box>
    );
  }

  // Format the value for display
  const formatValue = (value) => {
    return new Intl.NumberFormat().format(value);
  };
  
  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 80, bottom: 30 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            type="number" 
            tick={{ fill: theme.palette.text.secondary }}
            tickFormatter={value => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
          />
          <YAxis 
            type="category" 
            dataKey="name" 
            tick={{ 
              fill: theme.palette.text.primary, 
              fontSize: 12, 
              fontWeight: 500 
            }}
            width={80}
          />
          <Tooltip 
            formatter={(value) => [`${formatValue(value)} units`, "Demand"]} 
            labelFormatter={(name) => `Category: ${name}`}
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '8px',
              boxShadow: theme.shadows[3],
              padding: '10px 14px'
            }}
            itemStyle={{
              padding: '4px 0'
            }}
            labelStyle={{
              fontWeight: 'bold',
              marginBottom: '6px',
              borderBottom: `1px solid ${theme.palette.divider}`,
              paddingBottom: '6px'
            }}
          />
          <Legend 
            verticalAlign="bottom" 
            height={36}
            wrapperStyle={{
              paddingTop: '15px',
              fontWeight: 500
            }}
          />
          <Bar 
            dataKey="value" 
            name="Demand" 
            fill={theme.palette.primary.main}
            radius={[0, 4, 4, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={COLORS[index % COLORS.length]} 
                stroke={theme.palette.background.paper}
                strokeWidth={1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TopCategoriesChart;