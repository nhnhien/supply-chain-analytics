import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';

/**
 * Top Categories Chart Component
 * 
 * @param {Object} props Component props
 * @param {Array} props.categories Array of category names
 * @param {Object} props.categoryData Object with category data, where each key corresponds to a category name and its value is an array of data rows.
 * @param {String} props.chartType Type of chart to display ('pie' or 'bar', defaults to 'bar')
 */
const TopCategoriesChart = ({ categories, categoryData, chartType = 'bar' }) => {
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
      <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No demand data available
        </Typography>
      </Box>
    );
  }

  // Colors for the chart
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A4DE6C', '#8884D8', '#82CA9D'];
  
  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis 
            type="category" 
            dataKey="name" 
            tick={{ fontSize: 12 }}
            width={80}
          />
          <Tooltip 
            formatter={(value) => new Intl.NumberFormat().format(value)} 
            labelFormatter={(name) => `Category: ${name}`}
          />
          <Legend />
          <Bar 
            dataKey="value" 
            name="Demand" 
            fill="#8884d8"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TopCategoriesChart;