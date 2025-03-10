import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

/**
 * Top Categories Chart Component
 * 
 * @param {Object} props Component props
 * @param {Array} props.categories Array of category names
 * @param {Object} props.categoryData Object with category data
 */
const TopCategoriesChart = ({ categories, categoryData }) => {
  // Calculate total demand for each category
  const chartData = useMemo(() => {
    if (!categories || !categoryData) return [];
    
    return categories.map(category => {
      const categoryRows = categoryData[category] || [];
      const totalDemand = categoryRows.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0);
      
      return {
        name: category,
        value: totalDemand
      };
    });
  }, [categories, categoryData]);
  
  // Colors for the pie chart
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A4DE6C'];
  
  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value) => new Intl.NumberFormat().format(value)}
          />
          <Legend layout="vertical" verticalAlign="bottom" align="center" />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TopCategoriesChart;