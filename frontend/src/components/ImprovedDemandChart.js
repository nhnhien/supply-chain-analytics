import React, { useState } from 'react';
import { 
  Box, Typography, Paper, FormControl, InputLabel, Select, MenuItem, Tooltip, Checkbox, FormGroup, FormControlLabel
} from '@mui/material';
import { TrendingUpIcon } from '@mui/icons-material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip as RechartsTooltip, Legend, ResponsiveContainer 
} from 'recharts';

const ImprovedDemandChart = ({ demandData, categories, theme }) => {
  // State to track which categories are visible
  const [visibleCategories, setVisibleCategories] = useState(
    // Initially show only top 3 categories
    (categories.topCategories || []).slice(0, 3).reduce((acc, cat) => {
      acc[cat] = true;
      return acc;
    }, {})
  );
  
  // State for grouping option
  const [groupingOption, setGroupingOption] = useState('individual');
  
  // Chart colors for consistency
  const chartColors = [
    '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c', 
    '#d884b2', '#c49c94', '#8dd1e1', '#b0b0d8', '#e6c57a'
  ];
  
  // Handle category visibility toggle
  const handleCategoryToggle = (category) => {
    setVisibleCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };
  
  // Handle grouping option change
  const handleGroupingChange = (event) => {
    setGroupingOption(event.target.value);
  };
  
  // Group data by month for the "grouped" view
  const getGroupedData = () => {
    if (!demandData || !demandData.length) return [];
    
    // Create a map of date to total count
    const groupedMap = demandData.reduce((acc, item) => {
      const dateKey = item.date instanceof Date 
        ? item.date.toISOString().split('T')[0] 
        : item.date;
      
      if (!acc[dateKey]) {
        acc[dateKey] = { 
          date: item.date,
          total: 0,
          categories: {}
        };
      }
      
      acc[dateKey].total += item.count || 0;
      
      // Also track individual category counts for the stacked view
      if (item.category) {
        if (!acc[dateKey].categories[item.category]) {
          acc[dateKey].categories[item.category] = 0;
        }
        acc[dateKey].categories[item.category] += item.count || 0;
      }
      
      return acc;
    }, {});
    
    // Convert to array and sort by date
    return Object.values(groupedMap).sort((a, b) => {
      return new Date(a.date) - new Date(b.date);
    });
  };
  
  // Filter categories to only show selected ones
  const getVisibleCategoriesData = () => {
    return (categories.topCategories || [])
      .filter(category => visibleCategories[category])
      .map((category, index) => ({
        category,
        data: categories.categoryData?.[category] || [],
        color: chartColors[index % chartColors.length]
      }));
  };
  
  // Prepare data based on view option
  const groupedData = getGroupedData();
  const visibleCategoriesData = getVisibleCategoriesData();

  return (
    <Paper elevation={3} sx={{ 
      p: 4, 
      display: 'flex', 
      flexDirection: 'column', 
      height: 450,
      borderRadius: 2,
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
    }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography component="h2" variant="h6" sx={{ 
          color: theme.palette.primary.main,
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
        }}>
          Historical Monthly Demand (2017-2018)
        </Typography>
        
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>View Mode</InputLabel>
          <Select
            value={groupingOption}
            label="View Mode"
            onChange={handleGroupingChange}
          >
            <MenuItem value="individual">Individual Categories</MenuItem>
            <MenuItem value="grouped">Total Demand</MenuItem>
            <MenuItem value="multiline">Selected Categories</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      <Box sx={{ display: 'flex', height: 'calc(100% - 40px)' }}>
        {/* Category selection sidebar - only show in multiline mode */}
        {groupingOption === 'multiline' && (
          <Box sx={{ 
            width: '20%', 
            pr: 2, 
            overflow: 'auto',
            borderRight: `1px solid ${theme.palette.divider}` 
          }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Select Categories:</Typography>
            <FormGroup>
              {(categories.topCategories || []).slice(0, 10).map((category, index) => (
                <FormControlLabel
                  key={category}
                  control={
                    <Checkbox
                      checked={!!visibleCategories[category]}
                      onChange={() => handleCategoryToggle(category)}
                      size="small"
                      sx={{ 
                        color: chartColors[index % chartColors.length],
                        '&.Mui-checked': {
                          color: chartColors[index % chartColors.length],
                        }
                      }}
                    />
                  }
                  label={
                    <Typography variant="body2" noWrap sx={{ maxWidth: '150px' }}>
                      {category}
                    </Typography>
                  }
                />
              ))}
            </FormGroup>
          </Box>
        )}
        
        {/* Chart area */}
        <Box sx={{ 
          width: groupingOption === 'multiline' ? '80%' : '100%',
          height: '100%' 
        }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              data={groupingOption === 'grouped' ? groupedData : null}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="date" 
                tickFormatter={(date) => {
                  if (date instanceof Date) {
                    return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                  }
                  return '';
                }}
                stroke={theme.palette.text.secondary}
              />
              <YAxis stroke={theme.palette.text.secondary} />
              <RechartsTooltip 
                formatter={(value) => new Intl.NumberFormat().format(value)}
                labelFormatter={(label) => {
                  if (label instanceof Date) {
                    return label.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
                  }
                  return label;
                }}
                contentStyle={{ 
                  backgroundColor: theme.palette.background.paper,
                  borderColor: theme.palette.divider,
                  borderRadius: 8,
                  boxShadow: theme.shadows[3]
                }}
              />
              <Legend 
                verticalAlign="bottom" 
                height={36} 
                wrapperStyle={{ paddingTop: '10px' }}
              />
              
              {/* Render based on selected view mode */}
              {groupingOption === 'grouped' && (
                <Line 
                  type="monotone" 
                  dataKey="total"
                  name="Total Demand"
                  stroke={theme.palette.primary.main}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6, stroke: theme.palette.background.paper, strokeWidth: 2 }}
                />
              )}
              
              {groupingOption === 'multiline' && visibleCategoriesData.map(({ category, data, color }) => (
                <Line 
                  key={category}
                  type="monotone" 
                  dataKey="count"
                  data={data}
                  name={category}
                  stroke={color}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 6, stroke: theme.palette.background.paper, strokeWidth: 2 }}
                />
              ))}
              
              {groupingOption === 'individual' && categories.topCategories && (
                <Line 
                  type="monotone" 
                  dataKey="count"
                  data={categories.categoryData?.[categories.topCategories[0]] || []}
                  name={categories.topCategories[0] || "Top Category"}
                  stroke={chartColors[0]}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6, stroke: theme.palette.background.paper, strokeWidth: 2 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Box>
    </Paper>
  );
};

export default ImprovedDemandChart;