import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  CardHeader, Divider, List, ListItem, ListItemText,
  FormControl, InputLabel, Select, MenuItem,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, BarChart, Bar, Cell,
  PieChart, Pie, Sector
} from 'recharts';

/**
 * Product Categories Page Component
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Category data
 */
const ProductCategoriesPage = ({ data }) => {
  const [selectedCategory, setSelectedCategory] = useState('');
  const [categoryData, setCategoryData] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  
  useEffect(() => {
    if (data && data.topCategories && data.topCategories.length > 0) {
      // Set the initial category selection if not already set
      if (!selectedCategory) {
        setSelectedCategory(data.topCategories[0]);
      }
      
      // Calculate total demand by category for pie chart
      const categoryDemand = data.topCategories.map(category => {
        const categoryRows = data.categoryData[category] || [];
        const totalDemand = categoryRows.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0);
        
        return {
          name: category,
          value: totalDemand
        };
      });
      
      setCategoryData(categoryDemand);
    }
  }, [data, selectedCategory]);
  
  // Handle category change
  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
  };
  
  // Handle pie chart hover
  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };
  
  // Render active shape for pie chart
  const renderActiveShape = (props) => {
    const RADIAN = Math.PI / 180;
    const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle,
      fill, payload, percent, value } = props;
    const sin = Math.sin(-RADIAN * midAngle);
    const cos = Math.cos(-RADIAN * midAngle);
    const sx = cx + (outerRadius + 10) * cos;
    const sy = cy + (outerRadius + 10) * sin;
    const mx = cx + (outerRadius + 30) * cos;
    const my = cy + (outerRadius + 30) * sin;
    const ex = mx + (cos >= 0 ? 1 : -1) * 22;
    const ey = my;
    const textAnchor = cos >= 0 ? 'start' : 'end';
    
    return (
      <g>
        <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>
          {payload.name}
        </text>
        <Sector
          cx={cx}
          cy={cy}
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          startAngle={startAngle}
          endAngle={endAngle}
          fill={fill}
        />
        <Sector
          cx={cx}
          cy={cy}
          startAngle={startAngle}
          endAngle={endAngle}
          innerRadius={outerRadius + 6}
          outerRadius={outerRadius + 10}
          fill={fill}
        />
        <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
        <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333">
          {`${new Intl.NumberFormat().format(value)}`}
        </text>
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999">
          {`(${(percent * 100).toFixed(2)}%)`}
        </text>
      </g>
    );
  };
  
  // Get monthly data for selected category
  const getMonthlyData = () => {
    if (!selectedCategory || !data || !data.categoryData) return [];
    
    const categoryRows = data.categoryData[selectedCategory] || [];
    
    return categoryRows.map(row => ({
      ...row,
      month: new Date(row.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
    })).sort((a, b) => new Date(a.date) - new Date(b.date));
  };
  
  // Calculate growth metrics for selected category
  const getCategoryGrowth = () => {
    const monthlyData = getMonthlyData();
    
    if (monthlyData.length < 2) return { growth: 0, trend: 'flat' };
    
    // If we have at least 6 data points, compare the average of first 3 and last 3 months
    if (monthlyData.length >= 6) {
      const firstThreeAvg = monthlyData.slice(0, 3).reduce((sum, row) => 
        sum + (row.count || row.order_count || 0), 0) / 3;
      
      const lastThreeAvg = monthlyData.slice(-3).reduce((sum, row) => 
        sum + (row.count || row.order_count || 0), 0) / 3;
      
      // Cap growth at reasonable bounds (-99% to +1000%)
      const growth = firstThreeAvg > 0 ? 
        Math.max(Math.min(((lastThreeAvg - firstThreeAvg) / firstThreeAvg) * 100, 1000), -99) : 0;
      
      return {
        growth: growth.toFixed(2),
        trend: growth > 5 ? 'up' : growth < -5 ? 'down' : 'flat',
        firstMonth: firstThreeAvg,
        lastMonth: lastThreeAvg
      };
    }
    
    // For fewer data points, use the original calculation but with capping
    const firstMonth = monthlyData[0].count || monthlyData[0].order_count || 0;
    const lastMonth = monthlyData[monthlyData.length - 1].count || monthlyData[monthlyData.length - 1].order_count || 0;
    
    // Cap growth at reasonable bounds (-99% to +1000%)
    const growth = firstMonth > 0 ? 
      Math.max(Math.min(((lastMonth - firstMonth) / firstMonth) * 100, 1000), -99) : 0;
    
    return {
      growth: growth.toFixed(2),
      trend: growth > 5 ? 'up' : growth < -5 ? 'down' : 'flat',
      firstMonth,
      lastMonth
    };
  };
  
  if (!data) {
    return <Typography>No category data available</Typography>;
  }
  
  const monthlyData = getMonthlyData();
  const growthMetrics = getCategoryGrowth();
  
  // Colors for the charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A4DE6C', '#8884D8', '#82CA9D'];
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Product Category Analysis
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
                {data.topCategories.map(category => (
                  <MenuItem key={category} value={category}>
                    {category}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>
        </Grid>
        
        {/* Category Demand Distribution */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Category Demand Distribution
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  activeIndex={activeIndex}
                  activeShape={renderActiveShape}
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  onMouseEnter={onPieEnter}
                >
                  {categoryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Category Metrics */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              {selectedCategory} Metrics
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, p: 2, flexGrow: 1 }}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    Total Demand
                  </Typography>
                  <Typography variant="h5">
                    {new Intl.NumberFormat().format(
                      monthlyData.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0)
                    )}
                  </Typography>
                </CardContent>
              </Card>
              
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    Average Monthly Demand
                  </Typography>
                  <Typography variant="h5">
                    {monthlyData.length > 0 ? 
                      new Intl.NumberFormat().format(
                        Math.round(monthlyData.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0) / monthlyData.length)
                      ) : '0'}
                  </Typography>
                </CardContent>
              </Card>
              
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    Growth Rate
                  </Typography>
                  <Typography variant="h5" color={
                    growthMetrics.trend === 'up' ? 'success.main' : 
                    growthMetrics.trend === 'down' ? 'error.main' : 
                    'text.primary'
                  }>
                    {growthMetrics.growth}%
                    {growthMetrics.trend === 'up' ? ' ↑' : growthMetrics.trend === 'down' ? ' ↓' : ''}
                  </Typography>
                </CardContent>
              </Card>
              
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    Data Points
                  </Typography>
                  <Typography variant="h5">
                    {monthlyData.length}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Paper>
        </Grid>
        
        {/* Monthly Trend */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              {selectedCategory} Monthly Trend
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={monthlyData}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="month" 
                  angle={-45} 
                  textAnchor="end"
                  height={70}
                />
                <YAxis />
                <Tooltip
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  labelFormatter={(label) => `Month: ${label}`}
                />
                <Legend />
                <Bar 
                  dataKey="count" 
                  name="Orders" 
                  fill="#8884d8"
                />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Monthly Data Table */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              {selectedCategory} Monthly Data
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Month</TableCell>
                    <TableCell align="right">Orders</TableCell>
                    <TableCell align="right">Month-over-Month Change</TableCell>
                    <TableCell align="right">% Change</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monthlyData.map((row, index) => {
                    const prevCount = index > 0 ? monthlyData[index - 1].count || monthlyData[index - 1].order_count || 0 : 0;
                    const currentCount = row.count || row.order_count || 0;
                    const change = index > 0 ? currentCount - prevCount : 0;
                    const percentChange = prevCount > 0 ? (change / prevCount) * 100 : 0;
                    
                    return (
                      <TableRow key={index}>
                        <TableCell>{row.month}</TableCell>
                        <TableCell align="right">{new Intl.NumberFormat().format(currentCount)}</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ color: change > 0 ? 'success.main' : change < 0 ? 'error.main' : 'text.primary' }}
                        >
                          {index > 0 ? `${change > 0 ? '+' : ''}${new Intl.NumberFormat().format(change)}` : '-'}
                        </TableCell>
                        <TableCell 
                          align="right"
                          sx={{ color: percentChange > 0 ? 'success.main' : percentChange < 0 ? 'error.main' : 'text.primary' }}
                        >
                          {index > 0 ? `${percentChange > 0 ? '+' : ''}${percentChange.toFixed(2)}%` : '-'}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ProductCategoriesPage;