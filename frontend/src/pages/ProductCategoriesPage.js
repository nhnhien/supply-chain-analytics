import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  CardHeader, Divider, List, ListItem, ListItemText,
  FormControl, InputLabel, Select, MenuItem,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  useTheme, Chip, Avatar, Tooltip, IconButton
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, BarChart, Bar, Cell,
  PieChart, Pie, Sector
} from 'recharts';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  ShowChart as ShowChartIcon,
  PieChart as PieChartIcon,
  Category as CategoryIcon,
  ListAlt as ListAltIcon,
  Info as InfoIcon
} from '@mui/icons-material';

/**
 * Product Categories Page Component
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Category data
 */
const ProductCategoriesPage = ({ data }) => {
  const theme = useTheme();
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
        <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill} fontWeight="bold">
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
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333" fontWeight="bold">
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
  
  console.log("Growth calculation started with monthly data:", monthlyData);
  
  if (monthlyData.length < 2) {
    console.log("Insufficient data points for growth calculation (need at least 2)");
    return { growth: 0, trend: 'flat' };
  }
  
  let growth, firstValue, lastValue;
  
  // If we have at least 6 data points, compare the average of first 3 and last 3 months
  if (monthlyData.length >= 6) {
    const firstThreeAvg = monthlyData.slice(0, 3).reduce((sum, row) => 
      sum + (row.count || row.order_count || 0), 0) / 3;
    
    const lastThreeAvg = monthlyData.slice(-3).reduce((sum, row) => 
      sum + (row.count || row.order_count || 0), 0) / 3;
    
    console.log("Using 3-month average method:");
    console.log("- First three months avg:", firstThreeAvg);
    console.log("- Last three months avg:", lastThreeAvg);
    
    // Cap growth at reasonable bounds (-99% to +1000%)
    growth = firstThreeAvg > 0 ? 
      Math.max(Math.min(((lastThreeAvg - firstThreeAvg) / firstThreeAvg) * 100, 1000), -99) : 0;
    
    firstValue = firstThreeAvg;
    lastValue = lastThreeAvg;
  } else {
    // For fewer data points, use the original calculation but with capping
    const firstMonth = monthlyData[0].count || monthlyData[0].order_count || 0;
    const lastMonth = monthlyData[monthlyData.length - 1].count || monthlyData[monthlyData.length - 1].order_count || 0;
    
    console.log("Using first-to-last month method:");
    console.log("- First month count:", firstMonth);
    console.log("- Last month count:", lastMonth);
    
    // Cap growth at reasonable bounds (-99% to +1000%)
    growth = firstMonth > 0 ? 
      Math.max(Math.min(((lastMonth - firstMonth) / firstMonth) * 100, 1000), -99) : 0;
    
    firstValue = firstMonth;
    lastValue = lastMonth;
  }
  
  console.log("Raw growth rate calculation:", growth);
  
  const result = {
    growth: growth.toFixed(2),
    trend: growth > 5 ? 'up' : growth < -5 ? 'down' : 'flat',
    firstMonth: firstValue,
    lastMonth: lastValue
  };
  
  console.log("Final growth calculation result:", result);
  
  return result;
};
  
  if (!data) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h5" color="text.secondary">No category data available</Typography>
      </Box>
    );
  }
  
  const monthlyData = getMonthlyData();
  const growthMetrics = getCategoryGrowth();
  
  // Colors for the charts
  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
    '#8884D8'
  ];
  
  // Gets trend icon based on trend direction
  const getTrendIcon = (trend) => {
    switch(trend) {
      case 'up':
        return <TrendingUpIcon fontSize="small" sx={{ color: theme.palette.success.main }} />;
      case 'down':
        return <TrendingDownIcon fontSize="small" sx={{ color: theme.palette.error.main }} />;
      default:
        return <TrendingFlatIcon fontSize="small" sx={{ color: theme.palette.text.secondary }} />;
    }
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Typography variant="h4" gutterBottom sx={{ 
        fontWeight: 'bold',
        color: theme.palette.primary.main,
        borderBottom: `2px solid ${theme.palette.divider}`,
        pb: 1,
        mb: 3,
        display: 'flex',
        alignItems: 'center'
      }}>
        <CategoryIcon sx={{ mr: 1 }} />
        Product Category Analysis
      </Typography>
      
      <Grid container spacing={3}>
        {/* Category Selector */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
          }}>
            <Box sx={{ 
              display: 'flex', 
              flexDirection: { xs: 'column', sm: 'row' },
              alignItems: { xs: 'flex-start', sm: 'center' },
              justifyContent: 'space-between',
              mb: { xs: 2, sm: 0 }
            }}>
              <Typography component="h2" variant="h6" gutterBottom={false} sx={{ 
                color: theme.palette.primary.main,
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                mb: { xs: 2, sm: 0 }
              }}>
                <CategoryIcon sx={{ mr: 1 }} /> Select Product Category
              </Typography>
              
              {selectedCategory && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip 
                    icon={getTrendIcon(growthMetrics.trend)} 
                    label={`Growth: ${growthMetrics.growth}%`} 
                    color={growthMetrics.trend === 'up' ? 'success' : growthMetrics.trend === 'down' ? 'error' : 'default'} 
                    size="small"
                    variant="outlined"
                  />
                </Box>
              )}
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <FormControl fullWidth variant="outlined" sx={{ mt: 1 }}>
              <InputLabel id="category-select-label">Product Category</InputLabel>
              <Select
                labelId="category-select-label"
                id="category-select"
                value={selectedCategory}
                label="Product Category"
                onChange={handleCategoryChange}
                sx={{ 
                  '& .MuiSelect-select': { 
                    display: 'flex', 
                    alignItems: 'center' 
                  }
                }}
              >
                {data.topCategories.map(category => (
                  <MenuItem key={category} value={category} sx={{ 
                    display: 'flex', 
                    alignItems: 'center' 
                  }}>
                    <CategoryIcon sx={{ mr: 1, fontSize: 20, color: 'primary.light' }} />
                    {category}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>
        </Grid>
        
        {/* Main content section heading */}
        <Grid item xs={12}>
          <Box sx={{ mt: 2, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.primary.main}`,
              pl: 2
            }}>
              Category Overview
            </Typography>
          </Box>
        </Grid>
        
        {/* Category Demand Distribution */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            height: 400,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.primary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1
            }}>
              <PieChartIcon sx={{ mr: 1 }} /> Category Demand Distribution
            </Typography>
            
            <Box sx={{ flexGrow: 1 }}>
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
                    fill={theme.palette.primary.main}
                    dataKey="value"
                    onMouseEnter={onPieEnter}
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        
        {/* Category Metrics - Fixed the scrolling issue */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            height: 400,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.primary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1
            }}>
              <ShowChartIcon sx={{ mr: 1 }} /> {selectedCategory} Metrics
            </Typography>
            
            {/* Added overflow auto to ensure all cards are viewable */}
            <Box sx={{ height: "calc(100% - 40px)", overflowY: "auto", pr: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card variant="outlined" sx={{ 
                    borderRadius: 1,
                    transition: 'box-shadow 0.3s',
                    '&:hover': { boxShadow: 2 }
                  }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Total Demand
                      </Typography>
                      <Typography variant="h5" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center' }}>
                        {new Intl.NumberFormat().format(
                          monthlyData.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0)
                        )}
                        <Tooltip title="Total units ordered across all time periods">
                          <IconButton size="small" sx={{ ml: 1, p: 0.5 }}>
                            <InfoIcon fontSize="small" color="action" />
                          </IconButton>
                        </Tooltip>
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12}>
                  <Card variant="outlined" sx={{ 
                    borderRadius: 1,
                    transition: 'box-shadow 0.3s',
                    '&:hover': { boxShadow: 2 }
                  }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Average Monthly Demand
                      </Typography>
                      <Typography variant="h5" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center' }}>
                        {monthlyData.length > 0 ? 
                          new Intl.NumberFormat().format(
                            Math.round(monthlyData.reduce((sum, row) => sum + (row.count || row.order_count || 0), 0) / monthlyData.length)
                          ) : '0'}
                        <Tooltip title="Average units ordered per month">
                          <IconButton size="small" sx={{ ml: 1, p: 0.5 }}>
                            <InfoIcon fontSize="small" color="action" />
                          </IconButton>
                        </Tooltip>
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12}>
                  <Card variant="outlined" sx={{ 
                    borderRadius: 1,
                    transition: 'box-shadow 0.3s',
                    '&:hover': { boxShadow: 2 }
                  }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Growth Rate
                      </Typography>
                      <Typography variant="h5" fontWeight="medium" sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        color: growthMetrics.trend === 'up' ? theme.palette.success.main : 
                               growthMetrics.trend === 'down' ? theme.palette.error.main : 
                               theme.palette.text.primary
                      }}>
                        {getTrendIcon(growthMetrics.trend)}
                        <Box component="span" sx={{ ml: 1 }}>
                          {growthMetrics.growth}%
                        </Box>
                        <Tooltip title={
                          monthlyData.length >= 6 
                            ? "Based on comparing first 3 and last 3 months" 
                            : "Based on comparing first and last data points"
                        }>
                          <IconButton size="small" sx={{ ml: 1, p: 0.5 }}>
                            <InfoIcon fontSize="small" color="action" />
                          </IconButton>
                        </Tooltip>
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12}>
                  <Card variant="outlined" sx={{ 
                    borderRadius: 1,
                    transition: 'box-shadow 0.3s',
                    '&:hover': { boxShadow: 2 }
                  }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Data Points
                      </Typography>
                      <Typography variant="h5" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center' }}>
                        {monthlyData.length}
                        <Tooltip title="Number of months with available data">
                          <IconButton size="small" sx={{ ml: 1, p: 0.5 }}>
                            <InfoIcon fontSize="small" color="action" />
                          </IconButton>
                        </Tooltip>
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </Grid>
        
        {/* Monthly Trend */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            height: 400,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.primary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1
            }}>
              <ShowChartIcon sx={{ mr: 1 }} /> {selectedCategory} Historical Monthly Trend
            </Typography>
            
            <Box sx={{ flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={monthlyData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                  <XAxis 
                    dataKey="month" 
                    angle={-45} 
                    textAnchor="end"
                    height={70}
                    tick={{ fill: theme.palette.text.secondary }}
                  />
                  <YAxis 
                    tick={{ fill: theme.palette.text.secondary }}
                  />
                  <RechartsTooltip
                    formatter={(value) => new Intl.NumberFormat().format(value)}
                    labelFormatter={(label) => `Month: ${label}`}
                    contentStyle={{ 
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                      boxShadow: theme.shadows[3]
                    }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '10px' }} />
                  <Bar 
                    dataKey="count" 
                    name="Orders" 
                    fill={theme.palette.primary.main}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        
        {/* Section heading for data table */}
        <Grid item xs={12}>
          <Box sx={{ mt: 4, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.secondary.main}`,
              pl: 2
            }}>
              Detailed Data
            </Typography>
          </Box>
        </Grid>
        
        {/* Monthly Data Table */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ 
            p: 3,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)' 
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.secondary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1
            }}>
              <ListAltIcon sx={{ mr: 1 }} /> {selectedCategory} Monthly Data
            </Typography>
            
            <TableContainer sx={{ mt: 2 }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: theme.palette.action.hover }}>Month</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold', backgroundColor: theme.palette.action.hover }}>Orders</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold', backgroundColor: theme.palette.action.hover }}>Month-over-Month Change</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold', backgroundColor: theme.palette.action.hover }}>% Change</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monthlyData.map((row, index) => {
                    const prevCount = index > 0 ? monthlyData[index - 1].count || monthlyData[index - 1].order_count || 0 : 0;
                    const currentCount = row.count || row.order_count || 0;
                    const change = index > 0 ? currentCount - prevCount : 0;
                    const percentChange = prevCount > 0 ? (change / prevCount) * 100 : 0;
                    
                    return (
                      <TableRow 
                        key={index}
                        sx={{ 
                          '&:nth-of-type(odd)': { 
                            backgroundColor: theme.palette.action.hover
                          },
                          '&:hover': { 
                            backgroundColor: theme.palette.action.selected
                          }
                        }}
                      >
                        <TableCell sx={{ fontWeight: 'medium' }}>{row.month}</TableCell>
                        <TableCell align="right">{new Intl.NumberFormat().format(currentCount)}</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: change > 0 ? theme.palette.success.main : 
                                   change < 0 ? theme.palette.error.main : 
                                   theme.palette.text.primary,
                            fontWeight: index > 0 ? 'medium' : 'normal'
                          }}
                        >
                          {index > 0 ? (
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                              {change > 0 ? (
                                <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} />
                              ) : change < 0 ? (
                                <TrendingDownIcon fontSize="small" sx={{ mr: 0.5 }} />
                              ) : (
                                <TrendingFlatIcon fontSize="small" sx={{ mr: 0.5 }} />
                              )}
                              {`${change > 0 ? '+' : ''}${new Intl.NumberFormat().format(change)}`}
                            </Box>
                          ) : '-'}
                        </TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: percentChange > 0 ? theme.palette.success.main : 
                                   percentChange < 0 ? theme.palette.error.main : 
                                   theme.palette.text.primary,
                            fontWeight: index > 0 ? 'medium' : 'normal'
                          }}
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