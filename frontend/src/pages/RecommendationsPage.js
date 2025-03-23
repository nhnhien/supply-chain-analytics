import React, { useState } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Divider, List, ListItem, ListItemText, ListItemIcon,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, Collapse, IconButton, Alert, AlertTitle, useTheme
} from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer
} from 'recharts';
import {
  Error as WarningIcon,
  CheckCircle as SuccessIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Inventory as InventoryIcon,
  LocalShipping as ShippingIcon,
  AttachMoney as MoneyIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Equalizer as EqualizerIcon,
  Assignment as AssignmentIcon,
  Category as CategoryIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

/**
 * Recommendations Page Component
 * Displays inventory optimization recommendations and reorder strategies
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Recommendations data
 */
const RecommendationsPage = ({ data }) => {
  const theme = useTheme();
  const [expandedCategory, setExpandedCategory] = useState(null);
  
  if (!data || !data.inventory || data.inventory.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ 
          fontWeight: 'bold',
          color: theme.palette.primary.main,
          borderBottom: `2px solid ${theme.palette.divider}`,
          pb: 1,
          display: 'flex',
          alignItems: 'center'
        }}>
          <InventoryIcon sx={{ mr: 1 }} />
          Inventory Recommendations
        </Typography>
        
        <Alert 
          severity="info" 
          variant="filled"
          sx={{ 
            mt: 4,
            boxShadow: theme.shadows[3],
            '& .MuiAlert-icon': {
              fontSize: '2rem'
            }
          }}
        >
          <AlertTitle sx={{ fontWeight: 'bold' }}>No recommendations available</AlertTitle>
          <Typography variant="body1">
            Run the supply chain analysis to generate inventory recommendations.
          </Typography>
        </Alert>
      </Box>
    );
  }
  
  // Toggle category expansion
  const toggleCategory = (category) => {
    if (expandedCategory === category) {
      setExpandedCategory(null);
    } else {
      setExpandedCategory(category);
    }
  };
  
  // Get priority color based on recommendation priority
  const getPriorityColor = (priority) => {
    if (!priority) return 'default';
    
    switch (priority.toLowerCase()) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };
  
  // Get priority icon based on recommendation priority
  const getPriorityIcon = (priority) => {
    if (!priority) return <InfoIcon />;
    
    switch (priority.toLowerCase()) {
      case 'high':
        return <WarningIcon color="error" />;
      case 'medium':
        return <WarningIcon color="warning" />;
      case 'low':
        return <SuccessIcon color="success" />;
      default:
        return <InfoIcon />;
    }
  };
  
  // Format data for bar chart visualization with robust numerical parsing.
  const barChartData = data.inventory.map(item => {
    const category = item.product_category || item.category;
    // Parse safety stock and reorder point as numbers; default to 0 if invalid.
    const safetyStockRaw = parseFloat(item.safety_stock);
    const reorderPointRaw = parseFloat(item.reorder_point);
    const validSafetyStock = isNaN(safetyStockRaw) ? 0 : safetyStockRaw;
    const validReorderPoint = isNaN(reorderPointRaw) ? 0 : reorderPointRaw;
    // Calculate leadTimeDemand ensuring it's not negative.
    const leadTimeDemand = Math.max(validReorderPoint - validSafetyStock, 0);
    // Parse forecast values and default to 0 if invalid.
    const forecastRaw = parseFloat(item.next_month_forecast || item.forecast_demand);
    const nextMonthForecast = isNaN(forecastRaw) ? 0 : forecastRaw;
    
    return {
      name: category,
      safetyStock: validSafetyStock,
      leadTimeDemand,
      nextMonthForecast
    };
  });
  
  // Calculate priority counts
  const priorityCounts = data.inventory.reduce((counts, item) => {
    const priority = item.priority || 
      (item.growth_rate > 10 ? 'High' : 
       item.growth_rate > 0 ? 'Medium' : 'Low');
    
    counts[priority] = (counts[priority] || 0) + 1;
    return counts;
  }, {});
  
  // Convert priority counts object to array for mapping
  const priorityCountsArray = Object.entries(priorityCounts).map(([priority, count]) => ({
    priority,
    count
  }));
  
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
        <InventoryIcon sx={{ mr: 1 }} />
        Inventory Recommendations
      </Typography>
      
      <Grid container spacing={4}>
        {/* Main content section heading */}
        <Grid item xs={12}>
          <Box sx={{ mt: 1, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.primary.main}`,
              pl: 2
            }}>
              Inventory Planning Overview
            </Typography>
          </Box>
        </Grid>
        
        {/* Visual Recommendations */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            height: 450,
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
              <EqualizerIcon sx={{ mr: 1 }} /> Inventory Planning Visualization
            </Typography>
            
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={barChartData.slice(0, 7)} // Limit to top 7 for readability
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end"
                  height={70}
                  tick={{ fill: theme.palette.text.secondary }}
                />
                <YAxis 
                  label={{ 
                    value: 'Units', 
                    angle: -90, 
                    position: 'insideLeft', 
                    style: { fill: theme.palette.text.secondary } 
                  }}
                  tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
                  tick={{ fill: theme.palette.text.secondary }}
                />
                <RechartsTooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: '8px',
                    boxShadow: theme.shadows[3]
                  }}
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '10px', paddingBottom: '20px' }}
                  formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
                />
                <Bar 
                  dataKey="safetyStock" 
                  stackId="inventory" 
                  name="Safety Stock" 
                  fill={theme.palette.primary.main}
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="leadTimeDemand" 
                  stackId="inventory" 
                  name="Lead Time Demand" 
                  fill={theme.palette.secondary.main} 
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="nextMonthForecast" 
                  name="Next Month Forecast" 
                  fill={theme.palette.warning.main} 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Summary Card */}
        <Grid item xs={12} lg={4}>
          <Paper elevation={3} sx={{ 
            p: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            height: 450,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            overflowY: 'auto'
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.primary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1
            }}>
              <AssignmentIcon sx={{ mr: 1 }} /> Summary of Recommendations
            </Typography>
            
            <List>
              {priorityCountsArray.map((item) => (
                <ListItem 
                  key={item.priority}
                  sx={{ 
                    mb: 1,
                    bgcolor: theme.palette.action.hover,
                    borderRadius: 1,
                    border: `1px solid ${theme.palette.divider}`
                  }}
                >
                  <ListItemIcon>
                    {getPriorityIcon(item.priority)}
                  </ListItemIcon>
                  <ListItemText 
                    primary={<Typography fontWeight="medium">{`${item.priority} Priority Items`}</Typography>}
                    secondary={`${item.count} product categories`}
                  />
                  <Chip 
                    label={item.count} 
                    color={getPriorityColor(item.priority)} 
                    size="small"
                  />
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body1" gutterBottom fontWeight="medium">
              Key actions to take:
            </Typography>
            
            <List dense>
              <ListItem sx={{ 
                py: 1, 
                borderLeft: `3px solid ${theme.palette.primary.main}`,
                pl: 2,
                mb: 1
              }}>
                <ListItemIcon>
                  <InventoryIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography fontWeight="medium">Update safety stock levels based on growth patterns</Typography>}
                />
              </ListItem>
              
              <ListItem sx={{ 
                py: 1, 
                borderLeft: `3px solid ${theme.palette.secondary.main}`,
                pl: 2,
                mb: 1
              }}>
                <ListItemIcon>
                  <ShippingIcon color="secondary" />
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography fontWeight="medium">Review supplier agreements for high-growth categories</Typography>}
                />
              </ListItem>
              
              <ListItem sx={{ 
                py: 1, 
                borderLeft: `3px solid ${theme.palette.success.main}`,
                pl: 2,
                mb: 1
              }}>
                <ListItemIcon>
                  <MoneyIcon sx={{ color: theme.palette.success.main }} />
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography fontWeight="medium">Optimize inventory capital allocation</Typography>}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
        
        {/* Detailed section heading */}
        <Grid item xs={12}>
          <Box sx={{ mt: 4, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.secondary.main}`,
              pl: 2
            }}>
              Detailed Recommendations
            </Typography>
          </Box>
        </Grid>
        
        {/* Detailed Recommendations Table */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ 
            p: 3,
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            overflow: 'hidden'
          }}>
            <Typography component="h2" variant="h6" gutterBottom sx={{ 
              color: theme.palette.secondary.main,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 1,
              mb: 2
            }}>
              <StorageIcon sx={{ mr: 1 }} /> Detailed Inventory Recommendations
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ 
                    backgroundColor: theme.palette.action.hover,
                  }}>
                    <TableCell sx={{ fontWeight: 'bold' }}>Category</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Safety Stock</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Reorder Point</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Forecast</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Growth Rate</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Priority</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Recommendation</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.inventory.map((item, index) => {
                    const category = item.product_category || item.category;
                    const safetyStock = parseFloat(item.safety_stock);
                    const reorderPoint = parseFloat(item.reorder_point);
                    const validSafetyStock = isNaN(safetyStock) ? 0 : safetyStock;
                    const validReorderPoint = isNaN(reorderPoint) ? 0 : reorderPoint;
                    const forecast = parseFloat(item.next_month_forecast || item.forecast_demand);
                    const validForecast = isNaN(forecast) ? 0 : forecast;
                    const growthRate = item.growth_rate || 0;
                    const priority = item.priority || 
                      (growthRate > 10 ? 'High' : 
                       growthRate > 0 ? 'Medium' : 'Low');
                    
                    // Generate recommendation text if not already provided
                    const recommendation = item.recommendation || 
                      (growthRate > 10 ? `Increase safety stock by ${Math.round(growthRate)}%` : 
                       growthRate > 5 ? 'Maintain elevated safety stock levels' : 
                       growthRate > 0 ? 'Maintain current safety stock levels' : 
                       'Consider reducing safety stock');
                    
                    return (
                      <React.Fragment key={index}>
                        <TableRow 
                          sx={{ 
                            '& > *': { borderBottom: 'unset' },
                            '&:nth-of-type(even)': {
                              backgroundColor: expandedCategory !== category ? theme.palette.action.hover : 'transparent',
                            }
                          }}
                        >
                          <TableCell component="th" scope="row" sx={{ fontWeight: 'medium' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <CategoryIcon fontSize="small" sx={{ mr: 1, color: theme.palette.primary.light }} />
                              {category}
                            </Box>
                          </TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validSafetyStock))}</TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validReorderPoint))}</TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validForecast))}</TableCell>
                          <TableCell 
                            align="right"
                            sx={{ fontWeight: 'medium' }}
                          >
                            <Box sx={{ 
                              display: 'flex', 
                              alignItems: 'center',
                              justifyContent: 'flex-end',
                              color: growthRate > 0 ? theme.palette.success.main : 
                                     growthRate < 0 ? theme.palette.error.main : 'inherit'
                            }}>
                              {growthRate > 0 ? (
                                <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} />
                              ) : growthRate < 0 ? (
                                <TrendingDownIcon fontSize="small" sx={{ mr: 0.5 }} />
                              ) : null}
                              {growthRate > 0 ? '+' : ''}{growthRate.toFixed(2)}%
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={priority}
                              color={getPriorityColor(priority)}
                              size="small"
                              sx={{ fontWeight: 'medium' }}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                              {recommendation}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <IconButton
                              aria-label="expand row"
                              size="small"
                              onClick={() => toggleCategory(category)}
                              sx={{ 
                                bgcolor: expandedCategory === category ? theme.palette.action.selected : 'transparent',
                                '&:hover': { bgcolor: theme.palette.action.hover }
                              }}
                            >
                              {expandedCategory === category ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                            </IconButton>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={8}>
                            <Collapse in={expandedCategory === category} timeout="auto" unmountOnExit>
                              <Box sx={{ 
                                margin: 1, 
                                py: 2, 
                                px: 2, 
                                bgcolor: theme.palette.action.hover,
                                borderRadius: 2
                              }}>
                                <Typography variant="h6" gutterBottom component="div" sx={{ 
                                  fontWeight: 'bold',
                                  color: theme.palette.primary.main,
                                  borderBottom: `1px solid ${theme.palette.divider}`,
                                  pb: 1
                                }}>
                                  Details for {category}
                                </Typography>
                                <Grid container spacing={3}>
                                  <Grid item xs={12} md={6}>
                                    <Card variant="outlined" sx={{ 
                                      borderRadius: 2,
                                      boxShadow: theme.shadows[2]
                                    }}>
                                      <CardContent>
                                        <Typography color="primary" variant="subtitle1" gutterBottom fontWeight="bold" sx={{
                                          display: 'flex',
                                          alignItems: 'center'
                                        }}>
                                          <InventoryIcon fontSize="small" sx={{ mr: 1 }} />
                                          Inventory Planning
                                        </Typography>
                                        <Divider sx={{ mb: 2 }} />
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                          <Typography variant="body2" color="text.secondary">Safety Stock:</Typography>
                                          <Typography variant="body2" fontWeight="medium">
                                            {new Intl.NumberFormat().format(Math.round(validSafetyStock))} units
                                          </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                          <Typography variant="body2" color="text.secondary">Lead Time Demand:</Typography>
                                          <Typography variant="body2" fontWeight="medium">
                                            {new Intl.NumberFormat().format(Math.round(Math.max(validReorderPoint - validSafetyStock, 0)))} units
                                          </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                          <Typography variant="body2" color="text.secondary">Reorder Point:</Typography>
                                          <Typography variant="body2" fontWeight="medium">
                                            {new Intl.NumberFormat().format(Math.round(validReorderPoint))} units
                                          </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                          <Typography variant="body2" color="text.secondary">Next Month Forecast:</Typography>
                                          <Typography variant="body2" fontWeight="medium">
                                            {new Intl.NumberFormat().format(Math.round(validForecast))} units
                                          </Typography>
                                        </Box>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                  <Grid item xs={12} md={6}>
                                    <Card variant="outlined" sx={{ 
                                      borderRadius: 2,
                                      boxShadow: theme.shadows[2],
                                      height: '100%'
                                    }}>
                                      <CardContent>
                                        <Typography color="secondary" variant="subtitle1" gutterBottom fontWeight="bold" sx={{
                                          display: 'flex',
                                          alignItems: 'center'
                                        }}>
                                          <AssignmentIcon fontSize="small" sx={{ mr: 1 }} />
                                          Action Plan
                                        </Typography>
                                        <Divider sx={{ mb: 2 }} />
                                        <Alert 
                                          severity={
                                            growthRate > 10 ? "warning" : 
                                            growthRate > 0 ? "info" : 
                                            "success"
                                          }
                                          variant="outlined"
                                          sx={{ mb: 2 }}
                                        >
                                          <Typography variant="body2" fontWeight="medium">
                                            {recommendation}
                                          </Typography>
                                        </Alert>
                                        <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                                          {growthRate > 10 
                                            ? 'High growth category requires additional inventory to prevent stockouts.' 
                                            : growthRate > 0 
                                            ? 'Moderate growth category indicates stable demand.' 
                                            : 'Declining demand suggests reducing inventory investment.'}
                                        </Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                </Grid>
                              </Box>
                            </Collapse>
                          </TableCell>
                        </TableRow>
                      </React.Fragment>
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

export default RecommendationsPage;