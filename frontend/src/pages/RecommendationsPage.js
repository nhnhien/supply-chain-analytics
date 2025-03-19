import React, { useState } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Divider, List, ListItem, ListItemText, ListItemIcon,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, Collapse, IconButton, Alert, AlertTitle
} from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, 
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
  AttachMoney as MoneyIcon
} from '@mui/icons-material';

/**
 * Recommendations Page Component
 * Displays inventory optimization recommendations and reorder strategies
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Recommendations data
 */
const RecommendationsPage = ({ data }) => {
  const [expandedCategory, setExpandedCategory] = useState(null);
  
  if (!data || !data.inventory || data.inventory.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Inventory Recommendations
        </Typography>
        <Alert severity="info">
          <AlertTitle>No recommendations available</AlertTitle>
          Run the supply chain analysis to generate inventory recommendations.
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
    <Box>
      <Typography variant="h4" gutterBottom>
        Inventory Recommendations
      </Typography>
      
      <Grid container spacing={3}>
        {/* Visual Recommendations */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={2} sx={{ p: 2, height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Inventory Planning Visualization
            </Typography>
            
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={barChartData.slice(0, 7)} // Limit to top 7 for readability
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end"
                  height={70}
                />
                <YAxis 
                  label={{ value: 'Units', angle: -90, position: 'insideLeft' }}
                  tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
                />
                <Tooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                />
                <Legend />
                <Bar 
                  dataKey="safetyStock" 
                  stackId="inventory" 
                  name="Safety Stock" 
                  fill="#8884d8" 
                />
                <Bar 
                  dataKey="leadTimeDemand" 
                  stackId="inventory" 
                  name="Lead Time Demand" 
                  fill="#82ca9d" 
                />
                <Bar 
                  dataKey="nextMonthForecast" 
                  name="Next Month Forecast" 
                  fill="#ffc658" 
                />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Summary Card */}
        <Grid item xs={12} lg={4}>
          <Paper elevation={2} sx={{ p: 2, height: 400, overflowY: 'auto' }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Summary of Recommendations
            </Typography>
            
            <List>
              {priorityCountsArray.map((item) => (
                <ListItem key={item.priority}>
                  <ListItemIcon>
                    {getPriorityIcon(item.priority)}
                  </ListItemIcon>
                  <ListItemText 
                    primary={`${item.priority} Priority Items`}
                    secondary={`${item.count} product categories`}
                  />
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body1" gutterBottom>
              Key actions to take:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <InventoryIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Update safety stock levels based on growth patterns"
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <ShippingIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Review supplier agreements for high-growth categories"
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <MoneyIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Optimize inventory capital allocation"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
        
        {/* Detailed Recommendations Table */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Detailed Inventory Recommendations
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Category</TableCell>
                    <TableCell align="right">Safety Stock</TableCell>
                    <TableCell align="right">Reorder Point</TableCell>
                    <TableCell align="right">Forecast</TableCell>
                    <TableCell align="right">Growth Rate</TableCell>
                    <TableCell>Priority</TableCell>
                    <TableCell>Recommendation</TableCell>
                    <TableCell>Details</TableCell>
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
                        <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
                          <TableCell component="th" scope="row">{category}</TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validSafetyStock))}</TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validReorderPoint))}</TableCell>
                          <TableCell align="right">{new Intl.NumberFormat().format(Math.round(validForecast))}</TableCell>
                          <TableCell 
                            align="right"
                            sx={{ color: growthRate > 0 ? 'success.main' : growthRate < 0 ? 'error.main' : 'inherit' }}
                          >
                            {growthRate > 0 ? '+' : ''}{growthRate.toFixed(2)}%
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={priority}
                              color={getPriorityColor(priority)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{recommendation}</TableCell>
                          <TableCell>
                            <IconButton
                              aria-label="expand row"
                              size="small"
                              onClick={() => toggleCategory(category)}
                            >
                              {expandedCategory === category ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                            </IconButton>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={8}>
                            <Collapse in={expandedCategory === category} timeout="auto" unmountOnExit>
                              <Box sx={{ margin: 1, py: 2 }}>
                                <Typography variant="h6" gutterBottom component="div">
                                  Details for {category}
                                </Typography>
                                <Grid container spacing={2}>
                                  <Grid item xs={12} md={6}>
                                    <Card variant="outlined">
                                      <CardContent>
                                        <Typography color="text.secondary" gutterBottom>
                                          Inventory Planning
                                        </Typography>
                                        <Typography variant="body2">
                                          Safety Stock: {new Intl.NumberFormat().format(Math.round(validSafetyStock))} units
                                        </Typography>
                                        <Typography variant="body2">
                                          Lead Time Demand: {new Intl.NumberFormat().format(Math.round(Math.max(validReorderPoint - validSafetyStock, 0)))} units
                                        </Typography>
                                        <Typography variant="body2">
                                          Reorder Point: {new Intl.NumberFormat().format(Math.round(validReorderPoint))} units
                                        </Typography>
                                        <Typography variant="body2">
                                          Next Month Forecast: {new Intl.NumberFormat().format(Math.round(validForecast))} units
                                        </Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                  <Grid item xs={12} md={6}>
                                    <Card variant="outlined">
                                      <CardContent>
                                        <Typography color="text.secondary" gutterBottom>
                                          Action Plan
                                        </Typography>
                                        <Typography variant="body2">
                                          {recommendation}
                                        </Typography>
                                        <Typography variant="body2" sx={{ mt: 1 }}>
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
