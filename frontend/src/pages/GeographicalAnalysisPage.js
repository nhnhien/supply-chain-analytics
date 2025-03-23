import React, { useState } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, TablePagination, useTheme, Divider, Alert, AlertTitle,
  Tooltip, IconButton
} from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';
import {
  Public as PublicIcon,
  Map as MapIcon,
  TrendingUp as TrendingUpIcon,
  LocalShipping as ShippingIcon,
  AttachMoney as MoneyIcon,
  PieChart as PieChartIcon,
  TableChart as TableChartIcon,
  Insights as InsightsIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const GeographicalAnalysisPage = ({ data }) => {
  const theme = useTheme();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  if (!data || !data.stateMetrics || data.stateMetrics.length === 0) {
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
          <PublicIcon sx={{ mr: 1 }} />
          Geographical Analysis
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
          <AlertTitle sx={{ fontWeight: 'bold' }}>No Geographical Data Available</AlertTitle>
          <Typography variant="body1">
            Run the supply chain analysis first to generate geographical insights.
          </Typography>
        </Alert>
      </Box>
    );
  }
  
  // Handle pagination
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  // Sort states by order count
  const sortedStates = [...data.stateMetrics].sort((a, b) => 
    (b.order_count || 0) - (a.order_count || 0)
  );
  
  // Get visible rows for pagination
  const visibleRows = sortedStates.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );
  
  // Get top 10 states by order count for visualization
  const topStates = sortedStates.slice(0, 10);
  const topPerformingRegion = topStates.length > 0 ? topStates[0] : null;
  
  // Calculate total orders
  const totalOrders = sortedStates.reduce((sum, state) => sum + (state.order_count || 0), 0);
  
  // Create data for pie chart
  const pieData = [];
  let otherOrders = 0;
  
  sortedStates.forEach((state, index) => {
    if (index < 5) {
      pieData.push({
        name: state.customer_state,
        value: state.order_count || 0,
        percentage: totalOrders > 0 ? ((state.order_count || 0) / totalOrders) * 100 : 0
      });
    } else {
      otherOrders += (state.order_count || 0);
    }
  });
  
  if (sortedStates.length > 5) {
    pieData.push({
      name: 'Others',
      value: otherOrders,
      percentage: totalOrders > 0 ? (otherOrders / totalOrders) * 100 : 0
    });
  }
  
  // Colors for charts
  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.grey[500]
  ];
  
  // Compute fastest delivery region safely
  const sortedByDelivery = [...sortedStates].sort((a, b) => (a.avg_delivery_days || 0) - (b.avg_delivery_days || 0));
  const fastestRegion = sortedByDelivery.length > 0 ? sortedByDelivery[0] : null;
  
  // Compute highest value region safely
  const sortedBySales = [...sortedStates].sort((a, b) => (b.total_sales || 0) - (a.total_sales || 0));
  const highestValueRegion = sortedBySales.length > 0 ? sortedBySales[0] : null;
  
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
        <PublicIcon sx={{ mr: 1 }} />
        Geographical Analysis
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
              Regional Performance
            </Typography>
          </Box>
        </Grid>
        
        {/* Top States Bar Chart */}
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
              <MapIcon sx={{ mr: 1 }} /> Top States by Order Count
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topStates}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                <XAxis 
                  type="number" 
                  tick={{ fill: theme.palette.text.secondary }}
                />
                <YAxis 
                  dataKey="customer_state" 
                  type="category" 
                  tick={{ fontSize: 12, fill: theme.palette.text.secondary }}
                  width={100}
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
                  formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
                />
                <Bar 
                  dataKey="order_count" 
                  name="Order Count" 
                  fill="#8884d8"
                  radius={[0, 4, 4, 0]}
                >
                  {topStates.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Order Distribution Pie Chart */}
        <Grid item xs={12} lg={4}>
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
              <PieChartIcon sx={{ mr: 1 }} /> Order Distribution
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={100}
                  innerRadius={40}
                  paddingAngle={2}
                  fill="#8884d8"
                  dataKey="value"
                  stroke={theme.palette.background.paper}
                  strokeWidth={2}
                  label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: '8px',
                    boxShadow: theme.shadows[3]
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Regional Insights Section */}
        <Grid item xs={12}>
          <Box sx={{ mt: 2, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.secondary.main}`,
              pl: 2
            }}>
              Regional Insights
            </Typography>
          </Box>
        </Grid>
        
        {/* Regional Insights Cards */}
        <Grid item xs={12}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card elevation={3} sx={{ 
                borderRadius: 2, 
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                height: '100%',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[6]
                }
              }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom color="primary" sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    pb: 1,
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    fontWeight: 'bold'
                  }}>
                    <TrendingUpIcon sx={{ mr: 1 }} />
                    Top Performing Region
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {topPerformingRegion ? topPerformingRegion.customer_state : 'N/A'}
                    </Typography>
                    <Chip 
                      label={topPerformingRegion ? `${new Intl.NumberFormat().format(topPerformingRegion.order_count || 0)} orders` : 'No data available'} 
                      color="primary"
                      sx={{ mt: 1, fontWeight: 'medium' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card elevation={3} sx={{ 
                borderRadius: 2, 
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                height: '100%',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[6]
                }
              }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom color="secondary" sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    pb: 1,
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    fontWeight: 'bold'
                  }}>
                    <ShippingIcon sx={{ mr: 1 }} />
                    Fastest Delivery Region
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {fastestRegion ? fastestRegion.customer_state : 'N/A'}
                    </Typography>
                    <Chip 
                      label={fastestRegion ? `${(fastestRegion.avg_delivery_days || 0).toFixed(1)} days average delivery` : 'No data available'} 
                      color="secondary"
                      sx={{ mt: 1, fontWeight: 'medium' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card elevation={3} sx={{ 
                borderRadius: 2, 
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                height: '100%',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[6]
                }
              }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom color="success.dark" sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    pb: 1,
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    fontWeight: 'bold'
                  }}>
                    <MoneyIcon sx={{ mr: 1 }} />
                    Highest Value Region
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {highestValueRegion ? highestValueRegion.customer_state : 'N/A'}
                    </Typography>
                    <Chip 
                      label={highestValueRegion ? `$${new Intl.NumberFormat().format(highestValueRegion.total_sales || 0)} total sales` : 'No data available'} 
                      color="success"
                      sx={{ mt: 1, fontWeight: 'medium' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
        
        {/* State Metrics Table Section */}
        <Grid item xs={12}>
          <Box sx={{ mt: 4, mb: 1 }}>
            <Typography variant="h5" component="h2" sx={{ 
              fontWeight: 'medium',
              borderLeft: `4px solid ${theme.palette.info.main}`,
              pl: 2
            }}>
              Detailed State Metrics
            </Typography>
          </Box>
        </Grid>
        
        {/* State Metrics Table */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ 
            width: '100%',
            borderRadius: 2,
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            overflow: 'hidden'
          }}>
            <Box sx={{ 
              p: 2, 
              pl: 3, 
              display: 'flex', 
              alignItems: 'center',
              borderBottom: `1px solid ${theme.palette.divider}`
            }}>
              <Typography component="h2" variant="h6" sx={{
                color: theme.palette.info.main,
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center'
              }}>
                <TableChartIcon sx={{ mr: 1 }} /> State Performance Metrics
              </Typography>
              <Tooltip title="This table shows detailed metrics for each state in the dataset">
                <IconButton size="small" sx={{ ml: 1 }}>
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ backgroundColor: theme.palette.action.hover }}>
                    <TableCell sx={{ fontWeight: 'bold' }}>State</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Order Count</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Total Sales</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Avg. Processing Time</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Avg. Delivery Days</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Top Category</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {visibleRows.map((state, index) => {
                    const topCategory = data.topCategoryByState && 
                      data.topCategoryByState.find(item => item.customer_state === state.customer_state);
                    
                    return (
                      <TableRow
                        key={state.customer_state}
                        hover
                        sx={{
                          '&:nth-of-type(even)': {
                            backgroundColor: theme.palette.action.hover,
                          },
                          '&:hover': {
                            backgroundColor: theme.palette.action.selected,
                          },
                        }}
                      >
                        <TableCell component="th" scope="row" sx={{ fontWeight: 'medium' }}>
                          {state.customer_state}
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            component="span"
                            fontWeight="medium"
                            color={index === 0 ? 'primary.main' : 'inherit'}
                          >
                            {new Intl.NumberFormat().format(state.order_count || 0)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                            <MoneyIcon 
                              fontSize="small" 
                              sx={{ 
                                mr: 0.5, 
                                color: state === highestValueRegion ? theme.palette.success.main : 'inherit',
                                opacity: state === highestValueRegion ? 1 : 0.5
                              }} 
                            />
                            ${new Intl.NumberFormat().format(state.total_sales || 0)}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          {(state.avg_processing_time || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                            <ShippingIcon 
                              fontSize="small" 
                              sx={{ 
                                mr: 0.5, 
                                color: state === fastestRegion ? theme.palette.secondary.main : 'inherit',
                                opacity: state === fastestRegion ? 1 : 0.5
                              }} 
                            />
                            {(state.avg_delivery_days || 0).toFixed(1)} days
                          </Box>
                        </TableCell>
                        <TableCell>
                          {topCategory ? (
                            <Chip
                              label={topCategory.product_category_name}
                              size="small"
                              color="primary"
                              variant="outlined"
                              sx={{ fontWeight: 'medium' }}
                            />
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                  {visibleRows.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} align="center" sx={{ py: 3 }}>
                        <Typography color="text.secondary">No state metrics available</Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              rowsPerPageOptions={[5, 10, 25, 50]}
              component="div"
              count={sortedStates.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              sx={{ 
                borderTop: `1px solid ${theme.palette.divider}`,
                '& .MuiTablePagination-toolbar': {
                  padding: '16px'
                }
              }}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GeographicalAnalysisPage;