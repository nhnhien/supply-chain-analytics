import React, { useState } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, TablePagination
} from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';

const GeographicalAnalysisPage = ({ data }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  if (!data || !data.stateMetrics || data.stateMetrics.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Geographical Analysis
        </Typography>
        <Typography>
          No geographical data available. Run the supply chain analysis first.
        </Typography>
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
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#AAAAAA'];
  
  // Compute fastest delivery region safely
  const sortedByDelivery = [...sortedStates].sort((a, b) => (a.avg_delivery_days || 0) - (b.avg_delivery_days || 0));
  const fastestRegion = sortedByDelivery.length > 0 ? sortedByDelivery[0] : null;
  
  // Compute highest value region safely
  const sortedBySales = [...sortedStates].sort((a, b) => (b.total_sales || 0) - (a.total_sales || 0));
  const highestValueRegion = sortedBySales.length > 0 ? sortedBySales[0] : null;
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Geographical Analysis
      </Typography>
      
      <Grid container spacing={3}>
        {/* Top States Bar Chart */}
        <Grid item xs={12} md={8}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Top States by Order Count
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topStates}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis 
                  dataKey="customer_state" 
                  type="category" 
                  tick={{ fontSize: 12 }}
                  width={80}
                />
                <Tooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                />
                <Legend />
                <Bar 
                  dataKey="order_count" 
                  name="Order Count" 
                  fill="#8884d8"
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
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Order Distribution
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* State Metrics Table */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ width: '100%' }}>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>State</TableCell>
                    <TableCell align="right">Order Count</TableCell>
                    <TableCell align="right">Total Sales</TableCell>
                    <TableCell align="right">Avg. Processing Time</TableCell>
                    <TableCell align="right">Avg. Delivery Days</TableCell>
                    <TableCell>Top Category</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {visibleRows.map((state) => {
                    const topCategory = data.topCategoryByState && 
                      data.topCategoryByState.find(item => item.customer_state === state.customer_state);
                    
                    return (
                      <TableRow
                        key={state.customer_state}
                        hover
                      >
                        <TableCell component="th" scope="row">
                          {state.customer_state}
                        </TableCell>
                        <TableCell align="right">
                          {new Intl.NumberFormat().format(state.order_count || 0)}
                        </TableCell>
                        <TableCell align="right">
                          ${new Intl.NumberFormat().format(state.total_sales || 0)}
                        </TableCell>
                        <TableCell align="right">
                          {(state.avg_processing_time || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell align="right">
                          {(state.avg_delivery_days || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell>
                          {topCategory ? (
                            <Chip
                              label={topCategory.product_category_name}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
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
            />
          </Paper>
        </Grid>
        
        {/* Regional Insights */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Regional Insights
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Top Performing Region
                    </Typography>
                    <Typography variant="body1">
                      {topPerformingRegion ? topPerformingRegion.customer_state : 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      {topPerformingRegion ? `${new Intl.NumberFormat().format(topPerformingRegion.order_count || 0)} orders` : 'No data available'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Fastest Delivery Region
                    </Typography>
                    {fastestRegion ? (
                      <>
                        <Typography variant="body1">
                          {fastestRegion.customer_state}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          {(fastestRegion.avg_delivery_days || 0).toFixed(1)} days average delivery
                        </Typography>
                      </>
                    ) : (
                      <Typography variant="body1">No data available</Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Highest Value Region
                    </Typography>
                    {highestValueRegion ? (
                      <>
                        <Typography variant="body1">
                          {highestValueRegion.customer_state}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          ${new Intl.NumberFormat().format(highestValueRegion.total_sales || 0)} total sales
                        </Typography>
                      </>
                    ) : (
                      <Typography variant="body1">No data available</Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GeographicalAnalysisPage;
