import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Divider, List, ListItem, ListItemText, ListItemIcon,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Avatar, TablePagination, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { 
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar
} from 'recharts';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  StarHalf as StarHalfIcon,
  LocalShipping as ShippingIcon,
  AttachMoney as MoneyIcon,
  Person as PersonIcon
} from '@mui/icons-material';

/**
 * Seller Performance Page Component
 * 
 * @param {Object} props Component props
 * @param {Object} props.data Seller performance data
 */
const SellerPerformancePage = ({ data }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [clusterFilter, setClusterFilter] = useState('all');
  const [sellerData, setSellerData] = useState([]);
  
  useEffect(() => {
    if (data && data.clusters) {
      setSellerData(data.clusters);
    }
  }, [data]);
  
  if (!data || !data.clusters || data.clusters.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Seller Performance Analysis
        </Typography>
        <Typography>
          No seller performance data available. Run the supply chain analysis first.
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
  
  // Handle cluster filter change
  const handleClusterFilterChange = (event) => {
    setClusterFilter(event.target.value);
    setPage(0);
  };
  
  // Filter seller data based on selected cluster
  const filteredSellers = sellerData.filter(seller => {
    if (clusterFilter === 'all') return true;
    return seller.prediction === parseInt(clusterFilter);
  });
  
  // Get visible rows for pagination
  const visibleRows = filteredSellers.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );
  
  // Cluster metrics for performance benchmarks
  const clusterMetrics = [0, 1, 2].map(cluster => {
    const clusteredSellers = sellerData.filter(seller => seller.prediction === cluster);
    
    if (clusteredSellers.length === 0) {
      return {
        cluster,
        clusterName: cluster === 0 ? 'High Performers' : cluster === 1 ? 'Average Performers' : 'Low Performers',
        count: 0,
        avgProcessingTime: 0,
        avgDeliveryDays: 0,
        avgOrderCount: 0,
        totalSales: 0
      };
    }
    
    const avgProcessingTime = clusteredSellers.reduce((sum, seller) => sum + (seller.avg_processing_time || 0), 0) / clusteredSellers.length;
    const avgDeliveryDays = clusteredSellers.reduce((sum, seller) => sum + (seller.avg_delivery_days || 0), 0) / clusteredSellers.length;
    const avgOrderCount = clusteredSellers.reduce((sum, seller) => sum + (seller.order_count || 0), 0) / clusteredSellers.length;
    const totalSales = clusteredSellers.reduce((sum, seller) => sum + (seller.total_sales || 0), 0);
    
    return {
      cluster,
      clusterName: cluster === 0 ? 'High Performers' : cluster === 1 ? 'Average Performers' : 'Low Performers',
      count: clusteredSellers.length,
      avgProcessingTime,
      avgDeliveryDays,
      avgOrderCount,
      totalSales
    };
  });
  
  // Calculate percentage of sellers in each cluster
  const totalSellers = sellerData.length;
  const clusterDistribution = clusterMetrics.map(metric => ({
    name: metric.clusterName,
    value: metric.count,
    percentage: (metric.count / totalSellers) * 100
  }));
  
  // Metrics for scatter chart
  const scatterData = [0, 1, 2].map(cluster => ({
    cluster,
    name: cluster === 0 ? 'High Performers' : cluster === 1 ? 'Average Performers' : 'Low Performers',
    data: sellerData
      .filter(seller => seller.prediction === cluster)
      .map(seller => ({
        x: seller.total_sales || 0,
        y: seller.avg_processing_time || 0,
        z: seller.order_count || 20, // Default value for bubble size
        name: seller.seller_id
      }))
  }));
  
  // Colors for visualization
  const COLORS = ['#00C49F', '#FFBB28', '#FF8042'];
  const clusterColors = {
    0: '#00C49F', // High performers
    1: '#FFBB28', // Average performers
    2: '#FF8042'  // Low performers
  };
  
  // Map performance rating from 1-5 stars
  const getPerformanceRating = (seller) => {
    const cluster = seller.prediction;
    
    // Base rating on cluster (0: 5 stars, 1: 3 stars, 2: 1 star)
    let baseRating = cluster === 0 ? 5 : cluster === 1 ? 3 : 1;
    
    // Adjust within cluster based on processing time relative to cluster average
    const clusterAvgTime = clusterMetrics[cluster].avgProcessingTime;
    const sellerTime = seller.avg_processing_time || 0;
    
    const adjustment = clusterAvgTime > 0 
      ? Math.round((clusterAvgTime - sellerTime) / clusterAvgTime * 1) // +/- 1 star based on processing time
      : 0;
    
    return Math.max(1, Math.min(5, baseRating + adjustment)); // Ensure rating is between 1-5
  };
  
  // Render star rating
  const renderStarRating = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<StarIcon key={`full-${i}`} color="primary" />);
    }
    
    if (hasHalfStar) {
      stars.push(<StarHalfIcon key="half" color="primary" />);
    }
    
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
    for (let i = 0; i < emptyStars; i++) {
      stars.push(<StarBorderIcon key={`empty-${i}`} color="primary" />);
    }
    
    return stars;
  };
  
  // Custom tooltip for scatter chart
  const ScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Card sx={{ p: 1, border: '1px solid #ccc', maxWidth: 200 }}>
          <Typography variant="subtitle2">{data.name || 'Seller'}</Typography>
          <Divider sx={{ my: 1 }} />
          <Typography variant="body2">
            <strong>Sales:</strong> ${data.x.toLocaleString()}
          </Typography>
          <Typography variant="body2">
            <strong>Processing Time:</strong> {data.y.toFixed(1)} days
          </Typography>
          <Typography variant="body2">
            <strong>Orders:</strong> {data.z}
          </Typography>
        </Card>
      );
    }
    return null;
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Seller Performance Analysis
      </Typography>
      
      <Grid container spacing={3}>
        {/* Cluster Distribution */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 320 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Seller Performance Clusters
            </Typography>
            
            <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={clusterDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                  >
                    {clusterDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value) => `${value} sellers`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        
        {/* Cluster Metrics */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 320 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Performance Metrics by Cluster
            </Typography>
            
            <Box sx={{ flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={clusterMetrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="clusterName" 
                    tick={{ fontSize: 12 }} 
                  />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Bar 
                    yAxisId="left" 
                    dataKey="avgProcessingTime" 
                    name="Avg. Processing Time (days)" 
                    fill="#8884d8" 
                  />
                  <Bar 
                    yAxisId="right" 
                    dataKey="avgOrderCount" 
                    name="Avg. Order Count" 
                    fill="#82ca9d" 
                  />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        
        {/* Cluster Scatterplot */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 320 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Sales vs. Processing Time
            </Typography>
            
            <Box sx={{ flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="Sales" 
                    unit="$" 
                    tickFormatter={(value) => `$${(value/1000).toFixed(0)}k`}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Processing Time" 
                    unit=" days" 
                  />
                  <ZAxis type="number" dataKey="z" range={[50, 400]} />
                  <Tooltip content={<ScatterTooltip />} />
                  <Legend />
                  
                  {scatterData.map((cluster) => (
                    <Scatter
                      key={cluster.cluster}
                      name={cluster.name}
                      data={cluster.data}
                      fill={clusterColors[cluster.cluster]}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        
        {/* Table Filter Controls */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">
                  {filteredSellers.length} sellers
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth variant="outlined" size="small">
                  <InputLabel id="cluster-filter-label">Cluster Filter</InputLabel>
                  <Select
                    labelId="cluster-filter-label"
                    id="cluster-filter"
                    value={clusterFilter}
                    onChange={handleClusterFilterChange}
                    label="Cluster Filter"
                  >
                    <MenuItem value="all">All Clusters</MenuItem>
                    <MenuItem value="0">High Performers</MenuItem>
                    <MenuItem value="1">Average Performers</MenuItem>
                    <MenuItem value="2">Low Performers</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        {/* Seller Table */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ width: '100%' }}>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Seller ID</TableCell>
                    <TableCell>Cluster</TableCell>
                    <TableCell align="right">Total Sales</TableCell>
                    <TableCell align="right">Order Count</TableCell>
                    <TableCell align="right">Avg. Processing Time</TableCell>
                    <TableCell align="right">Avg. Delivery Days</TableCell>
                    <TableCell>Performance Rating</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {visibleRows.map((seller) => {
                    const rating = getPerformanceRating(seller);
                    const clusterName = seller.prediction === 0 
                      ? 'High Performer' 
                      : seller.prediction === 1 
                      ? 'Average Performer' 
                      : 'Low Performer';
                      
                    return (
                      <TableRow
                        key={seller.seller_id}
                        hover
                        sx={{
                          '&:hover': {
                            backgroundColor: 'rgba(0, 0, 0, 0.04)',
                          },
                        }}
                      >
                        <TableCell component="th" scope="row">
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Avatar 
                              sx={{ 
                                width: 30, 
                                height: 30, 
                                mr: 1, 
                                bgcolor: clusterColors[seller.prediction] 
                              }}
                            >
                              <PersonIcon />
                            </Avatar>
                            {seller.seller_id}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={clusterName}
                            size="small"
                            sx={{ bgcolor: clusterColors[seller.prediction], color: 'white' }}
                          />
                        </TableCell>
                        <TableCell align="right">
                          ${new Intl.NumberFormat().format(seller.total_sales || 0)}
                        </TableCell>
                        <TableCell align="right">
                          {new Intl.NumberFormat().format(seller.order_count || 0)}
                        </TableCell>
                        <TableCell align="right">
                          {(seller.avg_processing_time || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell align="right">
                          {(seller.avg_delivery_days || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex' }}>
                            {renderStarRating(rating)}
                          </Box>
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
              count={filteredSellers.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </Paper>
        </Grid>
        
        {/* Performance Insights */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Performance Insights
            </Typography>
            
            <Grid container spacing={3}>
              {clusterMetrics.map((metric) => (
                <Grid item xs={12} md={4} key={metric.cluster}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom sx={{ color: clusterColors[metric.cluster] }}>
                        {metric.clusterName}
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Seller Count:
                          </Typography>
                          <Typography variant="body1">
                            {metric.count} ({((metric.count / totalSellers) * 100).toFixed(1)}%)
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Avg. Processing Time:
                          </Typography>
                          <Typography variant="body1">
                            {metric.avgProcessingTime.toFixed(1)} days
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Avg. Delivery Days:
                          </Typography>
                          <Typography variant="body1">
                            {metric.avgDeliveryDays.toFixed(1)} days
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Avg. Order Count:
                          </Typography>
                          <Typography variant="body1">
                            {metric.avgOrderCount.toFixed(0)} orders
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Total Sales:
                          </Typography>
                          <Typography variant="body1">
                            ${new Intl.NumberFormat().format(Math.round(metric.totalSales))}
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SellerPerformancePage;