import React, { useState, useEffect, useMemo } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  Divider, List, ListItem, ListItemText, ListItemIcon,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Avatar, TablePagination, FormControl, InputLabel, Select, MenuItem,
  Tooltip, IconButton
} from '@mui/material';
import { 
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip as RechartsTooltip, 
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
  Person as PersonIcon,
  Info as InfoIcon
} from '@mui/icons-material';

// Enhanced Winsorization function with proper percentile calculation
const applyWinsorization = (data, field, lowerPercentile = 0, upperPercentile = 99) => {
  if (!data || data.length === 0 || !field) return data;
  
  // Extract field values and filter out invalid ones
  const values = data
    .map(item => parseFloat(item?.[field] || 0))
    .filter(val => !isNaN(val) && isFinite(val));
  
  if (values.length === 0) return data;
  
  // Calculate percentiles
  const lowerCap = calculatePercentile(values, lowerPercentile);
  const upperCap = calculatePercentile(values, upperPercentile);
  
  console.log(`Winsorizing ${field}: lower cap at ${lowerCap}, upper cap at ${upperCap}`);
  
  // Apply caps to data
  return data.map(item => {
    if (!item) return item;
    
    let value = parseFloat(item[field]);
    if (!isNaN(value) && isFinite(value)) {
      if (value < lowerCap) value = lowerCap;
      if (value > upperCap) value = upperCap;
      return {...item, [field]: value};
    }
    return item;
  });
};

// Helper function to calculate percentile
const calculatePercentile = (array, percentile) => {
  if (!array || array.length === 0) return 0;
  const sorted = [...array].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * percentile / 100;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
};

// Enhanced function to normalize data for visualization
const normalizeDataForVisualization = (data, field, targetRange = [0, 100]) => {
  if (!data || data.length === 0 || !field) return data;
  
  // Extract field values and filter out invalid ones
  const values = data
    .map(item => parseFloat(item?.[field] || 0))
    .filter(val => !isNaN(val) && isFinite(val));
  
  if (values.length <= 1) return data; // Not enough data points to normalize
  
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  // If min and max are the same, return original data
  if (Math.abs(max - min) < 1e-6) return data;
  
  const [targetMin, targetMax] = targetRange;
  const scale = (targetMax - targetMin) / (max - min);
  
  const normalizedField = `normalized_${field}`;
  return data.map(item => {
    if (!item) return item;
    
    let value = parseFloat(item[field]);
    if (!isNaN(value) && isFinite(value)) {
      const normalizedValue = targetMin + (value - min) * scale;
      return {...item, [normalizedField]: normalizedValue};
    }
    
    return {...item, [normalizedField]: targetMin};
  });
};

const SellerPerformancePage = ({ data }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [clusterFilter, setClusterFilter] = useState('all');
  const [sellerData, setSellerData] = useState([]);
  const [processedData, setProcessedData] = useState({
    winsorized: [],
    normalized: [],
    clusterMetrics: [],
    clusterDistribution: [],
    scatterData: []
  });
  
  // Process and transform data when it changes
  useEffect(() => {
    if (data && data.clusters) {
      const rawData = data.clusters;
      
      // Apply data cleansing and transformation pipeline
      const processData = () => {
        // Step 1: Clean the data - handle missing values
        const cleanedData = rawData.map(seller => {
          if (!seller) return null;
          
          return {
            ...seller,
            // Ensure all necessary fields have valid values
            seller_id: seller.seller_id || `unknown-${Math.random().toString(36).substr(2, 9)}`,
            total_sales: parseFloat(seller.total_sales) || 0,
            avg_processing_time: parseFloat(seller.avg_processing_time) || 0,
            avg_delivery_days: parseFloat(seller.avg_delivery_days) || 0,
            order_count: parseFloat(seller.order_count) || 0,
            on_time_delivery_rate: parseFloat(seller.on_time_delivery_rate) || 0,
            // Default prediction to 1 (average) if invalid
            prediction: [0, 1, 2].includes(seller.prediction) ? seller.prediction : 1
          };
        }).filter(seller => seller !== null);
        
        // Step 2: Apply Winsorization to cap extreme values
        const winsorizedData = applyWinsorization(cleanedData, 'total_sales', 0, 99);
        
        // Step 3: Normalize data for visualization
        const normalizedData = normalizeDataForVisualization(
          winsorizedData, 
          'total_sales', 
          [10, 100]
        );
        
        // Step 4: Calculate cluster metrics
        const clusters = [0, 1, 2];
        const clusterNames = {
          0: 'High Performers',
          1: 'Average Performers',
          2: 'Low Performers'
        };
        
        const clusterMetrics = clusters.map(cluster => {
          const clusteredSellers = winsorizedData.filter(seller => seller.prediction === cluster);
          
          if (clusteredSellers.length === 0) return null;
          
          const avgProcessingTime = clusteredSellers.reduce((sum, seller) => 
            sum + seller.avg_processing_time, 0) / clusteredSellers.length;
            
          const avgDeliveryDays = clusteredSellers.reduce((sum, seller) => 
            sum + seller.avg_delivery_days, 0) / clusteredSellers.length;
            
          const avgOrderCount = clusteredSellers.reduce((sum, seller) => 
            sum + seller.order_count, 0) / clusteredSellers.length;
            
          const totalSales = clusteredSellers.reduce((sum, seller) => 
            sum + seller.total_sales, 0);
            
          const avgOnTimeRate = clusteredSellers.reduce((sum, seller) => 
            sum + seller.on_time_delivery_rate, 0) / clusteredSellers.length;
          
          return {
            cluster,
            clusterName: clusterNames[cluster],
            count: clusteredSellers.length,
            avgProcessingTime,
            avgDeliveryDays,
            avgOrderCount,
            totalSales,
            avgOnTimeRate
          };
        }).filter(metric => metric !== null);
        
        // Step 5: Calculate cluster distribution
        const totalSellers = winsorizedData.length;
        const clusterDistribution = clusterMetrics.map(metric => ({
          name: metric.clusterName,
          value: metric.count,
          percentage: totalSellers > 0 ? (metric.count / totalSellers) * 100 : 0
        }));
        
        // Step 6: Prepare scatter plot data
        const scatterData = clusters
          .filter(cluster => winsorizedData.some(seller => seller.prediction === cluster))
          .map(cluster => ({
            cluster,
            name: clusterNames[cluster],
            data: normalizedData
              .filter(seller => seller.prediction === cluster)
              .map(seller => ({
                x: seller.total_sales,
                y: seller.avg_processing_time,
                z: Math.max(20, seller.order_count), // Minimum size for visibility
                name: seller.seller_id,
                raw: {
                  sales: seller.total_sales,
                  processing: seller.avg_processing_time,
                  orders: seller.order_count,
                  delivery: seller.avg_delivery_days,
                  onTime: seller.on_time_delivery_rate
                }
              }))
          }));
        
        return {
          winsorized: winsorizedData,
          normalized: normalizedData,
          clusterMetrics,
          clusterDistribution,
          scatterData
        };
      };
      
      // Execute data processing and update state
      const processedResults = processData();
      setSellerData(processedResults.winsorized);
      setProcessedData(processedResults);
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
  
  // Colors for visualization
  const COLORS = ['#00C49F', '#FFBB28', '#FF8042'];
  const clusterColors = {
    0: '#00C49F', // High performers
    1: '#FFBB28', // Average performers
    2: '#FF8042'  // Low performers
  };
  
  // Map performance rating from 1-5 stars based on cluster and relative performance
  const getPerformanceRating = (seller) => {
    if (!seller) return 0;
    
    const cluster = seller.prediction;
    // Base rating on cluster (0: 5 stars, 1: 3 stars, 2: 1 star)
    let baseRating = cluster === 0 ? 5 : cluster === 1 ? 3 : 1;
    
    // Adjust within cluster based on processing time and on-time delivery
    const clusterMetric = processedData.clusterMetrics.find(metric => metric.cluster === cluster);
    if (!clusterMetric) return baseRating;
    
    const clusterAvgTime = clusterMetric.avgProcessingTime;
    const sellerTime = seller.avg_processing_time || 0;
    
    const timeAdjustment = clusterAvgTime > 0 
      ? Math.round((clusterAvgTime - sellerTime) / clusterAvgTime * 0.5) // +/- 0.5 star based on processing time
      : 0;
    
    const clusterAvgOnTime = clusterMetric.avgOnTimeRate;
    const sellerOnTime = seller.on_time_delivery_rate || 0;
    
    const onTimeAdjustment = clusterAvgOnTime > 0 
      ? Math.round((sellerOnTime - clusterAvgOnTime) / clusterAvgOnTime * 0.5) // +/- 0.5 star based on on-time rate
      : 0;
    
    return Math.max(1, Math.min(5, baseRating + timeAdjustment + onTimeAdjustment));
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
        <Card sx={{ p: 1, border: '1px solid #ccc', maxWidth: 250 }}>
          <Typography variant="subtitle2">{data.name || 'Seller'}</Typography>
          <Divider sx={{ my: 1 }} />
          <Typography variant="body2">
            <strong>Sales:</strong> ${data.raw.sales.toLocaleString()}
          </Typography>
          <Typography variant="body2">
            <strong>Processing Time:</strong> {data.raw.processing.toFixed(1)} days
          </Typography>
          <Typography variant="body2">
            <strong>Orders:</strong> {data.raw.orders}
          </Typography>
          <Typography variant="body2">
            <strong>On-Time Delivery:</strong> {data.raw.onTime.toFixed(1)}%
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
                    data={processedData.clusterDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                  >
                    {processedData.clusterDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip formatter={(value) => `${value} sellers`} />
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
                  data={processedData.clusterMetrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="clusterName" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <RechartsTooltip />
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
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography component="h2" variant="h6" color="primary">
                Sales vs. Processing Time
              </Typography>
              <Tooltip title="Data points have been scaled to improve visualization. Hover over points to see actual values.">
                <IconButton size="small" sx={{ ml: 1 }}>
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            <Box sx={{ flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="Sales" 
                    label={{ value: 'Sales', position: 'bottom', offset: 0 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Processing Time" 
                    label={{ value: 'Processing Time (days)', angle: -90, position: 'insideLeft' }}
                  />
                  <ZAxis type="number" dataKey="z" range={[50, 400]} />
                  <RechartsTooltip content={<ScatterTooltip />} />
                  <Legend />
                  
                  {processedData.scatterData.map((cluster) => (
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
                    <TableCell align="right">On-Time Rate</TableCell>
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
                        <TableCell align="right">
                          {(seller.on_time_delivery_rate || 0).toFixed(1)}%
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
              {processedData.clusterMetrics.map((metric) => (
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
                            {metric.count} ({(metric.percentage || 0).toFixed(1)}%)
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
                            Avg. On-Time Rate:
                          </Typography>
                          <Typography variant="body1">
                            {metric.avgOnTimeRate.toFixed(1)}%
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