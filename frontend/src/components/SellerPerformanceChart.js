import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography, Tooltip, IconButton, Divider, Paper, useTheme } from '@mui/material';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  Label 
} from 'recharts';
import { 
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  LocalShipping as ShippingIcon,
  Receipt as ReceiptIcon
} from '@mui/icons-material';

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

const SellerPerformanceChart = ({ sellerData, chartColors }) => {
  const theme = useTheme();
  const [processedData, setProcessedData] = useState([]);

  // Define colors for clusters - use theme colors if available
  const clusterColors = chartColors || {
    0: theme.palette.success.main, // High performers - green
    1: theme.palette.warning.main, // Average performers - amber
    2: theme.palette.error.main    // Low performers - orange/red
  };
  
  // Define cluster names for legend
  const clusterNames = {
    0: 'High Performers',
    1: 'Average Performers',
    2: 'Low Performers'
  };

  // Process and transform data when it changes
  useEffect(() => {
    let isMounted = true;
    
    const processSellerData = () => {
      if (!sellerData || sellerData.length === 0) return [];
      
      // Step 1: Clean the data - filter out invalid entries
      const cleanedData = sellerData
        .filter(seller => seller && typeof seller === 'object')
        .map(seller => ({
          ...seller,
          // Ensure required fields exist with default values if missing
          seller_id: seller.seller_id || `unknown-${Math.random().toString(36).substr(2, 9)}`,
          total_sales: parseFloat(seller.total_sales) || 0,
          avg_processing_time: parseFloat(seller.avg_processing_time) || 0,
          order_count: parseFloat(seller.order_count) || 20,
          // Default prediction to 1 (average) if invalid
          prediction: [0, 1, 2].includes(seller.prediction) ? seller.prediction : 1
        }));

      // Step 2: Apply Winsorization to cap extreme values
      // Get sales values and filter out invalid ones
      const salesValues = cleanedData
        .map(seller => seller.total_sales)
        .filter(val => !isNaN(val) && val > 0);
      
      // Calculate percentiles for capping
      const salesCap = calculatePercentile(salesValues, 99);
      
      // Apply caps to data
      const winsorizedData = cleanedData.map(seller => {
        const cappedSales = seller.total_sales > salesCap ? salesCap : seller.total_sales;
        return { ...seller, total_sales: cappedSales };
      });

      // Step 3: Normalize data for better visualization
      const allSales = winsorizedData.map(s => s.total_sales);
      const minSales = Math.min(...allSales);
      const maxSales = Math.max(...allSales);
      
      const allProcessingTimes = winsorizedData.map(s => s.avg_processing_time);
      const minProcessingTime = Math.min(...allProcessingTimes);
      const maxProcessingTime = Math.max(...allProcessingTimes);
      
      // Group data by clusters
      const clusters = {};
      
      winsorizedData.forEach(seller => {
        const cluster = seller.prediction;
        if (!clusters[cluster]) {
          clusters[cluster] = [];
        }
        
        // Calculate normalized values for better visualization
        // This keeps the relative positions but scales to a better visual range
        const normalizedSales = maxSales > minSales 
          ? ((seller.total_sales - minSales) / (maxSales - minSales)) * 100 
          : 50;
          
        const normalizedProcessingTime = maxProcessingTime > minProcessingTime 
          ? ((seller.avg_processing_time - minProcessingTime) / (maxProcessingTime - minProcessingTime)) * 100 
          : 50;
        
        clusters[cluster].push({
          x: normalizedSales,
          y: normalizedProcessingTime,
          z: Math.max(20, Math.min(100, seller.order_count)),
          name: seller.seller_id,
          original: {
            sales: seller.total_sales,
            processingTime: seller.avg_processing_time,
            orderCount: seller.order_count
          }
        });
      });
      
      // Convert to array format for visualization
      return Object.entries(clusters).map(([cluster, data]) => ({
        cluster: Number(cluster),
        data
      }));
    };
    
    try {
      const result = processSellerData();
      if (isMounted) {
        setProcessedData(result);
      }
    } catch (error) {
      console.error("Error processing seller data:", error);
      if (isMounted) {
        setProcessedData([]);
      }
    }
    
    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [sellerData]);

  // Calculate dynamic Z-axis range based on available data
  const dynamicZRange = useMemo(() => {
    const allZ = processedData.flatMap(cluster => cluster.data.map(d => d.z));
    if (allZ.length === 0) return [20, 100];
    
    const minZ = Math.min(...allZ);
    const maxZ = Math.max(...allZ);
    
    // If there's not much variation, use a fixed range
    if (maxZ - minZ < 10) return [50, 100];
    
    return [Math.max(20, minZ), Math.min(100, maxZ)];
  }, [processedData]);
  
  // Custom tooltip component for scatter chart
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper elevation={3} sx={{ 
          p: 1.5, 
          maxWidth: 220,
          borderRadius: 1,
          boxShadow: theme.shadows[3]
        }}>
          <Typography variant="subtitle2" fontWeight="bold" sx={{ 
            pb: 0.5, 
            borderBottom: `1px solid ${theme.palette.divider}`,
            color: theme.palette.primary.main
          }}>
            {data.name}
          </Typography>
          <Box sx={{ pt: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <TrendingUpIcon fontSize="small" sx={{ mr: 1, color: theme.palette.success.main }} />
              <Typography variant="body2">
                <strong>Sales:</strong> ${data.original.sales.toLocaleString()}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <ShippingIcon fontSize="small" sx={{ mr: 1, color: theme.palette.primary.main }} />
              <Typography variant="body2">
                <strong>Processing Time:</strong> {data.original.processingTime.toFixed(1)} days
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <ReceiptIcon fontSize="small" sx={{ mr: 1, color: theme.palette.info.main }} />
              <Typography variant="body2">
                <strong>Order Count:</strong> {data.original.orderCount}
              </Typography>
            </Box>
          </Box>
        </Paper>
      );
    }
    return null;
  };
  
  // If no processed data, show a friendly message
  if (processedData.length === 0) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        flexDirection: 'column',
        p: 2,
        color: theme.palette.text.secondary
      }}>
        <Box
          component="span"
          sx={{
            display: 'inline-block',
            p: 1,
            borderRadius: '50%',
            bgcolor: theme.palette.action.hover,
            width: 60,
            height: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2
          }}
        >
          <Box
            component="span"
            sx={{
              fontSize: '2rem',
              lineHeight: 1
            }}
          >
            📊
          </Box>
        </Box>
        <Typography variant="h6" color="text.secondary" align="center">
          No seller performance data available
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
          Run the seller analysis to generate performance clusters
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Info tooltip to explain data normalization */}
      <Tooltip 
        title="Data has been normalized for better visualization. Hover over data points to see actual values."
        arrow
        placement="top"
      >
        <IconButton 
          size="small" 
          sx={{ 
            position: 'absolute', 
            top: 0, 
            right: 0, 
            zIndex: 2,
            color: theme.palette.info.main
          }}
          aria-label="Visualization information"
        >
          <InfoIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis
            type="number"
            dataKey="x"
            name="Sales"
            tick={{ fill: theme.palette.text.secondary }}
            domain={[0, 100]}
          >
            <Label 
              value="Sales" 
              position="bottom" 
              offset={-5}
              style={{ 
                textAnchor: 'middle',
                fill: theme.palette.text.secondary,
                fontSize: 12
              }}
            />
          </XAxis>
          <YAxis
            type="number"
            dataKey="y"
            name="Processing Time"
            tick={{ fill: theme.palette.text.secondary }}
            domain={[0, 100]}
          >
            <Label 
              value="Processing Time" 
              angle={-90} 
              position="insideLeft" 
              offset={10}
              style={{ 
                textAnchor: 'middle',
                fill: theme.palette.text.secondary,
                fontSize: 12
              }}
            />
          </YAxis>
          <ZAxis type="number" dataKey="z" range={dynamicZRange} />
          <RechartsTooltip 
            content={<CustomTooltip />}
            cursor={{ strokeDasharray: '3 3', stroke: theme.palette.divider }}
          />
          <Legend 
            formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
            iconSize={10}
            wrapperStyle={{ paddingTop: 10 }}
          />
          {processedData.map(cluster => (
            <Scatter
              key={cluster.cluster}
              name={clusterNames[cluster.cluster] || `Cluster ${cluster.cluster}`}
              data={cluster.data}
              fill={clusterColors[cluster.cluster] || theme.palette.primary.main}
              shape="circle"
              strokeWidth={2}
              stroke={theme.palette.background.paper}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default SellerPerformanceChart;