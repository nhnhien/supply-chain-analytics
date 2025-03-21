import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography, Tooltip, IconButton, Divider } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import { Info as InfoIcon } from '@mui/icons-material';

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

const SellerPerformanceChart = ({ sellerData }) => {
  const [processedData, setProcessedData] = useState([]);

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

  // Define colors for clusters
  const clusterColors = {
    0: '#00C49F', // High performers - green
    1: '#FFBB28', // Average performers - amber
    2: '#FF8042'  // Low performers - orange/red
  };
  
  // Define cluster names for legend
  const clusterNames = {
    0: 'High Performers',
    1: 'Average Performers',
    2: 'Low Performers'
  };
  
  // Custom tooltip component for scatter chart
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box sx={{ bgcolor: 'background.paper', p: 1, border: '1px solid #ccc', maxWidth: 220 }}>
          <Typography variant="subtitle2">{data.name}</Typography>
          <Divider sx={{ my: 0.5 }} />
          <Typography variant="body2">
            Sales: ${data.original.sales.toLocaleString()}
          </Typography>
          <Typography variant="body2">
            Processing Time: {data.original.processingTime.toFixed(1)} days
          </Typography>
          <Typography variant="body2">
            Order Count: {data.original.orderCount}
          </Typography>
        </Box>
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
        p: 2 
      }}>
        <Typography color="text.secondary" align="center">
          No seller performance data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Info tooltip to explain data normalization */}
      <Tooltip title="Data has been normalized for better visualization. Hover over data points to see actual values.">
        <IconButton 
          size="small" 
          sx={{ position: 'absolute', top: 0, right: 0, zIndex: 2 }}
          aria-label="Visualization information"
        >
          <InfoIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="Sales"
            label={{ value: 'Sales', position: 'bottom', offset: -5 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Processing Time"
            label={{ value: 'Processing Time', angle: -90, position: 'insideLeft', offset: 10 }}
          />
          <ZAxis type="number" dataKey="z" range={dynamicZRange} />
          <RechartsTooltip content={<CustomTooltip />} />
          <Legend />
          {processedData.map(cluster => (
            <Scatter
              key={cluster.cluster}
              name={clusterNames[cluster.cluster] || `Cluster ${cluster.cluster}`}
              data={cluster.data}
              fill={clusterColors[cluster.cluster] || '#8884d8'}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default SellerPerformanceChart;