import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Seller Performance Chart Component with enhanced data validation.
 * 
 * @param {Object} props Component props
 * @param {Array} props.sellerData Array of seller performance data
 */
const SellerPerformanceChart = ({ sellerData }) => {
  // Process seller data for visualization with robust error handling
  const processedData = useMemo(() => {
    if (!sellerData || sellerData.length === 0) return [];
    
    // Group sellers by cluster
    const clusters = {};
    
    sellerData.forEach(seller => {
      if (!seller) return; // Skip null/undefined sellers
      
      // Default to cluster 1 (medium) if prediction is missing
      const cluster = seller.prediction !== undefined && seller.prediction !== null ? seller.prediction : 1;
      
      if (!clusters[cluster]) {
        clusters[cluster] = [];
      }
      
      // Validate and parse total_sales
      let totalSales = parseFloat(seller.total_sales);
      if (isNaN(totalSales)) {
        console.warn(`Missing or invalid total_sales for seller ${seller.seller_id || 'unknown'}. Defaulting to 0.`);
        totalSales = 0;
      }
      
      // Validate and parse avg_processing_time
      let processingTime = parseFloat(seller.avg_processing_time);
      if (isNaN(processingTime)) {
        console.warn(`Missing or invalid avg_processing_time for seller ${seller.seller_id || 'unknown'}. Defaulting to 0.`);
        processingTime = 0;
      }
      
      // Validate order_count; default to 20 if missing or invalid
      let orderCount = parseFloat(seller.order_count);
      if (isNaN(orderCount) || orderCount <= 0) {
        console.warn(`Missing or invalid order_count for seller ${seller.seller_id || 'unknown'}. Defaulting to 20.`);
        orderCount = 20;
      }
      
      clusters[cluster].push({
        x: totalSales,
        y: processingTime,
        z: orderCount,
        name: seller.seller_id || `Seller ${clusters[cluster].length + 1}`
      });
    });
    
    return Object.entries(clusters).map(([cluster, data]) => ({
      cluster: Number(cluster),
      data
    }));
  }, [sellerData]);
  
  // Colors for each cluster
  const clusterColors = ['#0088FE', '#00C49F', '#FF8042'];
  
  // Names for each cluster
  const clusterNames = {
    0: 'High Performers',
    1: 'Average Performers',
    2: 'Low Performers'
  };
  
  // Custom tooltip to display seller details
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box sx={{ bgcolor: 'background.paper', p: 1, border: '1px solid #ccc' }}>
          <Typography variant="subtitle2">{data.name || 'Seller'}</Typography>
          <Typography variant="body2">Sales: ${data.x.toLocaleString()}</Typography>
          <Typography variant="body2">Processing Time: {data.y.toFixed(1)} days</Typography>
          <Typography variant="body2">Orders: {data.z}</Typography>
        </Box>
      );
    }
    return null;
  };
  
  // If no data, show a message instead of an empty chart
  if (processedData.length === 0) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <Typography color="text.secondary">
          No seller performance data available
        </Typography>
      </Box>
    );
  }
  
  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="Sales"
            unit="$"
            label={{ value: 'Sales ($)', position: 'bottom', offset: 0 }}
            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Processing Time"
            unit=" days"
            label={{ value: 'Processing Time (days)', angle: -90, position: 'insideLeft', offset: 0 }}
          />
          <ZAxis type="number" dataKey="z" range={[50, 400]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {processedData.map((cluster) => (
            <Scatter
              key={cluster.cluster}
              name={clusterNames[cluster.cluster] || `Cluster ${cluster.cluster}`}
              data={cluster.data}
              fill={clusterColors[cluster.cluster % clusterColors.length]}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default SellerPerformanceChart;
