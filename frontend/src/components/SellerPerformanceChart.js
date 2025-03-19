import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SellerPerformanceChart = ({ sellerData }) => {
  const [processedData, setProcessedData] = useState([]);

  useEffect(() => {
    let isMounted = true;
    // Heavy data processing: group sellers by cluster.
    const computeData = () => {
      if (!sellerData || sellerData.length === 0) return [];
      const clusters = {};
      sellerData.forEach(seller => {
        if (!seller) return; // Skip null/undefined sellers
        // Validate prediction: if not 0, 1, or 2, default to 1 (medium)
        let cluster = seller.prediction;
        if (cluster !== 0 && cluster !== 1 && cluster !== 2) {
          cluster = 1;
        }
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
    };

    const result = computeData();
    if (isMounted) {
      setProcessedData(result);
    }
    // Cleanup function to clear heavy data when component unmounts.
    return () => {
      isMounted = false;
      setProcessedData([]);
    };
  }, [sellerData]);

  // Colors and names for clusters
  const clusterColors = ['#0088FE', '#00C49F', '#FF8042'];
  const clusterNames = {
    0: 'High Performers',
    1: 'Average Performers',
    2: 'Low Performers'
  };

  // Compute dynamic Z-axis range based on all order_count values.
  const dynamicZRange = useMemo(() => {
    const allZ = processedData.flatMap(cluster => cluster.data.map(d => d.z));
    if (allZ.length === 0) return [50, 400];
    const dataMin = Math.min(...allZ);
    const dataMax = Math.max(...allZ);
    const epsilon = 1e-6;
    const delta = dataMax - dataMin;
    if (delta < epsilon) return [50, 50]; // Use fixed size if values are uniform.
    const desiredMaxSize = 400;
    const scale = (desiredMaxSize - 50) / delta;
    const dynamicMax = 50 + delta * scale;
    return [50, dynamicMax];
  }, [processedData]);

  // Custom tooltip for seller details.
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

  // If no processed data, show a friendly message.
  if (processedData.length === 0) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: 400,  
        minHeight: 400,
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
      <ResponsiveContainer width="100%" height={400}>
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
          <ZAxis type="number" dataKey="z" range={dynamicZRange} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {processedData.map(cluster => (
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
