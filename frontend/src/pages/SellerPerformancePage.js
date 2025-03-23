import React, { useState, useEffect, useMemo } from "react";
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Avatar,
  TablePagination,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  IconButton,
  useTheme,
  Stack,
  Alert,
  Button,
} from "@mui/material";
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
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from "recharts";
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  StarHalf as StarHalfIcon,
  LocalShipping as ShippingIcon,
  LocalShipping,
  AttachMoney as MoneyIcon,
  Person as PersonIcon,
  Info as InfoIcon,
  FilterList as FilterListIcon,
  DonutLarge as DonutLargeIcon,
  BarChart as BarChartIcon,
  ScatterPlot as ScatterPlotIcon,
  TableChart as TableChartIcon,
  Insights as InsightsIcon,
  CheckCircleOutline
} from '@mui/icons-material';

// Enhanced Winsorization function with proper percentile calculation
const applyWinsorization = (
  data,
  field,
  lowerPercentile = 0,
  upperPercentile = 99
) => {
  if (!data || data.length === 0 || !field) return data;

  // Extract field values and filter out invalid ones
  const values = data
    .map((item) => parseFloat(item?.[field] || 0))
    .filter((val) => !isNaN(val) && isFinite(val));

  if (values.length === 0) return data;

  // Calculate percentiles
  const lowerCap = calculatePercentile(values, lowerPercentile);
  const upperCap = calculatePercentile(values, upperPercentile);

  console.log(
    `Winsorizing ${field}: lower cap at ${lowerCap}, upper cap at ${upperCap}`
  );

  // Apply caps to data
  return data.map((item) => {
    if (!item) return item;

    let value = parseFloat(item[field]);
    if (!isNaN(value) && isFinite(value)) {
      if (value < lowerCap) value = lowerCap;
      if (value > upperCap) value = upperCap;
      return { ...item, [field]: value };
    }
    return item;
  });
};

// Helper function to calculate percentile
const calculatePercentile = (array, percentile) => {
  if (!array || array.length === 0) return 0;
  const sorted = [...array].sort((a, b) => a - b);
  const pos = ((sorted.length - 1) * percentile) / 100;
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
    .map((item) => parseFloat(item?.[field] || 0))
    .filter((val) => !isNaN(val) && isFinite(val));

  if (values.length <= 1) return data; // Not enough data points to normalize

  const min = Math.min(...values);
  const max = Math.max(...values);

  // If min and max are the same, return original data
  if (Math.abs(max - min) < 1e-6) return data;

  const [targetMin, targetMax] = targetRange;
  const scale = (targetMax - targetMin) / (max - min);

  const normalizedField = `normalized_${field}`;
  return data.map((item) => {
    if (!item) return item;

    let value = parseFloat(item[field]);
    if (!isNaN(value) && isFinite(value)) {
      const normalizedValue = targetMin + (value - min) * scale;
      return { ...item, [normalizedField]: normalizedValue };
    }

    return { ...item, [normalizedField]: targetMin };
  });
};

const SellerPerformancePage = ({ data }) => {
  const theme = useTheme();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [clusterFilter, setClusterFilter] = useState("all");
  const [sellerData, setSellerData] = useState([]);
  const [processedData, setProcessedData] = useState({
    winsorized: [],
    normalized: [],
    clusterMetrics: [],
    clusterDistribution: [],
    scatterData: [],
  });

  // Define consistent cluster names
  const clusterNames = {
    0: "High Performers",
    1: "Medium Performers",
    2: "Low Performers",
  };

  // For singular form used in table display
  const clusterSingularNames = {
    0: "High Performer",
    1: "Medium Performer",
    2: "Low Performer",
  };

  // Colors for visualization
  const clusterColors = {
    0: theme.palette.success.main, // High performers - green
    1: theme.palette.warning.main, // Medium performers - amber
    2: theme.palette.error.main, // Low performers - red
  };

  // Process and transform data when it changes
  useEffect(() => {
    if (data && data.clusters) {
      const rawData = data.clusters;

      // Apply data cleansing and transformation pipeline
      const processData = () => {
        // Step 1: Clean the data - handle missing values and ensure types
        const cleanedData = rawData
          .map((seller) => {
            if (!seller) return null;

            // Ensure prediction is properly parsed as a number
            let prediction = null;
            try {
              prediction =
                typeof seller.prediction === "string"
                  ? parseInt(seller.prediction, 10)
                  : seller.prediction;

              // Validate prediction is in valid range
              prediction = [0, 1, 2].includes(prediction) ? prediction : 1; // Default to Medium (1)
            } catch (e) {
              prediction = 1; // Default to Medium (1) on error
            }

            return {
              ...seller,
              // Ensure all necessary fields have valid values
              seller_id:
                seller.seller_id ||
                `unknown-${Math.random().toString(36).substr(2, 9)}`,
              total_sales: parseFloat(seller.total_sales) || 0,
              avg_processing_time: parseFloat(seller.avg_processing_time) || 0,
              avg_delivery_days: parseFloat(seller.avg_delivery_days) || 0,
              order_count: parseFloat(seller.order_count) || 0,
              on_time_delivery_rate:
                parseFloat(seller.on_time_delivery_rate) || 0,
              prediction: prediction,
            };
          })
          .filter((seller) => seller !== null);

        // Step 2: Apply Winsorization to cap extreme values
        const winsorizedData = applyWinsorization(
          cleanedData,
          "total_sales",
          0,
          99
        );

        // Step 3: Normalize data for visualization
        const normalizedData = normalizeDataForVisualization(
          winsorizedData,
          "total_sales",
          [10, 100]
        );

        // Step 4: Calculate cluster metrics
        const clusters = [0, 1, 2];

        const clusterMetrics = clusters.map((cluster) => {
          const clusteredSellers = winsorizedData.filter(
            (seller) => seller.prediction === cluster
          );

          if (clusteredSellers.length === 0) {
            console.log(`No sellers found for cluster ${cluster}`);
            // Return dummy data for empty clusters to maintain UI consistency
            return {
              cluster,
              clusterName: clusterNames[cluster],
              count: 0,
              avgProcessingTime: 0,
              avgDeliveryDays: 0,
              avgOrderCount: 0,
              totalSales: 0,
              avgOnTimeRate: 0,
            };
          }

          const avgProcessingTime =
            clusteredSellers.reduce(
              (sum, seller) => sum + seller.avg_processing_time,
              0
            ) / clusteredSellers.length;

          const avgDeliveryDays =
            clusteredSellers.reduce(
              (sum, seller) => sum + seller.avg_delivery_days,
              0
            ) / clusteredSellers.length;

          const avgOrderCount =
            clusteredSellers.reduce(
              (sum, seller) => sum + seller.order_count,
              0
            ) / clusteredSellers.length;

          const totalSales = clusteredSellers.reduce(
            (sum, seller) => sum + seller.total_sales,
            0
          );

          const avgOnTimeRate =
            clusteredSellers.reduce(
              (sum, seller) => sum + seller.on_time_delivery_rate,
              0
            ) / clusteredSellers.length;

          return {
            cluster,
            clusterName: clusterNames[cluster],
            count: clusteredSellers.length,
            avgProcessingTime,
            avgDeliveryDays,
            avgOrderCount,
            totalSales,
            avgOnTimeRate,
          };
        });

        // Make sure we have all three clusters represented
        const ensureAllClusters = () => {
          const result = [];

          // Create a map of existing clusters
          const clusterMap = {};
          clusterMetrics.forEach((metric) => {
            clusterMap[metric.cluster] = metric;
          });

          // Ensure all three clusters exist
          for (let i = 0; i < 3; i++) {
            if (clusterMap[i]) {
              result.push(clusterMap[i]);
            } else {
              // Add a placeholder for missing clusters
              result.push({
                cluster: i,
                clusterName: clusterNames[i],
                count: 0,
                avgProcessingTime: 0,
                avgDeliveryDays: 0,
                avgOrderCount: 0,
                totalSales: 0,
                avgOnTimeRate: 0,
              });
            }
          }

          return result.sort((a, b) => a.cluster - b.cluster);
        };

        // Replace the clusterMetrics with the complete set
        const completeClusterMetrics = ensureAllClusters();

        // Filter out empty clusters and sort by cluster number for consistent order
        const validClusterMetrics = completeClusterMetrics
          .filter((metric) => metric.count > 0)
          .sort((a, b) => a.cluster - b.cluster);

        // Log for debugging
        console.log("Cluster metrics:", validClusterMetrics);

        // Step 5: Calculate cluster distribution
        const totalSellers = winsorizedData.length;
        const clusterDistribution = validClusterMetrics.map((metric) => ({
          name: metric.clusterName,
          value: metric.count,
          percentage:
            totalSellers > 0 ? (metric.count / totalSellers) * 100 : 0,
          cluster: metric.cluster, // Keep track of the original cluster number
        }));

        // Log for debugging
        console.log("Cluster distribution:", clusterDistribution);

        // Step 6: Prepare scatter plot data
        const scatterData = clusters
          .filter((cluster) =>
            winsorizedData.some((seller) => seller.prediction === cluster)
          )
          .map((cluster) => ({
            cluster,
            name: clusterNames[cluster],
            data: normalizedData
              .filter((seller) => seller.prediction === cluster)
              .map((seller) => ({
                x: seller.total_sales,
                y: seller.avg_processing_time,
                z: Math.max(20, seller.order_count), // Minimum size for visibility
                name: seller.seller_id,
                raw: {
                  sales: seller.total_sales,
                  processing: seller.avg_processing_time,
                  orders: seller.order_count,
                  delivery: seller.avg_delivery_days,
                  onTime: seller.on_time_delivery_rate,
                },
              })),
          }));

        // Debug logging
        console.log("Final cluster metrics:", completeClusterMetrics);
        console.log(
          "Final cluster metrics by name:",
          completeClusterMetrics.map((m) => m.clusterName)
        );
        console.log("Performance metrics data for chart:", [
          {
            clusterName: "High Performers",
            avgProcessingTime:
              completeClusterMetrics.find((m) => m.cluster === 0)
                ?.avgProcessingTime || 0,
            avgOrderCount:
              completeClusterMetrics.find((m) => m.cluster === 0)
                ?.avgOrderCount || 0,
          },
          {
            clusterName: "Medium Performers",
            avgProcessingTime:
              completeClusterMetrics.find((m) => m.cluster === 1)
                ?.avgProcessingTime || 0,
            avgOrderCount:
              completeClusterMetrics.find((m) => m.cluster === 1)
                ?.avgOrderCount || 0,
          },
          {
            clusterName: "Low Performers",
            avgProcessingTime:
              completeClusterMetrics.find((m) => m.cluster === 2)
                ?.avgProcessingTime || 0,
            avgOrderCount:
              completeClusterMetrics.find((m) => m.cluster === 2)
                ?.avgOrderCount || 0,
          },
        ]);

        return {
          winsorized: winsorizedData,
          normalized: normalizedData,
          clusterMetrics: completeClusterMetrics,
          clusterDistribution,
          scatterData,
        };
      };

      // Execute data processing and update state
      try {
        const processedResults = processData();
        setSellerData(processedResults.winsorized);
        setProcessedData(processedResults);
      } catch (error) {
        console.error("Error processing seller data:", error);
      }
    }
  }, [data]);

  if (!data || !data.clusters || data.clusters.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography
          variant="h4"
          gutterBottom
          sx={{
            fontWeight: "bold",
            color: theme.palette.primary.main,
            borderBottom: `2px solid ${theme.palette.divider}`,
            pb: 1,
          }}
        >
          <InsightsIcon sx={{ mr: 1, verticalAlign: "text-bottom" }} />
          Seller Performance Analysis
        </Typography>

        <Alert
          severity="info"
          variant="filled"
          sx={{
            mt: 4,
            boxShadow: theme.shadows[3],
            "& .MuiAlert-icon": {
              fontSize: "2rem",
            },
          }}
        >
          <Typography variant="h6" sx={{ mb: 1, fontWeight: "medium" }}>
            No Seller Performance Data Available
          </Typography>
          <Typography variant="body1">
            Run the supply chain analysis first to generate seller performance
            insights.
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

  // Handle cluster filter change
  const handleClusterFilterChange = (event) => {
    setClusterFilter(event.target.value);
    setPage(0);
  };

  // Filter seller data based on selected cluster
  const filteredSellers = sellerData.filter((seller) => {
    if (clusterFilter === "all") return true;
    return seller.prediction === parseInt(clusterFilter);
  });

  // Get visible rows for pagination
  const visibleRows = filteredSellers.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  // Map performance rating from 1-5 stars based on cluster and relative performance
  const getPerformanceRating = (seller) => {
    if (!seller) return 0;

    const cluster = seller.prediction;
    // Base rating on cluster (0: 5 stars, 1: 3 stars, 2: 1 star)
    let baseRating = cluster === 0 ? 5 : cluster === 1 ? 3 : 1;

    // Adjust within cluster based on processing time and on-time delivery
    const clusterMetric = processedData.clusterMetrics.find(
      (metric) => metric.cluster === cluster
    );
    if (!clusterMetric) return baseRating;

    const clusterAvgTime = clusterMetric.avgProcessingTime;
    const sellerTime = seller.avg_processing_time || 0;

    // For processing time, lower is better, so we give better rating if below average
    const timeAdjustment =
      clusterAvgTime > 0
        ? Math.round(((clusterAvgTime - sellerTime) / clusterAvgTime) * 0.5) // +/- 0.5 star based on processing time
        : 0;

    const clusterAvgOnTime = clusterMetric.avgOnTimeRate;
    const sellerOnTime = seller.on_time_delivery_rate || 0;

    // For on-time delivery, higher is better
    const onTimeAdjustment =
      clusterAvgOnTime > 0
        ? Math.round(
            ((sellerOnTime - clusterAvgOnTime) / clusterAvgOnTime) * 0.5
          ) // +/- 0.5 star based on on-time rate
        : 0;

    return Math.max(
      1,
      Math.min(5, baseRating + timeAdjustment + onTimeAdjustment)
    );
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
        <Card
          sx={{
            p: 1.5,
            border: `1px solid ${theme.palette.divider}`,
            maxWidth: 250,
            boxShadow: theme.shadows[3],
            borderRadius: 1,
          }}
        >
          <Typography variant="subtitle2" fontWeight="bold">
            {data.name || "Seller"}
          </Typography>
          <Divider sx={{ my: 1 }} />
          <Stack spacing={0.5}>
            <Typography
              variant="body2"
              sx={{ display: "flex", alignItems: "center" }}
            >
              <MoneyIcon
                fontSize="small"
                sx={{ mr: 1, color: theme.palette.success.main }}
              />
              <strong>Sales:</strong> ${data.raw.sales.toLocaleString()}
            </Typography>
            <Typography
              variant="body2"
              sx={{ display: "flex", alignItems: "center" }}
            >
              <ShippingIcon
                fontSize="small"
                sx={{ mr: 1, color: theme.palette.primary.main }}
              />
              <strong>Processing Time:</strong> {data.raw.processing.toFixed(1)}{" "}
              days
            </Typography>
            <Typography
              variant="body2"
              sx={{ display: "flex", alignItems: "center" }}
            >
              <TrendingUpIcon
                fontSize="small"
                sx={{ mr: 1, color: theme.palette.info.main }}
              />
              <strong>Orders:</strong> {data.raw.orders}
            </Typography>
            <Typography
              variant="body2"
              sx={{ display: "flex", alignItems: "center" }}
            >
              <StarIcon
                fontSize="small"
                sx={{ mr: 1, color: theme.palette.warning.main }}
              />
              <strong>On-Time Delivery:</strong> {data.raw.onTime.toFixed(1)}%
            </Typography>
          </Stack>
        </Card>
      );
    }
    return null;
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{
          fontWeight: "bold",
          color: theme.palette.primary.main,
          borderBottom: `2px solid ${theme.palette.divider}`,
          pb: 1,
          mb: 3,
          display: "flex",
          alignItems: "center",
        }}
      >
        <InsightsIcon sx={{ mr: 1 }} />
        Seller Performance Analysis
      </Typography>

      <Grid container spacing={3}>
   {/* Main content section heading */}
<Grid item xs={12}>
  <Box sx={{ mt: 1, mb: 1 }}>
    <Typography variant="h5" component="h2" sx={{ 
      fontWeight: 'medium',
      borderLeft: `4px solid ${theme.palette.primary.main}`,
      pl: 2
    }}>
      Performance Overview
    </Typography>
  </Box>
</Grid>

{/* Cluster Distribution */}
<Grid item xs={12} md={8} lg={6}>
  <Paper elevation={3} sx={{ 
    p: 3, 
    display: 'flex', 
    flexDirection: 'column', 
    height: 350,
    width: '100%',
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
      <DonutLargeIcon sx={{ mr: 1 }} /> Seller Performance Clusters
    </Typography>
    
    <Box sx={{ 
      flexGrow: 1, 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      width: '100%',
      minHeight: 250
    }}>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={[
              { name: "High Performers", value: processedData.clusterMetrics.find(m => m.cluster === 0)?.count || 0, cluster: 0 },
              { name: "Medium Performers", value: processedData.clusterMetrics.find(m => m.cluster === 1)?.count || 0, cluster: 1 },
              { name: "Low Performers", value: processedData.clusterMetrics.find(m => m.cluster === 2)?.count || 0, cluster: 2 }
            ].filter(item => item.value > 0)}
            cx="35%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            fill="#8884d8"
            paddingAngle={5}
            dataKey="value"
            strokeWidth={2}
            stroke={theme.palette.background.paper}
            label={false}
          >
            {[0, 1, 2].map((cluster) => (
              <Cell 
                key={`cell-${cluster}`}
                fill={clusterColors[cluster]} 
              />
            ))}
          </Pie>
          <Legend 
            formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
            layout="vertical"
            verticalAlign="middle"
            align="right"
            wrapperStyle={{ 
              paddingLeft: '20px', 
              right: 0
            }}
          />
          <RechartsTooltip 
            formatter={(value, name) => [
              `${value} sellers (${(value / sellerData.length * 100).toFixed(1)}%)`, 
              name
            ]}
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '8px',
              boxShadow: theme.shadows[3]
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  </Paper>
</Grid>

{/* Cluster Metrics */}
<Grid item xs={12} md={8} lg={6}>
  <Paper elevation={3} sx={{ 
    p: 3, 
    display: 'flex', 
    flexDirection: 'column', 
    height: 350,
    width: '100%',
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
      <BarChartIcon sx={{ mr: 1 }} /> Performance Metrics by Cluster
    </Typography>
    
    <Box sx={{ flexGrow: 1 }}>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={[
            {
              clusterName: "High Performers",
              avgProcessingTime: processedData.clusterMetrics.find(m => m.cluster === 0)?.avgProcessingTime || 0,
              avgOrderCount: processedData.clusterMetrics.find(m => m.cluster === 0)?.avgOrderCount || 0
            },
            {
              clusterName: "Medium Performers",
              avgProcessingTime: processedData.clusterMetrics.find(m => m.cluster === 1)?.avgProcessingTime || 0,
              avgOrderCount: processedData.clusterMetrics.find(m => m.cluster === 1)?.avgOrderCount || 0
            },
            {
              clusterName: "Low Performers",
              avgProcessingTime: processedData.clusterMetrics.find(m => m.cluster === 2)?.avgProcessingTime || 0,
              avgOrderCount: processedData.clusterMetrics.find(m => m.cluster === 2)?.avgOrderCount || 0
            }
          ]}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="clusterName" 
            tick={{ fontSize: 12, fill: theme.palette.text.secondary }} 
          />
          <YAxis 
            yAxisId="left" 
            orientation="left" 
            stroke={theme.palette.primary.main}
            tick={{ fill: theme.palette.text.secondary }}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            stroke={theme.palette.secondary.main}
            tick={{ fill: theme.palette.text.secondary }}
          />
          <RechartsTooltip 
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '8px',
              boxShadow: theme.shadows[3]
            }}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
          />
          <Bar 
            yAxisId="left" 
            dataKey="avgProcessingTime" 
            name="Avg. Processing Time (days)" 
            fill={theme.palette.primary.main}
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            yAxisId="right" 
            dataKey="avgOrderCount" 
            name="Avg. Order Count" 
            fill={theme.palette.secondary.main}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  </Paper>
</Grid>

{/* Cluster Scatterplot - Now twice as wide */}
<Grid item xs={12} md={12} lg={12}>
  <Paper elevation={3} sx={{ 
    p: 3, 
    display: 'flex', 
    flexDirection: 'column', 
    height: 350,
    width: '100%',
    borderRadius: 2,
    boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
  }}>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
      <Typography component="h2" variant="h6" color="primary" sx={{ 
        fontWeight: 'bold',
        display: 'flex',
        alignItems: 'center',
        borderBottom: `1px solid ${theme.palette.divider}`,
        pb: 1,
        width: '100%'
      }}>
        <ScatterPlotIcon sx={{ mr: 1 }} /> Sales vs. Processing Time
        <Tooltip title="Data points have been scaled to improve visualization. Hover over points to see actual values.">
          <IconButton size="small" sx={{ ml: 'auto' }}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Typography>
    </Box>
    
    <Box sx={{ flexGrow: 1 }}>
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid stroke={theme.palette.divider} />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Sales" 
            tick={{ fill: theme.palette.text.secondary }}
            label={{ 
              value: 'Sales', 
              position: 'bottom', 
              offset: 0,
              fill: theme.palette.text.secondary
            }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="Processing Time" 
            tick={{ fill: theme.palette.text.secondary }}
            label={{ 
              value: 'Processing Time (days)', 
              angle: -90, 
              position: 'insideLeft',
              fill: theme.palette.text.secondary
            }}
          />
          <ZAxis type="number" dataKey="z" range={[50, 400]} />
          <RechartsTooltip content={<ScatterTooltip />} />
          <Legend 
            formatter={(value) => <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>{value}</span>}
          />
          
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

        {/* Table section heading */}
        <Grid item xs={12}>
          <Box sx={{ mt: 4, mb: 1 }}>
            <Typography
              variant="h5"
              component="h2"
              sx={{
                fontWeight: "medium",
                borderLeft: `4px solid ${theme.palette.secondary.main}`,
                pl: 2,
              }}
            >
              Seller Details
            </Typography>
          </Box>
        </Grid>

        {/* Table Filter Controls */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              p: 3,
              mb: 2,
              borderRadius: 2,
              boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            }}
          >
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <Typography
                  variant="h6"
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    color: theme.palette.secondary.main,
                    fontWeight: "bold",
                  }}
                >
                  <FilterListIcon sx={{ mr: 1 }} />
                  Filter Sellers
                  <Chip
                    label={`${filteredSellers.length} sellers`}
                    size="small"
                    color="primary"
                    sx={{ ml: 2 }}
                  />
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth variant="outlined" size="small">
                  <InputLabel id="cluster-filter-label">
                    Cluster Filter
                  </InputLabel>
                  <Select
                    labelId="cluster-filter-label"
                    id="cluster-filter"
                    value={clusterFilter}
                    onChange={handleClusterFilterChange}
                    label="Cluster Filter"
                    sx={{
                      "& .MuiSelect-select": {
                        display: "flex",
                        alignItems: "center",
                      },
                    }}
                  >
                    <MenuItem
                      value="all"
                      sx={{ display: "flex", alignItems: "center" }}
                    >
                      <DonutLargeIcon sx={{ mr: 1 }} /> All Clusters
                    </MenuItem>
                    <MenuItem
                      value="0"
                      sx={{ display: "flex", alignItems: "center" }}
                    >
                      <Box
                        component="span"
                        sx={{
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          bgcolor: clusterColors[0],
                          mr: 1,
                        }}
                      />
                      High Performers
                    </MenuItem>
                    <MenuItem
                      value="1"
                      sx={{ display: "flex", alignItems: "center" }}
                    >
                      <Box
                        component="span"
                        sx={{
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          bgcolor: clusterColors[1],
                          mr: 1,
                        }}
                      />
                      Medium Performers
                    </MenuItem>
                    <MenuItem
                      value="2"
                      sx={{ display: "flex", alignItems: "center" }}
                    >
                      <Box
                        component="span"
                        sx={{
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          bgcolor: clusterColors[2],
                          mr: 1,
                        }}
                      />
                      Low Performers
                    </MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Seller Table */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              width: "100%",
              borderRadius: 2,
              boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
              overflow: "hidden",
            }}
          >
            <TableContainer>
              <Typography
                component="h2"
                variant="h6"
                sx={{
                  p: 2,
                  pl: 3,
                  color: theme.palette.secondary.main,
                  fontWeight: "bold",
                  display: "flex",
                  alignItems: "center",
                  borderBottom: `1px solid ${theme.palette.divider}`,
                }}
              >
                <TableChartIcon sx={{ mr: 1 }} /> Seller Performance Data
              </Typography>
              <Table>
                <TableHead>
                  <TableRow
                    sx={{
                      backgroundColor: theme.palette.action.hover,
                    }}
                  >
                    <TableCell sx={{ fontWeight: "bold" }}>Seller ID</TableCell>
                    <TableCell sx={{ fontWeight: "bold" }}>Cluster</TableCell>
                    <TableCell align="right" sx={{ fontWeight: "bold" }}>
                      Total Sales
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: "bold" }}>
                      Order Count
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: "bold" }}>
                      Avg. Processing Time
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: "bold" }}>
                      Avg. Delivery Days
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: "bold" }}>
                      On-Time Rate
                    </TableCell>
                    <TableCell sx={{ fontWeight: "bold" }}>
                      Performance Rating
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {visibleRows.map((seller, index) => {
                    const rating = getPerformanceRating(seller);

                    return (
                      <TableRow
                        key={seller.seller_id}
                        hover
                        sx={{
                          "&:nth-of-type(even)": {
                            backgroundColor: theme.palette.action.hover,
                          },
                          "&:hover": {
                            backgroundColor: theme.palette.action.selected,
                          },
                        }}
                      >
                        <TableCell component="th" scope="row">
                          <Box sx={{ display: "flex", alignItems: "center" }}>
                            <Avatar
                              sx={{
                                width: 30,
                                height: 30,
                                mr: 1,
                                bgcolor: clusterColors[seller.prediction],
                                boxShadow: theme.shadows[2],
                              }}
                            >
                              <PersonIcon />
                            </Avatar>
                            <Typography fontWeight="medium">
                              {seller.seller_id}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={clusterSingularNames[seller.prediction]}
                            size="small"
                            sx={{
                              bgcolor: clusterColors[seller.prediction],
                              color: "white",
                              fontWeight: "medium",
                            }}
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "flex-end",
                            }}
                          >
                            <MoneyIcon
                              fontSize="small"
                              sx={{
                                mr: 0.5,
                                color: theme.palette.success.main,
                              }}
                            />
                            $
                            {new Intl.NumberFormat().format(
                              seller.total_sales || 0
                            )}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          {new Intl.NumberFormat().format(
                            seller.order_count || 0
                          )}
                        </TableCell>
                        <TableCell align="right">
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "flex-end",
                            }}
                          >
                            <ShippingIcon
                              fontSize="small"
                              sx={{ mr: 0.5, color: theme.palette.info.main }}
                            />
                            {(seller.avg_processing_time || 0).toFixed(1)} days
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          {(seller.avg_delivery_days || 0).toFixed(1)} days
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            component="span"
                            sx={{
                              color:
                                seller.on_time_delivery_rate >= 90
                                  ? theme.palette.success.main
                                  : seller.on_time_delivery_rate >= 70
                                  ? theme.palette.warning.main
                                  : theme.palette.error.main,
                              fontWeight: "medium",
                            }}
                          >
                            {(seller.on_time_delivery_rate || 0).toFixed(1)}%
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: "flex" }}>
                            {renderStarRating(rating)}
                          </Box>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                  {visibleRows.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={8} align="center" sx={{ py: 3 }}>
                        <Typography color="text.secondary">
                          No sellers found matching the selected filter
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              component="div"
              count={filteredSellers.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[5, 10, 25, 50]}
              sx={{
                borderTop: `1px solid ${theme.palette.divider}`,
                "& .MuiTablePagination-toolbar": {
                  padding: "16px",
                },
              }}
            />
          </Paper>
        </Grid>

        {/* Insights section heading */}
        <Grid item xs={12}>
          <Box sx={{ mt: 4, mb: 1 }}>
            <Typography
              variant="h5"
              component="h2"
              sx={{
                fontWeight: "medium",
                borderLeft: `4px solid ${theme.palette.info.main}`,
                pl: 2,
              }}
            >
              Cluster Insights
            </Typography>
          </Box>
        </Grid>

        {/* Performance Insights */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              p: 3,
              borderRadius: 2,
              boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            }}
          >
            <Typography
              component="h2"
              variant="h6"
              gutterBottom
              sx={{
                color: theme.palette.info.main,
                fontWeight: "bold",
                display: "flex",
                alignItems: "center",
                borderBottom: `1px solid ${theme.palette.divider}`,
                pb: 1,
                mb: 3,
              }}
            >
              <InsightsIcon sx={{ mr: 1 }} /> Performance Insights by Cluster
            </Typography>

            <Grid container spacing={3}>
              {processedData.clusterMetrics.map((metric) => (
                <Grid item xs={12} md={4} key={metric.cluster}>
                  <Card
                    variant="outlined"
                    sx={{
                      borderRadius: 2,
                      borderColor: clusterColors[metric.cluster],
                      borderWidth: "2px",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        boxShadow: theme.shadows[8],
                        transform: "translateY(-4px)",
                      },
                    }}
                  >
                    <CardContent>
                      <Typography
                        variant="h6"
                        gutterBottom
                        sx={{
                          color: clusterColors[metric.cluster],
                          fontWeight: "bold",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                        }}
                      >
                        {metric.clusterName}
                        <Chip
                          label={`${metric.count} sellers`}
                          size="small"
                          sx={{
                            bgcolor: clusterColors[metric.cluster],
                            color: "white",
                            fontWeight: "medium",
                          }}
                        />
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      <Stack spacing={1.5}>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <PersonIcon fontSize="small" sx={{ mr: 0.5 }} />
                            Seller Count:
                          </Typography>
                          <Typography variant="body1" fontWeight="medium">
                            {metric.count} (
                            {sellerData.length > 0
                              ? (
                                  (metric.count / sellerData.length) *
                                  100
                                ).toFixed(1)
                              : "0"}
                            %)
                          </Typography>
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <ShippingIcon fontSize="small" sx={{ mr: 0.5 }} />
                            Processing Time:
                          </Typography>
                          <Typography variant="body1" fontWeight="medium">
                            {metric.avgProcessingTime.toFixed(1)} days
                          </Typography>
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <LocalShipping fontSize="small" sx={{ mr: 0.5 }} />
                            Delivery Days:
                          </Typography>
                          <Typography variant="body1" fontWeight="medium">
                            {metric.avgDeliveryDays.toFixed(1)} days
                          </Typography>
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} />
                            Order Count:
                          </Typography>
                          <Typography variant="body1" fontWeight="medium">
                            {metric.avgOrderCount.toFixed(0)} orders
                          </Typography>
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <StarIcon fontSize="small" sx={{ mr: 0.5 }} />
                            On-Time Rate:
                          </Typography>
                          <Typography
                            variant="body1"
                            fontWeight="medium"
                            sx={{
                              color:
                                metric.avgOnTimeRate >= 90
                                  ? theme.palette.success.main
                                  : metric.avgOnTimeRate >= 70
                                  ? theme.palette.warning.main
                                  : theme.palette.error.main,
                            }}
                          >
                            {metric.avgOnTimeRate.toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ display: "flex", alignItems: "center" }}
                          >
                            <MoneyIcon fontSize="small" sx={{ mr: 0.5 }} />
                            Total Sales:
                          </Typography>
                          <Typography variant="body1" fontWeight="medium">
                            $
                            {new Intl.NumberFormat().format(
                              Math.round(metric.totalSales)
                            )}
                          </Typography>
                        </Box>
                      </Stack>
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
