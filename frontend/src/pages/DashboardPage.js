import React, { useState, useEffect, useMemo } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  FormControl, InputLabel, Select, MenuItem, Alert, AlertTitle, List, ListItem, ListItemText, Avatar, Tooltip 
} from '@mui/material';
import { 
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Report as ReportIcon,
  CheckCircle as CheckCircleIcon,
  LocalShipping as ShippingIcon,
  Inventory as InventoryIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import KPICard from '../components/KPICard';

// Import custom components
import TopCategoriesChart from '../components/TopCategoriesChart';
import SellerPerformanceChart from '../components/SellerPerformanceChart';

// Dummy MonthlyProfitChart component with corrected useEffect hook
const MonthlyProfitChart = ({ data }) => {
  useEffect(() => {
    // Example processing for monthly profit chart
    console.log("MonthlyProfitChart useEffect triggered");
  }, []); // Fixed: closing parenthesis added here
  
  return <div>Monthly Profit Chart Placeholder</div>;
};

const DashboardPage = ({ data }) => {
  if (!data) {
    return <Typography>No dashboard data available</Typography>;
  }
  
  // Destructure with default values to prevent undefined errors
  const { 
    demandData = [], 
    categories = { topCategories: [], categoryData: {} }, 
    forecasts = { performanceMetrics: [] }, 
    sellerPerformance = { clusters: [] }, 
    kpis = {}, 
    recommendations = { inventory: [] } 
  } = data;
  
  // Ensure kpis object exists
  if (!kpis) {
    return <Typography>No KPI data available</Typography>;
  }
  
  // Refactored processingTime with clearer logic
  let processingTimeValue = '0.5';
  if (kpis.avg_processing_time !== undefined) {
    processingTimeValue = kpis.avg_processing_time.toFixed(1);
  } else if (data && data.performance && data.performance.metrics && data.performance.metrics.avg_processing_time !== undefined) {
    processingTimeValue = data.performance.metrics.avg_processing_time.toFixed(1);
  }
  
  const formattedKPIs = {
    processingTime: processingTimeValue,
    forecastGrowth: (() => {
      // Calculate average growth rate from all forecasts
      if (forecasts && forecasts.forecastReport && forecasts.forecastReport.length > 0) {
        const validGrowthRates = forecasts.forecastReport
          .filter(f => f.growth_rate !== null && f.growth_rate !== undefined)
          .map(f => parseFloat(f.growth_rate));
        
        if (validGrowthRates.length > 0) {
          // Calculate average growth rate, avoiding extreme values
          const sumGrowth = validGrowthRates.reduce((sum, rate) => {
            const clippedRate = Math.max(Math.min(rate, 100), -80);
            return sum + clippedRate;
          }, 0);
          return (sumGrowth / validGrowthRates.length).toFixed(1);
        }
      }
      return '0.0';
    })(),
    onTimeDelivery: kpis.on_time_delivery !== undefined 
      ? kpis.on_time_delivery.toFixed(1) 
      : data && data.performance && data.performance.metrics && data.performance.metrics.on_time_delivery_rate !== undefined
        ? data.performance.metrics.on_time_delivery_rate.toFixed(1)
        : '85.0',
    perfectOrderRate: kpis.perfect_order_rate !== undefined 
      ? kpis.perfect_order_rate.toFixed(1) 
      : kpis.on_time_delivery !== undefined 
        ? (kpis.on_time_delivery * 0.9).toFixed(1)
        : '75.0',
    inventoryTurnover: kpis.inventory_turnover !== undefined 
      ? kpis.inventory_turnover.toFixed(1) 
      : '8.0',
    totalDemand: (() => {
      if (kpis.total_demand !== undefined) {
        return new Intl.NumberFormat().format(kpis.total_demand);
      }
      if (demandData && Array.isArray(demandData)) {
        const totalDemand = demandData.reduce((sum, row) => {
          return sum + (parseFloat(row.count || row.order_count || 0) || 0);
        }, 0);
        return new Intl.NumberFormat().format(totalDemand);
      }
      if (data && data.processed_orders && Array.isArray(data.processed_orders)) {
        return new Intl.NumberFormat().format(data.processed_orders.length);
      }
      return 'N/A';
    })()
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Supply Chain Analytics Dashboard
      </Typography>
      
      {/* KPI Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={4} lg={2}>
        <KPICard 
            title="Processing Time"
            value={`${formattedKPIs.processingTime} days`}
            icon={<ShippingIcon />}
            color="#1976d2"
            isEstimated={kpis.estimated_fields && kpis.estimated_fields.includes('avg_processing_time')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Forecast Growth"
            value={`${formattedKPIs.forecastGrowth}%`}
            icon={<TrendingUpIcon />}
            color="#2e7d32"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="On-Time Delivery"
            value={`${formattedKPIs.onTimeDelivery}%`}
            icon={<CheckCircleIcon />}
            color="#ed6c02"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Perfect Order Rate"
            value={`${formattedKPIs.perfectOrderRate}%`}
            icon={<ReportIcon />}
            color="#9c27b0"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Inventory Turnover"
            value={`${formattedKPIs.inventoryTurnover}x`}
            icon={<InventoryIcon />}
            color="#d32f2f"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Total Demand"
            value={formattedKPIs.totalDemand}
            icon={<TrendingUpIcon />}
            color="#0288d1"
          />
        </Grid>
      </Grid>
      
      {/* Main Dashboard Content */}
      <Grid container spacing={3}>
        {/* Demand Trends */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Monthly Demand Trends
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={demandData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(date) => {
                    if (date instanceof Date) {
                      return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                    }
                    return '';
                  }}
                />
                <YAxis />
                <RechartsTooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  labelFormatter={(label) => {
                    if (label instanceof Date) {
                      return label.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
                    }
                    return label;
                  }}
                />
                <Legend />
                {(categories.topCategories || []).slice(0, 3).map((category, index) => (
                  <Line 
                    key={category}
                    type="monotone" 
                    dataKey="count"
                    data={categories.categoryData && categories.categoryData[category] ? categories.categoryData[category] : []}
                    name={category}
                    stroke={['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c'][index % 5]}
                    activeDot={{ r: 8 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {/* Top Categories */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Top Product Categories
            </Typography>
            <TopCategoriesChart 
              categories={categories.topCategories} 
              categoryData={categories.categoryData}
            />
          </Paper>
        </Grid>
        
        {/* Forecast Summary */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Forecast Performance
            </Typography>
            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
              <List>
                {(forecasts.performanceMetrics || []).slice(0, 5).map((forecast) => (
                  <ListItem key={forecast?.category || 'unknown'} divider>
                    <ListItemText 
                      primary={forecast?.category || 'Unknown Category'} 
                      secondary={`MAPE: ${forecast?.mape != null ? forecast.mape.toFixed(2) : 'N/A'}%, Growth: ${forecast?.growth_rate != null ? forecast.growth_rate.toFixed(2) : 'N/A'}%`} 
                    />
                    {forecast?.growth_rate != null ? (
                      forecast.growth_rate > 0 ? (
                        <TrendingUpIcon style={{ color: 'green' }} />
                      ) : (
                        <TrendingDownIcon style={{ color: 'red' }} />
                      )
                    ) : null}
                  </ListItem>
                ))}
              </List>
            </Box>
          </Paper>
        </Grid>
        
        {/* Seller Performance */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Seller Performance Clusters
            </Typography>
            <SellerPerformanceChart sellerData={sellerPerformance.clusters} />
          </Paper>
        </Grid>
        
        {/* Recommendations */}
        <Grid item xs={12} md={6} lg={4}>
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 360 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Inventory Recommendations
            </Typography>
            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
              <List>
                {(recommendations.inventory || []).slice(0, 5).map((rec, index) => (
                  <ListItem key={index} divider>
                    <ListItemText 
                      primary={rec?.product_category || rec?.category || 'Unknown Category'} 
                      secondary={
                        rec?.recommendation || 
                        (rec?.reorder_point != null && rec?.safety_stock != null) ?
                          `Reorder at ${rec.reorder_point.toFixed(0)} units, Safety stock: ${rec.safety_stock.toFixed(0)} units` :
                          'No recommendation available'
                      } 
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Include MonthlyProfitChart to demonstrate fixed useEffect hook */}
      <MonthlyProfitChart data={[]} />
      
    </Box>
  );
};

export default DashboardPage;