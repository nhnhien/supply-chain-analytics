import React from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  CardHeader, Divider, List, ListItem, ListItemText
} from '@mui/material';
import { 
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Report as ReportIcon,
  CheckCircle as CheckCircleIcon,
  LocalShipping as ShippingIcon,
  Inventory as InventoryIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Import custom components
import KPICard from '../components/KPICard';
import TopCategoriesChart from '../components/TopCategoriesChart';
import SellerPerformanceChart from '../components/SellerPerformanceChart';

const DashboardPage = ({ data }) => {
  if (!data) {
    return <Typography>No dashboard data available</Typography>;
  }
  
  const { demandData, categories, forecasts, sellerPerformance, kpis, recommendations } = data;
  
  // Format KPI display values
  const formattedKPIs = {
    processingTime: kpis.avg_processing_time.toFixed(1),
    forecastGrowth: kpis.forecast_growth.toFixed(1),
    onTimeDelivery: kpis.on_time_delivery.toFixed(1),
    perfectOrderRate: kpis.perfect_order_rate.toFixed(1),
    inventoryTurnover: kpis.inventory_turnover.toFixed(1),
    totalDemand: new Intl.NumberFormat().format(kpis.total_demand)
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
                <Tooltip 
                  formatter={(value) => new Intl.NumberFormat().format(value)}
                  labelFormatter={(label) => {
                    if (label instanceof Date) {
                      return label.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
                    }
                    return label;
                  }}
                />
                <Legend />
                {categories.topCategories.slice(0, 3).map((category, index) => (
                  <Line 
                    key={category}
                    type="monotone" 
                    dataKey="count"
                    data={categories.categoryData[category]}
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
                {forecasts.performanceMetrics.slice(0, 5).map((forecast) => (
                  <ListItem key={forecast.category} divider>
                    <ListItemText 
                      primary={forecast.category} 
                      secondary={`MAPE: ${forecast.mape?.toFixed(2)}%, Growth: ${forecast.growth_rate?.toFixed(2)}%`} 
                    />
                    {forecast.growth_rate > 0 ? (
                      <TrendingUpIcon style={{ color: 'green' }} />
                    ) : (
                      <TrendingDownIcon style={{ color: 'red' }} />
                    )}
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
                {recommendations.inventory.slice(0, 5).map((rec, index) => (
                  <ListItem key={index} divider>
                    <ListItemText 
                      primary={rec.product_category || rec.category} 
                      secondary={
                        rec.recommendation || 
                        `Reorder at ${rec.reorder_point?.toFixed(0)} units, Safety stock: ${rec.safety_stock?.toFixed(0)} units`
                      } 
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;