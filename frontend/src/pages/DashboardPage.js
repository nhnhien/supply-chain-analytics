import React, { useState, useMemo } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  FormControl, InputLabel, Select, MenuItem, Alert, AlertTitle, List, ListItem, ListItemText, Avatar, Tooltip,
  Divider, useTheme, useMediaQuery
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

// Import custom components
import TopCategoriesChart from '../components/TopCategoriesChart';
import SellerPerformanceChart from '../components/SellerPerformanceChart';
import KPICard from '../components/KPICard';
import ImprovedDemandChart from '../components/ImprovedDemandChart';

const DashboardPage = ({ data }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  // Always define this regardless of data existence (solves the React Hook conditional usage error)
  const formattedKPIs = useMemo(() => {
    if (!data || !data.kpis) {
      return {
        processingTime: '0.5',
        forecastGrowth: '0.0',
        onTimeDelivery: '85.0',
        perfectOrderRate: '75.0',
        inventoryTurnover: '8.0',
        totalDemand: 'N/A'
      };
    }

    const kpis = data.kpis;
    
    const processingTimeValue = kpis.avg_processing_time?.toFixed(1) ||
                                data?.performance?.metrics?.avg_processing_time?.toFixed(1) ||
                                '0.5';
    
    return {
      processingTime: processingTimeValue,
      forecastGrowth: (() => {
        if (data.forecasts && data.forecasts.forecastReport && data.forecasts.forecastReport.length > 0) {
          const validGrowthRates = data.forecasts.forecastReport
            .filter(f => f.growth_rate != null)
            .map(f => parseFloat(f.growth_rate));
          
          if (validGrowthRates.length > 0) {
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
        : data?.performance?.metrics?.on_time_delivery_rate?.toFixed(1) || '85.0',
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
        if (data.demandData && Array.isArray(data.demandData)) {
          const totalDemand = data.demandData.reduce((sum, row) => {
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
  }, [data]);

  if (!data) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Alert severity="info">
          <AlertTitle>No Data Available</AlertTitle>
          No dashboard data is currently available. Please check your connection or try again later.
        </Alert>
      </Box>
    );
  }
  
  // Destructure with default values to prevent undefined errors
  const { 
    demandData = [], 
    categories = { topCategories: [], categoryData: {} }, 
    forecasts = { performanceMetrics: [] }, 
    sellerPerformance = { clusters: [] }, 
    recommendations = { inventory: [] } 
  } = data;

  // Chart colors for consistency
  const chartColors = {
    primary: theme.palette.primary.main,
    secondary: theme.palette.secondary.main,
    success: theme.palette.success.main,
    warning: theme.palette.warning.main,
    error: theme.palette.error.main,
    info: theme.palette.info.main,
    chart: ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c']
  };

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ 
        fontWeight: 'bold', 
        color: theme.palette.primary.main,
        borderBottom: `2px solid ${theme.palette.divider}`,
        pb: 1,
        mb: 3
      }}>
        Supply Chain Analytics Dashboard
      </Typography>
      
      {/* KPI Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Processing Time"
            value={`${formattedKPIs.processingTime} days`}
            icon={<ShippingIcon />}
            color={chartColors.primary}
            isEstimated={data.kpis?.estimated_fields && data.kpis.estimated_fields.includes('avg_processing_time')}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Forecast Growth"
            value={`${formattedKPIs.forecastGrowth}%`}
            icon={<TrendingUpIcon />}
            color={chartColors.success}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="On-Time Delivery"
            value={`${formattedKPIs.onTimeDelivery}%`}
            icon={<CheckCircleIcon />}
            color={chartColors.warning}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Perfect Order Rate"
            value={`${formattedKPIs.perfectOrderRate}%`}
            icon={<ReportIcon />}
            color={chartColors.secondary}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Inventory Turnover"
            value={`${formattedKPIs.inventoryTurnover}x`}
            icon={<InventoryIcon />}
            color={chartColors.error}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2}>
          <KPICard 
            title="Total Demand"
            value={formattedKPIs.totalDemand}
            icon={<TrendingUpIcon />}
            color={chartColors.info}
            sx={{ 
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[4]
              }
            }}
          />
        </Grid>
      </Grid>
      
{/* Main Dashboard Content */}
<Box sx={{ mb: 2 }}>
  <Typography variant="h5" component="h2" sx={{ 
    mb: 2, 
    fontWeight: 'medium',
    borderLeft: `4px solid ${theme.palette.primary.main}`,
    pl: 2
  }}>
    Demand & Performance Analytics
  </Typography>
</Box>

<Grid container spacing={4}>
  {/* First row: Improved Historical Monthly Demand */}
  <Grid item xs={12}>
    <ImprovedDemandChart 
      demandData={demandData}
      categories={categories}
      theme={theme}
    />
  </Grid>
  
  {/* Second row: Top Product Categories */}
  <Grid item xs={12}>
    <Paper elevation={3} sx={{ 
      p: 4, 
      display: 'flex', 
      flexDirection: 'column', 
      height: 450,
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
        <InventoryIcon sx={{ mr: 1 }} /> Top Product Categories
      </Typography>
      <TopCategoriesChart 
        categories={categories.topCategories} 
        categoryData={categories.categoryData}
        chartType="bar"
        chartColors={chartColors.chart}
      />
    </Paper>
  </Grid>
  
  <Grid item xs={12} sx={{ mt: 2 }}>
    <Divider />
    <Box sx={{ my: 2 }}>
      <Typography variant="h5" component="h2" sx={{ 
        mb: 2,
        fontWeight: 'medium',
        borderLeft: `4px solid ${theme.palette.secondary.main}`,
        pl: 2
      }}>
        Forecasts & Recommendations
      </Typography>
    </Box>
  </Grid>
  
  {/* First row of Forecasts: Seller Performance Clusters (full width) */}
  <Grid item xs={12}>
    <Paper elevation={3} sx={{ 
      p: 4, 
      display: 'flex', 
      flexDirection: 'column', 
      height: 450,
      borderRadius: 2,
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
    }}>
      <Typography component="h2" variant="h6" gutterBottom sx={{ 
        color: theme.palette.secondary.main,
        fontWeight: 'bold',
        display: 'flex',
        alignItems: 'center',
        borderBottom: `1px solid ${theme.palette.divider}`,
        pb: 1
      }}>
        <CheckCircleIcon sx={{ mr: 1 }} /> Seller Performance Clusters
      </Typography>
      <SellerPerformanceChart 
        sellerData={sellerPerformance.clusters} 
        chartColors={chartColors.chart}
      />
    </Paper>
  </Grid>
  
  {/* Second row of Forecasts: Split into two columns */}
  {/* Forecast Performance */}
  <Grid item xs={12} md={6}>
    <Paper elevation={3} sx={{ 
      p: 4, 
      display: 'flex', 
      flexDirection: 'column', 
      height: 450,
      borderRadius: 2,
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
    }}>
      <Typography component="h2" variant="h6" gutterBottom sx={{ 
        color: theme.palette.secondary.main,
        fontWeight: 'bold',
        display: 'flex',
        alignItems: 'center',
        borderBottom: `1px solid ${theme.palette.divider}`,
        pb: 1
      }}>
        <TrendingUpIcon sx={{ mr: 1 }} /> Forecast Performance
      </Typography>
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        <List sx={{ px: 1 }}>
          {(forecasts.performanceMetrics || []).slice(0, 5).map((forecast, index) => (
            <ListItem 
              key={forecast?.category || `unknown-${index}`} 
              divider={index < (forecasts.performanceMetrics || []).slice(0, 5).length - 1}
              sx={{ 
                borderRadius: 1,
                mb: 1,
                p: 2,
                backgroundColor: index % 2 === 0 ? 'rgba(0,0,0,0.02)' : 'transparent'
              }}
            >
              <ListItemText 
                primary={
                  <Typography variant="subtitle1" fontWeight="medium">
                    {forecast?.category || 'Unknown Category'}
                  </Typography>
                } 
                secondary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body2" component="span">
                      MAPE: <b>{forecast?.mape != null ? forecast.mape.toFixed(2) : 'N/A'}%</b>
                    </Typography>
                    <Typography variant="body2" component="span">
                      Growth: <b>{forecast?.growth_rate != null ? forecast.growth_rate.toFixed(2) : 'N/A'}%</b>
                    </Typography>
                  </Box>
                } 
              />
              {forecast?.growth_rate != null && (
                <Tooltip title={`${forecast.growth_rate > 0 ? 'Positive' : 'Negative'} growth trend`}>
                  <Box>
                    {forecast.growth_rate > 0 ? (
                      <TrendingUpIcon style={{ color: theme.palette.success.main }} />
                    ) : (
                      <TrendingDownIcon style={{ color: theme.palette.error.main }} />
                    )}
                  </Box>
                </Tooltip>
              )}
            </ListItem>
          ))}
        </List>
      </Box>
    </Paper>
  </Grid>
  
  {/* Inventory Recommendations */}
  <Grid item xs={12} md={6}>
    <Paper elevation={3} sx={{ 
      p: 4, 
      display: 'flex', 
      flexDirection: 'column', 
      height: 450,
      borderRadius: 2,
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
    }}>
      <Typography component="h2" variant="h6" gutterBottom sx={{ 
        color: theme.palette.secondary.main,
        fontWeight: 'bold',
        display: 'flex',
        alignItems: 'center',
        borderBottom: `1px solid ${theme.palette.divider}`,
        pb: 1
      }}>
        <InfoIcon sx={{ mr: 1 }} /> Inventory Recommendations
      </Typography>
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        <List sx={{ px: 1 }}>
          {(recommendations.inventory || []).slice(0, 8).map((rec, index) => (
            <ListItem 
              key={`rec-${index}`} 
              divider={index < (recommendations.inventory || []).slice(0, 8).length - 1}
              sx={{ 
                borderRadius: 1,
                mb: 1,
                p: 2,
                backgroundColor: index % 2 === 0 ? 'rgba(0,0,0,0.02)' : 'transparent'
              }}
            >
              <ListItemText 
                primary={
                  <Typography variant="subtitle1" fontWeight="medium">
                    {rec?.product_category || rec?.category || 'Unknown Category'}
                  </Typography>
                } 
                secondary={
                  <Box sx={{ mt: 0.5 }}>
                    <Typography variant="body2" component="span">
                      {rec?.recommendation ? (
                        rec.recommendation
                      ) : (rec?.reorder_point != null && rec?.safety_stock != null) ? (
                        <>
                          <Box component="span" sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                            <Box component="span" sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: theme.palette.warning.main, mr: 1 }} />
                            Reorder at: <b style={{ marginLeft: 4 }}>
                              {isFinite(parseFloat(rec.reorder_point)) ? parseFloat(rec.reorder_point).toFixed(0) : "N/A"} units
                            </b>
                          </Box>
                          <Box component="span" sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box component="span" sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: theme.palette.info.main, mr: 1 }} />
                            Safety stock: <b style={{ marginLeft: 4 }}>
                              {isFinite(parseFloat(rec.safety_stock)) ? parseFloat(rec.safety_stock).toFixed(0) : "N/A"} units
                            </b>
                          </Box>
                        </>
                      ) : (
                        'No recommendation available'
                      )}
                    </Typography>
                  </Box>
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