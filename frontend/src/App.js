import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  AppBar, Toolbar, Typography, Container, Box, Drawer, List, ListItem, ListItemIcon, ListItemText,
  CssBaseline, IconButton, Divider, Alert, Button 
} from '@mui/material';
import { 
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Category as CategoryIcon,
  ShoppingCart as OrderIcon,
  People as SellerIcon,
  Map as MapIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';

// Import components
import DashboardPage from './pages/DashboardPage';
import DemandForecastPage from './pages/DemandForecastPage';
import ProductCategoriesPage from './pages/ProductCategoriesPage';
import SellerPerformancePage from './pages/SellerPerformancePage';
import GeographicalAnalysisPage from './pages/GeographicalAnalysisPage';
import RecommendationsPage from './pages/RecommendationsPage';
// Import services
import { loadDashboardData } from './services/dataService';

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dataWarnings, setDataWarnings] = useState([]);

  // Define a fetchData function that can be reused (e.g., for retrying)
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await loadDashboardData();

      // Check for data warnings received from the API
      if (data.dataWarnings && data.dataWarnings.length > 0) {
        setDataWarnings(data.dataWarnings);
        console.warn("Data quality warnings:", data.dataWarnings);
      }
      
      setDashboardData(data);
      setLoading(false);
    } catch (err) {
      console.error("Error loading dashboard data:", err);
      const errorDetail = err?.message ? `Error: ${err.message}.` : "";
      const troubleshootingTip = "Please check your network connection or try refreshing the page.";
      setError(`Failed to load dashboard data. ${errorDetail} ${troubleshootingTip}`);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const drawerItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Demand Forecast', icon: <TimelineIcon />, path: '/forecast' },
    { text: 'Product Categories', icon: <CategoryIcon />, path: '/categories' },
    { text: 'Seller Performance', icon: <SellerIcon />, path: '/sellers' },
    { text: 'Geographical Analysis', icon: <MapIcon />, path: '/geography' },
    { text: 'Recommendations', icon: <OrderIcon />, path: '/recommendations' },
  ];

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" component="div">
          Supply Chain Analytics
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {drawerItems.map((item) => (
          <ListItem button key={item.text} component={Link} to={item.path} onClick={toggleDrawer}>
            <ListItemIcon>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </div>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography variant="h5">Loading dashboard data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh', p: 3 }}>
        <Typography variant="h5" color="error" gutterBottom>Error</Typography>
        <Typography sx={{ mb: 2 }}>{error}</Typography>
        <Button variant="contained" color="primary" onClick={fetchData}>
          Retry
        </Button>
      </Box>
    );
  }

  // Create safe empty objects for each data section to prevent null reference errors
  const safeData = dashboardData || {};
  const safeForecasts = safeData.forecasts || {};
  const safeCategories = safeData.categories || { topCategories: [], categoryData: {} };
  const safeSellerPerformance = safeData.sellerPerformance || { clusters: [] };
  const safeGeography = safeData.geography || { stateMetrics: [] };
  const safeRecommendations = safeData.recommendations || { inventory: [] };

  return (
    <Router>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div">
              Supply Chain Analytics Dashboard
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={toggleDrawer}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
            },
          }}
        >
          {drawer}
        </Drawer>
        
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Toolbar /> {/* This adds spacing below the AppBar */}
          
          {/* Display data warnings if any */}
          {dataWarnings.length > 0 && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              Some data quality issues were detected. Some visualizations may be incomplete.
            </Alert>
          )}
          
          <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Routes>
              <Route path="/" element={<DashboardPage data={safeData} />} />
              <Route path="/forecast" element={<DemandForecastPage data={safeForecasts} />} />
              <Route path="/categories" element={<ProductCategoriesPage data={safeCategories} />} />
              <Route path="/sellers" element={<SellerPerformancePage data={safeSellerPerformance} />} />
              <Route path="/geography" element={<GeographicalAnalysisPage data={safeGeography} />} />
              <Route path="/recommendations" element={<RecommendationsPage data={safeRecommendations} />} />
            </Routes>
          </Container>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
