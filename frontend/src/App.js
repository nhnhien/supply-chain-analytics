import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  AppBar, Toolbar, Typography, Container, Grid, Paper, 
  Box, Drawer, List, ListItem, ListItemIcon, ListItemText,
  CssBaseline, IconButton, Divider
} from '@mui/material';
import { 
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Category as CategoryIcon,
  ShoppingCart as OrderIcon,
  People as SellerIcon,
  Map as MapIcon,
  Menu as MenuIcon
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
  
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const data = await loadDashboardData();
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        console.error("Error loading dashboard data:", err);
        setError("Failed to load dashboard data. Please try again later.");
        setLoading(false);
      }
    }
    
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
    return <div>Loading dashboard data...</div>;
  }
  
  if (error) {
    return <div>Error: {error}</div>;
  }
  
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
          <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Routes>
              <Route path="/" element={<DashboardPage data={dashboardData} />} />
              <Route path="/forecast" element={<DemandForecastPage data={dashboardData?.forecasts} />} />
              <Route path="/categories" element={<ProductCategoriesPage data={dashboardData?.categories} />} />
              <Route path="/sellers" element={<SellerPerformancePage data={dashboardData?.sellerPerformance} />} />
              <Route path="/geography" element={<GeographicalAnalysisPage data={dashboardData?.geography} />} />
              <Route path="/recommendations" element={<RecommendationsPage data={dashboardData?.recommendations} />} />
            </Routes>
          </Container>
        </Box>
      </Box>
    </Router>
  );
}

export default App;