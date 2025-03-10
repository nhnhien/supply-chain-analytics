import React from 'react';
import { Box, Card, CardContent, Typography, Avatar } from '@mui/material';

/**
 * KPI Card Component that displays a key performance indicator with an icon
 * 
 * @param {Object} props Component props
 * @param {string} props.title Title of the KPI
 * @param {string} props.value Value to display
 * @param {React.ReactNode} props.icon Icon to display
 * @param {string} props.color Background color for the icon
 * @param {string} props.trend Trend direction: 'up', 'down', or 'flat'
 */
const KPICard = ({ title, value, icon, color = "#1976d2", trend = null }) => {
  return (
    <Card elevation={2} sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h5" component="div">
              {value}
            </Typography>
            {trend && (
              <Typography 
                variant="caption" 
                color={trend === 'up' ? 'success.main' : trend === 'down' ? 'error.main' : 'text.secondary'}
              >
                {trend === 'up' ? '↑ ' : trend === 'down' ? '↓ ' : ''}
                {trend !== 'flat' && 'vs. last period'}
              </Typography>
            )}
          </Box>
          <Avatar 
            sx={{ 
              bgcolor: color, 
              width: 48, 
              height: 48,
              boxShadow: 1
            }}
          >
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );
};

export default KPICard;