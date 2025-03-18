// In KPICard.js
import React from 'react';
import { Box, Card, CardContent, Typography, Avatar, Tooltip } from '@mui/material';
import { Info as InfoIcon } from '@mui/icons-material';

const KPICard = ({ title, value, icon, color = "#1976d2", trend = null, isEstimated = false }) => {
  return (
    <Card elevation={2} sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
              {isEstimated && (
                <Tooltip title="Value is estimated based on available data">
                  <InfoIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle', color: 'warning.main' }} />
                </Tooltip>
              )}
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