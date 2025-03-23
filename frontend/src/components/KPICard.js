// In KPICard.js
import React from 'react';
import { Box, Card, CardContent, Typography, Avatar, Tooltip, useTheme } from '@mui/material';
import { Info as InfoIcon } from '@mui/icons-material';

const KPICard = ({ title, value, icon, color = "#1976d2", trend = null, isEstimated = false, sx }) => {
  const theme = useTheme();
  
  // Determine if the value is negative (for Forecast Growth)
  const isNegative = value && typeof value === 'string' && value.indexOf('-') === 0;
  
  return (
    <Card 
      elevation={2} 
      sx={{ 
        height: '100%',
        borderRadius: 2,
        position: 'relative',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          boxShadow: theme.shadows[4],
        },
        ...sx
      }}
    >
      <CardContent sx={{ p: 2 }}>
        {/* Title and indicator section */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mb: 1.5
        }}>
          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ 
              fontWeight: 'medium',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            {title}
            {isEstimated && (
              <Tooltip title="Value is estimated based on available data">
                <InfoIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle', color: 'warning.main' }} />
              </Tooltip>
            )}
          </Typography>
          
          <Avatar
            sx={{
              bgcolor: color,
              width: 36,
              height: 36,
              boxShadow: theme.shadows[1],
            }}
          >
            {icon}
          </Avatar>
        </Box>
        
        {/* Value section */}
        <Typography 
          variant="h4" 
          component="div" 
          sx={{ 
            fontWeight: 'bold', 
            mb: 0.5,
            color: isNegative ? theme.palette.error.main : 'inherit',
            display: 'flex',
            alignItems: 'center',
            fontSize: { xs: '1.6rem', md: '1.8rem' }
          }}
        >
          {value}
          
          {trend && (
            <Box 
              component="span"
              sx={{ 
                display: 'inline-flex', 
                alignItems: 'center', 
                color: trend === 'up' ? theme.palette.success.main : 
                       trend === 'down' ? theme.palette.error.main : 
                       theme.palette.text.secondary,
                ml: 1,
                fontSize: '1rem',
                fontWeight: 'medium'
              }}
            >
              {trend === 'up' ? '↑' : trend === 'down' ? '↓' : ''}
            </Box>
          )}
        </Typography>
        
        {/* Subtitle section (if applicable) */}
        {title === "Forecast Growth" && (
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ 
              display: 'block',
              opacity: 0.8,
              fontStyle: 'italic'
            }}
          >
            Projected future trend
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default KPICard;