const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');

/**
 * GET /api/supply-chain-kpis
 * Returns calculated supply chain KPIs based on data analysis
 */
router.get('/supply-chain-kpis', (req, res) => {
  try {
    // Try to load KPIs from pre-calculated data files
    const outputDir = path.join(__dirname, '..', 'output');
    const kpiData = {};
    const estimatedFields = [];
    
    // Check for performance metrics file
    const perfMetricsPath = path.join(outputDir, 'performance_metrics.csv');
    if (fs.existsSync(perfMetricsPath)) {
      const perfData = fs.readFileSync(perfMetricsPath, 'utf8');
      const lines = perfData.trim().split('\n');
      if (lines.length > 1) {
        const headers = lines[0].split(',');
        const values = lines[1].split(',');
        
        headers.forEach((header, i) => {
          const value = parseFloat(values[i]);
          if (!isNaN(value)) {
            kpiData[header] = value;
          }
        });
      }
    } else {
      console.warn('Performance metrics file not found');
    }
    
    // Calculate on-time delivery from delivery data if available
    if (!kpiData.on_time_delivery) {
      const deliveryPath = path.join(outputDir, 'delivery_performance.csv');
      if (fs.existsSync(deliveryPath)) {
        // Process delivery data
        const deliveryData = fs.readFileSync(deliveryPath, 'utf8');
        // ... processing logic ...
      } else {
        // Use a conservative industry benchmark
        kpiData.on_time_delivery = 85.0;
        estimatedFields.push('on_time_delivery');
        console.warn('Using estimated on-time delivery metric');
      }
    }
    
    // Calculate perfect order rate if not available
    if (!kpiData.perfect_order_rate) {
      if (kpiData.on_time_delivery) {
        // Perfect order rate is typically lower than on-time delivery rate
        kpiData.perfect_order_rate = kpiData.on_time_delivery * 0.9;
      } else {
        kpiData.perfect_order_rate = 80.0;
      }
      estimatedFields.push('perfect_order_rate');
      console.warn('Using estimated perfect order rate metric');
    }
    
    // Calculate inventory turnover if not available
    if (!kpiData.inventory_turnover) {
      // Use industry average if not available
      kpiData.inventory_turnover = 8.0;
      estimatedFields.push('inventory_turnover');
      console.warn('Using estimated inventory turnover metric');
    }
    
    // Return KPIs with estimation flags
    return res.json({
      ...kpiData,
      estimated_fields: estimatedFields
    });
  } catch (error) {
    console.error('Error calculating KPIs:', error);
    return res.status(500).json({ 
      error: 'Failed to calculate KPIs',
      message: error.message 
    });
  }
});

module.exports = router;