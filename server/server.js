const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve data files from the output directory
app.use('/data', express.static(path.join(__dirname, 'output')));

// API endpoint to run the analytics pipeline
app.post('/api/run-analysis', (req, res) => {
  const { dataDir, topN, forecastPeriods, useSpark } = req.body;
  
  // Create command with parameters
  let command = `python main.py --data-dir=${dataDir || '.'} --output-dir=./output --top-n=${topN || 5} --forecast-periods=${forecastPeriods || 6}`;
  
  if (useSpark) {
    command += ' --use-spark';
  }
  
  console.log(`Running command: ${command}`);
  
  // Execute the Python script
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing command: ${error}`);
      return res.status(500).json({ error: error.message });
    }
    
    console.log(`Analysis completed: ${stdout}`);
    
    if (stderr) {
      console.error(`Analysis stderr: ${stderr}`);
    }
    
    return res.json({ success: true, message: 'Analysis completed successfully' });
  });
});

// API endpoint to get list of available data files
app.get('/api/data-files', (req, res) => {
  const outputDir = path.join(__dirname, 'output');
  
  fs.readdir(outputDir, (err, files) => {
    if (err) {
      console.error(`Error reading output directory: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
    // Filter and classify files
    const csvFiles = files.filter(file => file.endsWith('.csv'));
    const imageFiles = files.filter(file => file.endsWith('.png'));
    const reportFiles = files.filter(file => file.endsWith('.md'));
    
    return res.json({
      csvFiles,
      imageFiles,
      reportFiles,
      allFiles: files
    });
  });
});

// API endpoint to get forecast for a specific category
app.get('/api/forecasts/:category', (req, res) => {
  const { category } = req.params;
  const forecastFile = path.join(__dirname, 'output', 'forecast_report.csv');
  
  fs.readFile(forecastFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading forecast file: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
    // Parse CSV
    const lines = data.trim().split('\n');
    const headers = lines[0].split(',');
    
    const forecasts = lines.slice(1).map(line => {
      const values = line.split(',');
      const forecast = {};
      
      headers.forEach((header, index) => {
        forecast[header] = values[index];
      });
      
      return forecast;
    });
    
    // Find the requested category
    const categoryForecast = forecasts.find(f => f.category === category);
    
    if (!categoryForecast) {
      return res.status(404).json({ error: `Forecast for category ${category} not found` });
    }
    
    return res.json(categoryForecast);
  });
});

// API endpoint to get seller details
app.get('/api/sellers/:sellerId', (req, res) => {
  const { sellerId } = req.params;
  const sellerFile = path.join(__dirname, 'output', 'seller_clusters.csv');
  
  fs.readFile(sellerFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading seller file: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
    // Parse CSV
    const lines = data.trim().split('\n');
    const headers = lines[0].split(',');
    
    const sellers = lines.slice(1).map(line => {
      const values = line.split(',');
      const seller = {};
      
      headers.forEach((header, index) => {
        seller[header] = values[index];
      });
      
      return seller;
    });
    
    // Find the requested seller
    const sellerDetails = sellers.find(s => s.seller_id === sellerId);
    
    if (!sellerDetails) {
      return res.status(404).json({ error: `Seller ${sellerId} not found` });
    }
    
    return res.json(sellerDetails);
  });
});

// API endpoint to get all data for the dashboard
app.get('/api/dashboard-data', (req, res) => {
  const outputDir = path.join(__dirname, 'output');
  
  // Define files to load
  const files = {
    monthlyDemand: path.join(outputDir, 'monthly_demand.csv'),
    forecastReport: path.join(outputDir, 'forecast_report.csv'),
    sellerClusters: path.join(outputDir, 'seller_clusters.csv'),
    reorderRecommendations: path.join(outputDir, 'reorder_recommendations.csv')
  };
  
  // Try to load state metrics if available
  const stateMetricsPath = path.join(outputDir, 'state_metrics.csv');
  if (fs.existsSync(stateMetricsPath)) {
    files.stateMetrics = stateMetricsPath;
  }
  
  // Check if all required files exist
  const missingFiles = Object.entries(files)
    .filter(([_, filePath]) => !fs.existsSync(filePath))
    .map(([key, _]) => key);
  
  if (missingFiles.length > 0) {
    return res.status(404).json({ 
      error: 'Missing required data files',
      missingFiles
    });
  }
  
  // Load all files
  const filePromises = Object.entries(files).map(([key, filePath]) => {
    return new Promise((resolve, reject) => {
      fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
          console.error(`Error reading ${key} file: ${err}`);
          return reject(err);
        }
        
        // Parse CSV
        const lines = data.trim().split('\n');
        const headers = lines[0].split(',');
        
        const parsedData = lines.slice(1).map(line => {
          const values = line.split(',');
          const row = {};
          
          headers.forEach((header, index) => {
            // Try to convert numeric values
            const value = values[index];
            if (value && !isNaN(value) && value !== '') {
              row[header] = Number(value);
            } else {
              row[header] = value;
            }
          });
          
          return row;
        });
        
        resolve({ key, data: parsedData });
      });
    });
  });
  
  // Process all files and send response
  Promise.all(filePromises)
    .then(results => {
      // Convert array of results to an object
      const data = results.reduce((acc, { key, data }) => {
        acc[key] = data;
        return acc;
      }, {});
      
      // Process the data for the dashboard
      const dashboardData = {
        demandData: processDemandData(data.monthlyDemand),
        categories: {
          topCategories: getTopCategories(data.monthlyDemand),
          categoryData: groupByCategory(data.monthlyDemand)
        },
        forecasts: {
          forecastReport: data.forecastReport,
          performanceMetrics: data.forecastReport.map(row => ({
            category: row.category,
            mape: row.mape,
            rmse: row.rmse,
            mae: row.mae,
            growth_rate: row.growth_rate
          }))
        },
        sellerPerformance: {
          clusters: data.sellerClusters,
          metrics: extractSellerMetrics(data.sellerClusters)
        },
        recommendations: {
          inventory: data.reorderRecommendations
        },
        kpis: calculateKPIs(data.monthlyDemand, data.sellerClusters, data.forecastReport)
      };
      
      // Add state metrics if available
      if (data.stateMetrics) {
        dashboardData.geography = {
          stateMetrics: data.stateMetrics
        };
      }
      
      return res.json(dashboardData);
    })
    .catch(error => {
      console.error('Error processing dashboard data:', error);
      return res.status(500).json({ error: error.message });
    });
});

// Helper functions

/**
 * Process demand data for time series visualization
 * @param {Array} data - Raw monthly demand data
 * @returns {Array} - Processed demand data
 */
function processDemandData(data) {
  // Ensure date field exists
  return data.map(row => {
    // Create date from year and month if not already present
    if (!row.date) {
      const year = row.year || row.order_year;
      const month = row.month || row.order_month;
      row.date = new Date(year, month - 1, 1);
    } else if (typeof row.date === 'string') {
      row.date = new Date(row.date);
    }
    
    // Ensure count field exists
    if (!row.count && row.order_count) {
      row.count = row.order_count;
    }
    
    return row;
  });
}

/**
 * Get top categories by total demand
 * @param {Array} data - Monthly demand data
 * @param {number} limit - Maximum number of categories to return
 * @returns {Array} - Array of top category names
 */
function getTopCategories(data, limit = 5) {
  // Group by category and sum the demand
  const categoryTotals = {};
  
  data.forEach(row => {
    const category = row.product_category_name;
    const count = row.count || row.order_count || 0;
    
    if (category && !categoryTotals[category]) {
      categoryTotals[category] = 0;
    }
    
    if (category) {
      categoryTotals[category] += count;
    }
  });
  
  // Sort categories by total demand and get top N
  return Object.entries(categoryTotals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(entry => entry[0]);
}

/**
 * Group demand data by category
 * @param {Array} data - Monthly demand data
 * @returns {Object} - Object with category names as keys and arrays of data as values
 */
function groupByCategory(data) {
  const grouped = {};
  
  data.forEach(row => {
    const category = row.product_category_name;
    
    if (category && !grouped[category]) {
      grouped[category] = [];
    }
    
    if (category) {
      grouped[category].push(row);
    }
  });
  
  return grouped;
}

/**
 * Extract seller performance metrics
 * @param {Array} sellerClusters - Seller cluster data
 * @returns {Object} - Object with seller metrics
 */
function extractSellerMetrics(sellerClusters) {
  // Calculate average metrics by cluster
  const clusterMetrics = {};
  
  sellerClusters.forEach(seller => {
    const cluster = seller.prediction;
    
    if (!clusterMetrics[cluster]) {
      clusterMetrics[cluster] = {
        count: 0,
        total_sales: 0,
        avg_processing_time: 0,
        avg_delivery_days: 0
      };
    }
    
    clusterMetrics[cluster].count += 1;
    clusterMetrics[cluster].total_sales += seller.total_sales || 0;
    clusterMetrics[cluster].avg_processing_time += seller.avg_processing_time || 0;
    clusterMetrics[cluster].avg_delivery_days += seller.avg_delivery_days || 0;
  });
  
  // Calculate averages
  Object.keys(clusterMetrics).forEach(cluster => {
    const metrics = clusterMetrics[cluster];
    metrics.avg_sales = metrics.total_sales / metrics.count;
    metrics.avg_processing_time = metrics.avg_processing_time / metrics.count;
    metrics.avg_delivery_days = metrics.avg_delivery_days / metrics.count;
  });
  
  return {
    clusterMetrics: clusterMetrics,
    sellerCount: sellerClusters.length
  };
}

/**
 * Calculate key performance indicators
 * @param {Array} demandData - Monthly demand data
 * @param {Array} sellerData - Seller performance data
 * @param {Array} forecastData - Forecast data
 * @returns {Object} - Object with calculated KPIs
 */
function calculateKPIs(demandData, sellerData, forecastData) {
    // Average processing time
    let totalProcessingTime = 0;
    let sellerCount = 0;
    
    sellerData.forEach(seller => {
      if (seller.avg_processing_time) {
        totalProcessingTime += seller.avg_processing_time;
        sellerCount++;
      }
    });
    
    const avgProcessingTime = sellerCount > 0 ? totalProcessingTime / sellerCount : 0;
    
    // Average forecast growth rate
    let totalGrowthRate = 0;
    let forecastCount = 0;
    
    forecastData.forEach(forecast => {
      if (forecast.growth_rate) {
        totalGrowthRate += forecast.growth_rate;
        forecastCount++;
      }
    });
    
    const avgGrowthRate = forecastCount > 0 ? totalGrowthRate / forecastCount : 0;
    
    // Total demand
    let totalDemand = 0;
    
    demandData.forEach(row => {
      totalDemand += row.count || row.order_count || 0;
    });
    
    // On-time delivery rate calculation
    let onTimeDeliveryRate;
    try {
      // Try to load on-time delivery metrics from file
      const orderMetricsPath = path.join(__dirname, 'output', 'order_performance_metrics.csv');
      if (fs.existsSync(orderMetricsPath)) {
        const metrics = fs.readFileSync(orderMetricsPath, 'utf8');
        const lines = metrics.trim().split('\n');
        if (lines.length > 1) {
          const headers = lines[0].split(',');
          const values = lines[1].split(',');
          const metricsObj = {};
          headers.forEach((header, i) => {
            metricsObj[header] = values[i];
          });
          
          onTimeDeliveryRate = parseFloat(metricsObj.on_time_delivery_rate) || 0;
        }
      } 
    } catch (error) {
      console.warn('Error loading on-time delivery metrics:', error);
    }
    
    // If we couldn't load from file, estimate based on order data
    if (onTimeDeliveryRate === undefined) {
      // Look for delivery data in the order files
      try {
        const ordersPath = path.join(__dirname, 'output', 'order_delivery_analysis.csv');
        if (fs.existsSync(ordersPath)) {
          // Parse order delivery analysis if available
          // This would be generated by a separate analysis script
          // [parsing logic here]
        } else {
          // Use an estimated value based on industry benchmarks
          // The e-commerce industry average on-time delivery is typically 85-90%
          onTimeDeliveryRate = 85.0;
          console.warn('Using industry benchmark for on-time delivery rate (85%)');
        }
      } catch (error) {
        console.warn('Error estimating on-time delivery rate:', error);
        onTimeDeliveryRate = 85.0;
      }
    }
    
    // Perfect order rate calculation
    // Perfect order = delivered on time, complete, undamaged, with correct documentation
    let perfectOrderRate;
    try {
      // Similar logic as above, try to load from file first
      const qualityMetricsPath = path.join(__dirname, 'output', 'quality_metrics.csv');
      // [loading logic here]
      
      // If no data, estimate as a percentage of on-time deliveries
      // Industry standard is that perfect order rate is typically 80-90% of on-time delivery rate
      if (perfectOrderRate === undefined) {
        perfectOrderRate = onTimeDeliveryRate * 0.9;
        console.warn('Estimating perfect order rate as 90% of on-time delivery rate');
      }
    } catch (error) {
      console.warn('Error calculating perfect order rate:', error);
      perfectOrderRate = onTimeDeliveryRate * 0.9;
    }
    
    // Inventory turnover calculation
    let inventoryTurnover;
    try {
      // Try to calculate from COGS and average inventory if available
      // [calculation logic here]
      
      // If not available, use industry benchmarks based on product categories
      if (inventoryTurnover === undefined) {
        // E-commerce inventory turnover rates typically range from 4-12 depending on product type
        inventoryTurnover = 8.0;
        console.warn('Using industry benchmark for inventory turnover (8.0)');
      }
    } catch (error) {
      console.warn('Error calculating inventory turnover:', error);
      inventoryTurnover = 8.0;
    }
    
    return {
      avg_processing_time: avgProcessingTime,
      forecast_growth: avgGrowthRate,
      total_demand: totalDemand,
      on_time_delivery: onTimeDeliveryRate,
      perfect_order_rate: perfectOrderRate,
      inventory_turnover: inventoryTurnover,
      // Flag estimated values for transparency in the UI
      estimated_fields: ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
    };
  }

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});