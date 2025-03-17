const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { parse } = require('csv-parse/sync');  // Robust CSV parser

const app = express();
const PORT = process.env.PORT || 5000;

// Determine output directory from environment variable or use default.
const OUTPUT_DIR = process.env.OUTPUT_DIR || path.join(__dirname, '../output');

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve data files from the output directory.
app.use('/data', express.static(OUTPUT_DIR));

// API endpoint to run the analytics pipeline
app.post('/api/run-analysis', (req, res) => {
  const { dataDir, topN, forecastPeriods, useSpark } = req.body;
  
  // Create command with parameters. Note: output directory is taken from OUTPUT_DIR.
  let command = `python main.py --data-dir=${dataDir || '.'} --output-dir=${OUTPUT_DIR} --top-n=${topN || 5} --forecast-periods=${forecastPeriods || 6}`;
  
  if (useSpark) {
    command += ' --use-spark';
  }
  
  console.log(`Running command: ${command}`);
  
  // Execute the Python script.
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
  fs.readdir(OUTPUT_DIR, (err, files) => {
    if (err) {
      console.error(`Error reading output directory: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
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
  const forecastFile = path.join(OUTPUT_DIR, 'forecast_report.csv');
  
  fs.readFile(forecastFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading forecast file: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
    try {
      // Use csv-parse to properly handle quoted fields.
      const records = parse(data, { columns: true, skip_empty_lines: true });
      const categoryForecast = records.find(f => f.category === category);
      
      if (!categoryForecast) {
        return res.status(404).json({ error: `Forecast for category ${category} not found` });
      }
      
      return res.json(categoryForecast);
    } catch (parseError) {
      console.error(`Error parsing CSV for forecasts: ${parseError}`);
      return res.status(500).json({ error: parseError.message });
    }
  });
});

// API endpoint to get seller details
app.get('/api/sellers/:sellerId', (req, res) => {
  const { sellerId } = req.params;
  const sellerFile = path.join(OUTPUT_DIR, 'seller_clusters.csv');
  
  fs.readFile(sellerFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading seller file: ${err}`);
      return res.status(500).json({ error: err.message });
    }
    
    try {
      const records = parse(data, { columns: true, skip_empty_lines: true });
      const sellerDetails = records.find(s => s.seller_id === sellerId);
      
      if (!sellerDetails) {
        return res.status(404).json({ error: `Seller ${sellerId} not found` });
      }
      
      return res.json(sellerDetails);
    } catch (parseError) {
      console.error(`Error parsing CSV for sellers: ${parseError}`);
      return res.status(500).json({ error: parseError.message });
    }
  });
});

// API endpoint to get all data for the dashboard
app.get('/api/dashboard-data', (req, res) => {
  const files = {
    monthlyDemand: path.join(OUTPUT_DIR, 'monthly_demand.csv'),
    forecastReport: path.join(OUTPUT_DIR, 'forecast_report.csv'),
    sellerClusters: path.join(OUTPUT_DIR, 'seller_clusters.csv'),
    reorderRecommendations: path.join(OUTPUT_DIR, 'reorder_recommendations.csv')
  };
  
  const stateMetricsPath = path.join(OUTPUT_DIR, 'state_metrics.csv');
  if (fs.existsSync(stateMetricsPath)) {
    files.stateMetrics = stateMetricsPath;
  }
  
  const missingFiles = Object.entries(files)
    .filter(([_, filePath]) => !fs.existsSync(filePath))
    .map(([key, _]) => key);
  
  if (missingFiles.length > 0) {
    return res.status(404).json({ 
      error: 'Missing required data files',
      missingFiles
    });
  }
  
  const filePromises = Object.entries(files).map(([key, filePath]) => {
    return new Promise((resolve, reject) => {
      fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
          console.error(`Error reading ${key} file: ${err}`);
          return reject(err);
        }
        
        try {
          const records = parse(data, { columns: true, skip_empty_lines: true });
          resolve({ key, data: records });
        } catch (parseError) {
          console.error(`Error parsing ${key} CSV: ${parseError}`);
          return reject(parseError);
        }
      });
    });
  });
  
  Promise.all(filePromises)
    .then(results => {
      const data = results.reduce((acc, { key, data }) => {
        acc[key] = data;
        return acc;
      }, {});
      
      // Helper functions defined inline for dashboard processing.
      function processDemandData(data) {
        return data.map(row => {
          if (!row.date) {
            const year = row.year || row.order_year;
            const month = row.month || row.order_month;
            row.date = new Date(year, month - 1, 1);
          } else if (typeof row.date === 'string') {
            row.date = new Date(row.date);
          }
          if (!row.count && row.order_count) {
            row.count = row.order_count;
          }
          return row;
        });
      }
      
      function getTopCategories(data, limit = 5) {
        const categoryTotals = {};
        data.forEach(row => {
          const category = row.product_category_name;
          const count = row.count || row.order_count || 0;
          if (category) {
            if (!categoryTotals[category]) {
              categoryTotals[category] = 0;
            }
            categoryTotals[category] += Number(count) || 0;
          }
        });
        return Object.entries(categoryTotals)
          .sort((a, b) => b[1] - a[1])
          .slice(0, limit)
          .map(entry => entry[0]);
      }
      
      function groupByCategory(data) {
        const grouped = {};
        data.forEach(row => {
          const category = row.product_category_name;
          if (category) {
            if (!grouped[category]) {
              grouped[category] = [];
            }
            grouped[category].push(row);
          }
        });
        return grouped;
      }
      
      function extractSellerMetrics(sellerClusters) {
        const clusterMetrics = {};
        sellerClusters.forEach(seller => {
          const cluster = seller.prediction;
          if (cluster === undefined || cluster === null) return;
          if (!clusterMetrics[cluster]) {
            clusterMetrics[cluster] = {
              count: 0,
              total_sales: 0,
              avg_processing_time: 0,
              avg_delivery_days: 0
            };
          }
          clusterMetrics[cluster].count += 1;
          clusterMetrics[cluster].total_sales += Number(seller.total_sales) || 0;
          clusterMetrics[cluster].avg_processing_time += Number(seller.avg_processing_time) || 0;
          clusterMetrics[cluster].avg_delivery_days += Number(seller.avg_delivery_days) || 0;
        });
        Object.keys(clusterMetrics).forEach(cluster => {
          const metrics = clusterMetrics[cluster];
          metrics.avg_sales = metrics.total_sales / metrics.count;
          metrics.avg_processing_time = metrics.avg_processing_time / metrics.count;
          metrics.avg_delivery_days = metrics.avg_delivery_days / metrics.count;
        });
        return { clusterMetrics, sellerCount: sellerClusters.length };
      }
      
      function calculateKPIs(demandData, sellerData, forecastData) {
        let totalProcessingTime = 0;
        let sellerCount = 0;
        sellerData.forEach(seller => {
          if (seller.avg_processing_time) {
            totalProcessingTime += Number(seller.avg_processing_time);
            sellerCount++;
          }
        });
        const avgProcessingTime = sellerCount > 0 ? totalProcessingTime / sellerCount : 0;
        let totalGrowthRate = 0;
        let forecastCount = 0;
        forecastData.forEach(forecast => {
          if (forecast.growth_rate) {
            totalGrowthRate += Number(forecast.growth_rate);
            forecastCount++;
          }
        });
        const avgGrowthRate = forecastCount > 0 ? totalGrowthRate / forecastCount : 0;
        let totalDemand = 0;
        demandData.forEach(row => {
          totalDemand += Number(row.count || row.order_count || 0);
        });
        // Use default on-time delivery if not available.
        let onTimeDelivery = 85.0;
        try {
          const metricsPath = path.join(__dirname, '../output', 'order_performance_metrics.csv');
          if (fs.existsSync(metricsPath)) {
            const content = fs.readFileSync(metricsPath, 'utf8');
            const lines = content.trim().split('\n');
            if (lines.length > 1) {
              const headers = lines[0].split(',');
              const values = lines[1].split(',');
              const metricsObj = {};
              headers.forEach((header, i) => {
                metricsObj[header] = values[i];
              });
              onTimeDelivery = Number(metricsObj.on_time_delivery_rate) || 85.0;
            }
          }
        } catch (error) {
          console.warn('Error loading on-time delivery metrics:', error);
        }
        return {
          avg_processing_time: avgProcessingTime,
          forecast_growth: avgGrowthRate,
          total_demand: totalDemand,
          on_time_delivery: onTimeDelivery,
          perfect_order_rate: onTimeDelivery * 0.9,
          inventory_turnover: 8.0,
          estimated_fields: ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
        };
      }
      
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
      
      if (data.stateMetrics) {
        dashboardData.geography = { stateMetrics: data.stateMetrics };
      }
      
      return res.json(dashboardData);
    })
    .catch(error => {
      console.error('Error processing dashboard data:', error);
      return res.status(500).json({ error: error.message });
    });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
