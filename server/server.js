require('dotenv').config();
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
const { MongoClient, ObjectId } = require('mongodb');

// MongoDB connection string from environment variable or use default
const MONGODB_URI = process.env.MONGODB_URI
const DB_NAME = process.env.DB_NAME || 'supply_chain_analytics';

// MongoDB connection
let mongoClient;
let db;

async function connectToMongoDB() {
  try {
    mongoClient = new MongoClient(MONGODB_URI);
    await mongoClient.connect();
    db = mongoClient.db(DB_NAME);
    console.log(`Connected to MongoDB: ${DB_NAME}`);
    return true;
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    return false;
  }
}

// Initialize MongoDB connection on server start
let mongoAvailable = false;
connectToMongoDB()
  .then(result => {
    mongoAvailable = result;
  })
  .catch(err => {
    console.error('Failed to initialize MongoDB connection:', err);
    mongoAvailable = false;
  });

  
// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve data files from the output directory.
app.use('/data', express.static(OUTPUT_DIR));

// API endpoint to run the analytics pipeline
app.post('/api/run-analysis', (req, res) => {
  const { dataDir, topN, forecastPeriods, useSpark } = req.body;
  
  // Using path.join for reliable path resolution across environments
  let command = `python ${path.join(__dirname, '..', 'backend', 'main.py')} --data-dir=${dataDir || '.'} --output-dir=${OUTPUT_DIR} --top-n=${topN || 5} --forecast-periods=${forecastPeriods || 6}`;
  
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
          return reject({ key, error: err.message, type: 'file_read_error' });
        }
        
        try {
          const records = parse(data, { columns: true, skip_empty_lines: true });
          
          // Check if we have valid data (at least some rows)
          if (!records || records.length === 0) {
            console.warn(`Warning: ${key} file contains no data records`);
          }
          
          resolve({ key, data: records, status: 'success' });
        } catch (parseError) {
          console.error(`Error parsing ${key} CSV: ${parseError}`);
          
          // Determine if this is a critical file
          const isCritical = ['monthlyDemand', 'forecastReport'].includes(key);
          
          if (isCritical) {
            // For critical files, reject with error details
            reject({ 
              key, 
              error: `Failed to parse critical file: ${parseError.message}`, 
              type: 'parse_error_critical' 
            });
          } else {
            // For non-critical files, resolve with error status but empty data
            resolve({ 
              key, 
              data: [], 
              status: 'parse_error',
              error: parseError.message
            });
          }
        }
      });
    });
  });
  
  Promise.all(filePromises)
    .then(results => {
      // Process results and track any parsing errors
      const data = {};
      const dataWarnings = [];
      
      results.forEach(result => {
        data[result.key] = result.data;
        
        // If there was a parse error for non-critical files, track it
        if (result.status === 'parse_error') {
          dataWarnings.push({
            dataType: result.key,
            message: result.error
          });
        }
      });
      
      // Helper functions defined inline for dashboard processing.
      function processDemandData(data) {
        return data.map(row => {
          if (!row) return null;
          // Normalize date: use row.date if valid; otherwise try year/month fields.
          if (!row.date) {
            const year = row.order_year || row.year;
            const month = row.order_month || row.month;
            if (year && month) {
              const parsedYear = parseInt(year);
              const parsedMonth = parseInt(month);
              if (!isNaN(parsedYear) && !isNaN(parsedMonth)) {
                row.date = new Date(parsedYear, parsedMonth - 1, 1);
              } else {
                console.warn(`Invalid year/month in row: ${JSON.stringify(row)}. Using current date as fallback.`);
                row.date = new Date();
              }
            } else {
              console.warn(`Missing date/year/month in row: ${JSON.stringify(row)}. Using current date as fallback.`);
              row.date = new Date();
            }
          } else if (typeof row.date === 'string') {
            const parsedDate = new Date(row.date);
            if (isNaN(parsedDate.getTime())) {
              console.warn(`Invalid date string: ${row.date}. Using current date as fallback.`);
              row.date = new Date();
            } else {
              row.date = parsedDate;
            }
          }
          // Ensure count exists; fallback to order_count or 0.
          if (!row.count && row.order_count) {
            row.count = row.order_count;
          }
          row.count = Number(row.count) || 0;
          return row;
        }).filter(row => row !== null);
      }
      
      function getTopCategories(data, limit = 5) {
        const categoryTotals = {};
        data.forEach(row => {
          if (!row) return;
          const category = row.product_category_name || row.category;
          const count = Number(row.count || row.order_count || 0);
          if (category) {
            if (!categoryTotals[category]) {
              categoryTotals[category] = 0;
            }
            categoryTotals[category] += count;
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
          if (!row) return;
          const category = row.product_category_name || row.category;
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
          if (!seller) return;
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
          metrics.avg_sales = metrics.count > 0 ? metrics.total_sales / metrics.count : 0;
          metrics.avg_processing_time = metrics.count > 0 ? metrics.avg_processing_time / metrics.count : 0;
          metrics.avg_delivery_days = metrics.count > 0 ? metrics.avg_delivery_days / metrics.count : 0;
        });
        return { clusterMetrics, sellerCount: sellerClusters.length };
      }
      
      function calculateKPIs(demandData, sellerData, forecastData) {
        let totalProcessingTime = 0;
        let sellerCount = 0;
        sellerData = Array.isArray(sellerData) ? sellerData : [];
        forecastData = Array.isArray(forecastData) ? forecastData : [];
        demandData = Array.isArray(demandData) ? demandData : [];
        sellerData.forEach(seller => {
          if (seller && seller.avg_processing_time) {
            totalProcessingTime += Number(seller.avg_processing_time) || 0;
            sellerCount++;
          }
        });
        const avgProcessingTime = sellerCount > 0 ? totalProcessingTime / sellerCount : 0;
        let totalGrowthRate = 0;
        let forecastCount = 0;
        forecastData.forEach(forecast => {
          if (forecast && forecast.growth_rate !== undefined && forecast.growth_rate !== null) {
            totalGrowthRate += Number(forecast.growth_rate) || 0;
            forecastCount++;
          }
        });
        const avgGrowthRate = forecastCount > 0 ? totalGrowthRate / forecastCount : 0;
        let totalDemand = 0;
        demandData.forEach(row => {
          totalDemand += Number(row.count || row.order_count || 0);
        });
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
          performanceMetrics: Array.isArray(data.forecastReport) ? data.forecastReport.map(row => ({
            category: row.category,
            mape: row.mape,
            rmse: row.rmse,
            mae: row.mae,
            growth_rate: row.growth_rate
          })) : []
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
      
      // If there were any data warnings, include them in the response
      if (dataWarnings.length > 0) {
        dashboardData.dataWarnings = dataWarnings;
      }
      
      return res.json(dashboardData);
    })
    .catch(error => {
      console.error('Error processing dashboard data:', error);
      
      // Provide a meaningful error to the client
      if (error.type && error.key) {
        return res.status(500).json({
          error: `Data processing failed: ${error.error}`,
          failedComponent: error.key,
          errorType: error.type
        });
      } else {
        return res.status(500).json({ error: 'Failed to process dashboard data' });
      }
    });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});