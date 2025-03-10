import axios from 'axios';
import Papa from 'papaparse';

// Base URL for API endpoints
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

/**
 * Load CSV data from a file
 * @param {string} filePath - Path to the CSV file
 * @returns {Promise<Array>} - Parsed CSV data as an array of objects
 */
async function loadCsvData(filePath) {
  try {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    return new Promise((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          resolve(results.data);
        },
        error: (error) => {
          reject(error);
        }
      });
    });
  } catch (error) {
    console.error(`Error loading CSV from ${filePath}:`, error);
    throw error;
  }
}

/**
 * Load dashboard data from multiple CSV files
 * @returns {Promise<Object>} - Object containing all dashboard data
 */
export async function loadDashboardData() {
  try {
    // Load all required data files
    const [
      monthlyDemand,
      forecastReport,
      sellerClusters,
      reorderRecommendations,
      stateMetrics
    ] = await Promise.all([
      loadCsvData('/data/monthly_demand.csv'),
      loadCsvData('/data/forecast_report.csv'),
      loadCsvData('/data/seller_clusters.csv'),
      loadCsvData('/data/reorder_recommendations.csv'),
      loadCsvData('/data/state_metrics.csv')
    ]);
    
    // Process monthly demand data for time series
    const processedDemand = processDemandData(monthlyDemand);
    
    // Get top categories from the demand data
    const topCategories = getTopCategories(monthlyDemand);
    
    return {
      demandData: processedDemand,
      categories: {
        topCategories: topCategories,
        categoryData: groupByCategory(monthlyDemand)
      },
      forecasts: {
        forecastReport: forecastReport,
        performanceMetrics: extractForecastPerformance(forecastReport)
      },
      sellerPerformance: {
        clusters: sellerClusters,
        metrics: extractSellerMetrics(sellerClusters)
      },
      geography: {
        stateMetrics: stateMetrics
      },
      recommendations: {
        inventory: reorderRecommendations
      },
      kpis: calculateKPIs(monthlyDemand, sellerClusters, forecastReport)
    };
  } catch (error) {
    console.error("Error loading dashboard data:", error);
    throw error;
  }
}

/**
 * Process demand data for time series visualization
 * @param {Array} data - Raw monthly demand data
 * @returns {Object} - Processed demand data
 */
function processDemandData(data) {
  // Ensure date field exists
  const processedData = data.map(row => {
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
  
  return processedData;
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
    const count = row.count || row.order_count;
    
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
 * Extract forecast performance metrics
 * @param {Array} forecastReport - Forecast report data
 * @returns {Object} - Object with performance metrics
 */
function extractForecastPerformance(forecastReport) {
  return forecastReport.map(row => ({
    category: row.category,
    mape: row.mape,
    rmse: row.rmse,
    mae: row.mae,
    growth_rate: row.growth_rate
  }));
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
    clusterMetrics[cluster].total_sales += seller.total_sales;
    clusterMetrics[cluster].avg_processing_time += seller.avg_processing_time;
    clusterMetrics[cluster].avg_delivery_days += seller.avg_delivery_days;
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
    
    // Try to load additional KPIs from API if available
    return fetch('/api/supply-chain-kpis')
      .then(response => response.json())
      .then(apiKpis => {
        return {
          avg_processing_time: avgProcessingTime,
          forecast_growth: avgGrowthRate,
          total_demand: totalDemand,
          on_time_delivery: apiKpis.on_time_delivery || 85.0,
          perfect_order_rate: apiKpis.perfect_order_rate || 80.0,
          inventory_turnover: apiKpis.inventory_turnover || 8.0,
          // Keep track of which values are estimated
          estimated_fields: apiKpis.estimated_fields || 
            ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
        };
      })
      .catch(error => {
        console.warn('Error loading KPIs from API, using calculated values:', error);
        // Use derived estimates if API fails
        return {
          avg_processing_time: avgProcessingTime,
          forecast_growth: avgGrowthRate,
          total_demand: totalDemand,
          on_time_delivery: 85.0, // Industry benchmark
          perfect_order_rate: 80.0, // Industry benchmark
          inventory_turnover: 8.0, // Industry benchmark
          estimated_fields: ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
        };
      });
  }

/**
 * API function to fetch forecasts for a specific category
 * @param {string} category - Product category name
 * @returns {Promise<Object>} - Forecast data for the category
 */
export async function fetchCategoryForecast(category) {
  try {
    const response = await axios.get(`${API_BASE_URL}/forecasts/${encodeURIComponent(category)}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching forecast for ${category}:`, error);
    throw error;
  }
}

/**
 * API function to fetch seller performance details
 * @param {string} sellerId - Seller ID
 * @returns {Promise<Object>} - Seller performance details
 */
export async function fetchSellerDetails(sellerId) {
  try {
    const response = await axios.get(`${API_BASE_URL}/sellers/${encodeURIComponent(sellerId)}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching seller details for ${sellerId}:`, error);
    throw error;
  }
}