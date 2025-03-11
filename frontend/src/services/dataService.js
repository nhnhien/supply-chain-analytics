import axios from 'axios';
import Papa from 'papaparse';

// Base URL for API endpoints
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

/**
 * Enhanced function to parse CSV files with proper handling of data types
 * and special values
 * 
 * @param {string} filePath - Path to the CSV file
 * @param {Function} mockDataGenerator - Function to generate mock data if needed
 * @returns {Promise<Array>} - Parsed CSV data as an array of objects
 */
async function loadCsvData(filePath, mockDataGenerator) {
  try {
    const response = await fetch(filePath);
    
    if (!response.ok) {
      console.warn(`CSV file not found: ${filePath}, using mock data instead`);
      return mockDataGenerator ? mockDataGenerator() : [];
    }
    
    const csvText = await response.text();
    
    return new Promise((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,  // This helps but doesn't handle all cases
        skipEmptyLines: true, // Skip empty lines in the CSV
        delimitersToGuess: [',', '\t', '|', ';'], // Try to detect different delimiters
        // Handle values that should be parsed as special cases
        transformHeader: header => header.trim(),
        transform: (value, field) => {
          // Handle empty values
          if (value === "" || value === undefined || value === null) {
            return null;
          }
          
          // Handle N/A values
          if (value === "N/A" || value === "n/a" || value === "NA") {
            return null;
          }
          
          // Handle CSV-quoted ARIMA parameters (e.g. "(1,1,1)")
          if (field === "arima_params" && typeof value === "string") {
            // Remove any extra quotes that might be in the field
            return value.replace(/^["'](.+)["']$/, "$1");
          }
          
          return value;
        },
        complete: (results) => {
          // Check if parsing resulted in error
          if (results.errors && results.errors.length > 0) {
            console.warn("CSV parsing had errors:", results.errors);
          }
          
          // Process the data further to ensure correct types
          const processedData = results.data
            .filter(row => Object.keys(row).length > 1) // Filter out empty rows
            .map(row => {
              // Process numeric fields that might still be strings
              const numericFields = [
                'avg_historical_demand', 'forecast_demand', 'min_forecast', 'max_forecast',
                'growth_rate', 'mape', 'rmse', 'mae', 'order_count', 'count',
                'avg_processing_time', 'avg_delivery_days', 'total_sales',
                'safety_stock', 'reorder_point', 'next_month_forecast',
                'on_time_delivery', 'perfect_order_rate', 'inventory_turnover'
              ];
              
              for (const field of numericFields) {
                if (field in row && row[field] !== null) {
                  const parsed = parseFloat(row[field]);
                  if (!isNaN(parsed)) {
                    row[field] = parsed;
                  }
                }
              }
              
              // Ensure date fields are proper Date objects when possible
              if (row.date && typeof row.date === 'string') {
                try {
                  row.date = new Date(row.date);
                } catch (error) {
                  console.warn(`Could not parse date: ${row.date}`);
                }
              }
              
              // If both year and month fields exist, create a date field if not already present
              if (!row.date && (row.order_year || row.year) && (row.order_month || row.month)) {
                const year = parseInt(row.order_year || row.year);
                const month = parseInt(row.order_month || row.month) - 1; // JavaScript months are 0-indexed
                if (!isNaN(year) && !isNaN(month)) {
                  row.date = new Date(year, month, 1);
                }
              }
              
              return row;
            });
            
          resolve(processedData);
        },
        error: (error) => {
          console.error(`Error parsing CSV: ${error}`);
          reject(error);
        }
      });
    });
  } catch (error) {
    console.error(`Error loading CSV from ${filePath}:`, error);
    return mockDataGenerator ? mockDataGenerator() : [];
  }
}

/**
 * Generate consistent mock forecast report data
 * @returns {Array} - Mock forecast report data with realistic values
 */
function createMockForecastReport() {
  console.log("Generating mock forecast report data");
  
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  
  return categories.map(category => {
    // Base demand is consistent for a given category
    const baseValue = category === "Electronics" ? 1540 : 
                     category === "Furniture" ? 980 :
                     category === "Clothing" ? 2100 :
                     category === "Books" ? 650 :
                     850; // Home Goods
    
    // Random variation but within reasonable bounds
    const variation = 0.1; // 10% variation
    const avgHistoricalDemand = Math.floor(baseValue * (1 + (Math.random() * 2 - 1) * variation));
    
    // Different categories have different growth trends
    let growthRate;
    let mape;
    
    switch(category) {
      case "Electronics":
        growthRate = 12.5; // Growing fast
        mape = 8.2;  // Good model accuracy
        break;
      case "Furniture":
        growthRate = 5.3;  // Moderate growth
        mape = 15.4; // Average model accuracy
        break;
      case "Clothing":
        growthRate = 7.8;  // Good growth
        mape = 12.7; // Good model accuracy
        break;
      case "Books":
        growthRate = -2.3; // Declining
        mape = 18.9; // Poorer model accuracy
        break;
      case "Home Goods":
        growthRate = 3.2;  // Slow growth
        mape = 10.5; // Good model accuracy
        break;
    }
    
    // Forecast demand calculated directly from growth rate
    const forecastDemand = avgHistoricalDemand * (1 + growthRate / 100);
    
    // RMSE and MAE based on MAPE and demand
    const rmse = avgHistoricalDemand * (mape / 100) * 0.8;
    const mae = avgHistoricalDemand * (mape / 100) * 0.6;
    
    // Generate 6 months of forecast values
    const forecastValues = [];
    for (let i = 1; i <= 6; i++) {
      // Apply growth rate gradually over 6 months
      const monthlyGrowth = growthRate / 100 * (i / 6);
      // Add some random noise to make it realistic
      const noise = (Math.random() * 2 - 1) * (mape / 200);
      forecastValues.push(
        avgHistoricalDemand * (1 + monthlyGrowth + noise)
      );
    }
    
    return {
      category: category,
      avg_historical_demand: avgHistoricalDemand,
      forecast_demand: forecastDemand,
      growth_rate: growthRate,
      mape: mape,
      rmse: rmse,
      mae: mae,
      arima_params: category === "Electronics" ? "(3,1,2)" :
                    category === "Furniture" ? "(2,1,2)" :
                    category === "Clothing" ? "(2,1,1)" :
                    category === "Books" ? "(1,1,1)" :
                    "(2,1,0)", // Home Goods
      forecast_values: forecastValues
    };
  });
}

/**
 * Generate consistent mock monthly demand data
 * @returns {Array} - Mock monthly demand data for past 12 months
 */
function createMockMonthlyDemand() {
  console.log("Generating mock monthly demand data");
  
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  const results = [];
  
  // Generate 12 months of data for each category
  categories.forEach(category => {
    // Base demand is consistent for a given category
    const baseValue = category === "Electronics" ? 1540 : 
                     category === "Furniture" ? 980 :
                     category === "Clothing" ? 2100 :
                     category === "Books" ? 650 :
                     850; // Home Goods
    
    // Get today's date and go back 12 months
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = today.getMonth();
    
    for (let i = 0; i < 12; i++) {
      // Calculate month and year for this data point
      const monthOffset = (currentMonth - i) % 12;
      const month = monthOffset >= 0 ? monthOffset + 1 : monthOffset + 13; // 1-12
      const year = currentYear - Math.floor((i - currentMonth) / 12);
      
      // Add seasonality effects
      // Spring/summer peak for clothing, winter peak for electronics, etc.
      let seasonality = 0;
      
      if (category === "Electronics") {
        // Peak in November/December (11-12)
        seasonality = (month >= 11 || month <= 1) ? 0.25 : 
                     (month >= 8 && month <= 10) ? 0.1 : 
                     -0.05;
      } else if (category === "Clothing") {
        // Peak in spring/summer (4-8)
        seasonality = (month >= 4 && month <= 8) ? 0.2 : 
                     (month >= 2 && month <= 3) ? 0.1 : 
                     (month >= 9 && month <= 10) ? 0.05 : 
                     -0.1;
      } else if (category === "Books") {
        // More stable with small peaks for back-to-school (8-9)
        seasonality = (month >= 8 && month <= 9) ? 0.15 : 
                     (month === 12) ? 0.1 : 
                     0;
      } else if (category === "Furniture") {
        // Peak in January and summer
        seasonality = (month === 1) ? 0.2 : 
                     (month >= 6 && month <= 8) ? 0.15 : 
                     0;
      } else {
        // Home goods - peak during holiday season
        seasonality = (month >= 11 && month <= 12) ? 0.2 : 
                     (month >= 5 && month <= 7) ? 0.1 : 
                     0;
      }
      
      // Add trend (growth or decline) over time
      // More recent months should reflect the growth rate from createMockForecastReport
      const growthRate = category === "Electronics" ? 0.125 : 
                        category === "Furniture" ? 0.053 :
                        category === "Clothing" ? 0.078 :
                        category === "Books" ? -0.023 :
                        0.032; // Home Goods
                        
      const monthsFromNow = i;
      const trend = -growthRate * monthsFromNow / 12;
      
      // Add some random noise
      const noise = (Math.random() * 2 - 1) * 0.05;
      
      // Calculate final count with seasonality, trend and noise
      const count = Math.max(10, Math.round(
        baseValue * (1 + seasonality + trend + noise)
      ));
      
      results.push({
        product_category_name: category,
        order_year: year,
        order_month: month,
        count: count,
        date: new Date(year, month - 1, 1)
      });
    }
  });
  
  return results;
}

/**
 * Generate mock seller clusters data
 * @returns {Array} - Mock seller cluster data
 */
function createMockSellerClusters() {
  console.log("Generating mock seller clusters data");
  
  const sellers = [];
  const clusters = [0, 1, 2]; // High, Medium, Low performers
  
  for (let i = 0; i < 50; i++) {
    const cluster = clusters[Math.floor(Math.random() * clusters.length)];
    
    // Characteristics based on cluster
    let processingTime, deliveryDays, totalSales;
    
    if (cluster === 0) { // High performers
      processingTime = 1 + Math.random() * 1.5;
      deliveryDays = 3 + Math.random() * 2;
      totalSales = 25000 + Math.random() * 75000;
    } else if (cluster === 1) { // Medium performers
      processingTime = 2.5 + Math.random() * 2;
      deliveryDays = 5 + Math.random() * 2;
      totalSales = 10000 + Math.random() * 20000;
    } else { // Low performers
      processingTime = 4 + Math.random() * 3;
      deliveryDays = 7 + Math.random() * 4;
      totalSales = 2000 + Math.random() * 8000;
    }
    
    sellers.push({
      seller_id: `s${1000 + i}`,
      order_count: Math.floor(totalSales / (50 + Math.random() * 100)),
      avg_processing_time: processingTime,
      avg_delivery_days: deliveryDays,
      total_sales: totalSales,
      prediction: cluster
    });
  }
  
  return sellers;
}

/**
 * Generate mock reorder recommendations data
 * @returns {Array} - Mock reorder recommendations data
 */
function createMockReorderRecommendations() {
  console.log("Generating mock reorder recommendations data");
  
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  
  return categories.map(category => {
    // Base demand is consistent per category
    const baseValue = category === "Electronics" ? 1540 : 
                     category === "Furniture" ? 980 :
                     category === "Clothing" ? 2100 :
                     category === "Books" ? 650 :
                     850; // Home Goods
                     
    // Calculate safety stock based on demand volatility
    const safetyFactor = category === "Electronics" ? 0.5 : // Higher for volatile categories
                        category === "Furniture" ? 0.4 :
                        category === "Clothing" ? 0.55 : // Fashion is volatile
                        category === "Books" ? 0.3 : // Books are stable
                        0.45; // Home Goods
                        
    const safetyStock = Math.round(baseValue * safetyFactor);
    
    // Calculate lead time
    const leadTime = category === "Electronics" ? 5 : // Days
                    category === "Furniture" ? 10 :
                    category === "Clothing" ? 7 :
                    category === "Books" ? 4 :
                    6; // Home Goods
                    
    // Calculate reorder point
    const leadTimeFraction = leadTime / 30; // As fraction of month
    const reorderPoint = Math.round((baseValue * leadTimeFraction) + safetyStock);
    
    // Calculate next month forecast with growth rate
    const growthRate = category === "Electronics" ? 0.125 : 
                      category === "Furniture" ? 0.053 :
                      category === "Clothing" ? 0.078 :
                      category === "Books" ? -0.023 :
                      0.032; // Home Goods
                      
    const nextMonthForecast = Math.round(baseValue * (1 + growthRate));
    
    return {
      product_category: category,
      avg_monthly_demand: baseValue,
      safety_stock: safetyStock,
      reorder_point: reorderPoint,
      next_month_forecast: nextMonthForecast,
      growth_rate: growthRate * 100 // Convert to percentage
    };
  });
}

/**
 * Generate mock state metrics data
 * @returns {Array} - Mock state metrics data
 */
function createMockStateMetrics() {
  console.log("Generating mock state metrics data");
  
  const states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"];
  
  return states.map(state => {
    // States have different characteristics
    let orderMultiplier, processingTimeModifier, deliveryDaysModifier, salesModifier;
    
    // Set state-specific modifiers
    switch(state) {
      case "CA":
        orderMultiplier = 1.5;
        processingTimeModifier = 0.9;
        deliveryDaysModifier = 1.0;
        salesModifier = 1.4;
        break;
      case "TX":
        orderMultiplier = 1.3;
        processingTimeModifier = 0.95;
        deliveryDaysModifier = 0.9;
        salesModifier = 1.2;
        break;
      case "NY":
        orderMultiplier = 1.4;
        processingTimeModifier = 1.1;
        deliveryDaysModifier = 1.1;
        salesModifier = 1.5;
        break;
      case "FL":
        orderMultiplier = 1.2;
        processingTimeModifier = 1.0;
        deliveryDaysModifier = 0.95;
        salesModifier = 1.1;
        break;
      default:
        // Random modifiers for other states
        orderMultiplier = 0.8 + Math.random() * 0.4;
        processingTimeModifier = 0.9 + Math.random() * 0.3;
        deliveryDaysModifier = 0.9 + Math.random() * 0.3;
        salesModifier = 0.8 + Math.random() * 0.4;
    }
    
    // Calculate metrics with state modifiers
    const baseOrders = 500;
    const orderCount = Math.floor(baseOrders * orderMultiplier);
    const avgProcessingTime = (2 + Math.random()) * processingTimeModifier;
    const avgDeliveryDays = (5 + Math.random() * 2) * deliveryDaysModifier;
    const totalSales = (50000 + Math.random() * 50000) * salesModifier;
    
    return {
      customer_state: state,
      order_count: orderCount,
      avg_processing_time: avgProcessingTime,
      avg_delivery_days: avgDeliveryDays,
      total_sales: totalSales
    };
  });
}

/**
 * Load dashboard data from CSV files with fallback to mock data
 * @returns {Promise<Object>} - Object containing all dashboard data
 */
export async function loadDashboardData() {
  try {
    // Load all required data files with fallbacks to mock data generators
    const [
      monthlyDemand,
      forecastReport,
      sellerClusters,
      reorderRecommendations,
      stateMetrics
    ] = await Promise.all([
      loadCsvData('/data/monthly_demand.csv', createMockMonthlyDemand),
      loadCsvData('/data/forecast_report.csv', createMockForecastReport),
      loadCsvData('/data/seller_clusters.csv', createMockSellerClusters),
      loadCsvData('/data/reorder_recommendations.csv', createMockReorderRecommendations),
      loadCsvData('/data/state_metrics.csv', createMockStateMetrics)
    ]);
    
    // Process monthly demand data for time series
    const processedDemand = processDemandData(monthlyDemand);
    
    // Get top categories from the demand data
    const topCategories = getTopCategories(monthlyDemand);
    
    // Calculate KPIs (now returns directly without API fetch which was causing 404)
    const kpis = calculateKPIsDirectly(monthlyDemand, sellerClusters, forecastReport);
    
    // Get top category by state data if available
    let topCategoryByState = [];
    try {
      topCategoryByState = await loadCsvData('/data/top_category_by_state.csv', () => []);
    } catch (error) {
      console.warn('Could not load top category by state data');
    }
    
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
        stateMetrics: stateMetrics,
        topCategoryByState: topCategoryByState
      },
      recommendations: {
        inventory: reorderRecommendations
      },
      kpis: kpis
    };
  } catch (error) {
    console.error("Error loading dashboard data:", error);
    
    // If everything fails, return complete mock data
    const mockMonthlyDemand = createMockMonthlyDemand();
    const mockForecastReport = createMockForecastReport();
    const mockSellerClusters = createMockSellerClusters();
    const mockStateMetrics = createMockStateMetrics();
    
    return {
      demandData: processDemandData(mockMonthlyDemand),
      categories: {
        topCategories: getTopCategories(mockMonthlyDemand),
        categoryData: groupByCategory(mockMonthlyDemand)
      },
      forecasts: {
        forecastReport: mockForecastReport,
        performanceMetrics: extractForecastPerformance(mockForecastReport)
      },
      sellerPerformance: {
        clusters: mockSellerClusters,
        metrics: extractSellerMetrics(mockSellerClusters)
      },
      geography: {
        stateMetrics: mockStateMetrics,
        topCategoryByState: [] // No mock data for this
      },
      recommendations: {
        inventory: createMockReorderRecommendations()
      },
      kpis: calculateKPIsDirectly(mockMonthlyDemand, mockSellerClusters, mockForecastReport)
    };
  }
}

/**
 * Process demand data for time series visualization
 * @param {Array} data - Raw monthly demand data
 * @returns {Array} - Processed demand data
 */
function processDemandData(data) {
  if (!Array.isArray(data)) {
    console.warn('processDemandData received non-array data');
    return [];
  }
  
  // Ensure date field exists
  const processedData = data.map(row => {
    if (!row) return null;
    
    // Create a copy to avoid mutating the original
    const processedRow = {...row};
    
    // Create date from year and month if not already present
    if (!processedRow.date) {
      const year = processedRow.year || processedRow.order_year;
      const month = processedRow.month || processedRow.order_month;
      if (year && month) {
        processedRow.date = new Date(parseInt(year), parseInt(month) - 1, 1);
      }
    } else if (typeof processedRow.date === 'string') {
      processedRow.date = new Date(processedRow.date);
    }
    
    // Ensure count field exists
    if (!processedRow.count && processedRow.order_count) {
      processedRow.count = processedRow.order_count;
    }
    
    return processedRow;
  }).filter(row => row !== null);
  
  return processedData;
}

/**
 * Get top categories by total demand
 * @param {Array} data - Monthly demand data
 * @param {number} limit - Maximum number of categories to return
 * @returns {Array} - Array of top category names
 */
function getTopCategories(data, limit = 5) {
  if (!Array.isArray(data)) {
    console.warn('getTopCategories received non-array data');
    return [];
  }
  
  // Group by category and sum the demand
  const categoryTotals = {};
  
  data.forEach(row => {
    if (!row) return;
    
    const category = row.product_category_name;
    const count = parseFloat(row.count || row.order_count || 0);
    
    if (category && !categoryTotals[category]) {
      categoryTotals[category] = 0;
    }
    
    if (category) {
      categoryTotals[category] += isNaN(count) ? 0 : count;
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
  if (!Array.isArray(data)) {
    console.warn('groupByCategory received non-array data');
    return {};
  }
  
  const grouped = {};
  
  data.forEach(row => {
    if (!row) return;
    
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
 * @returns {Array} - Array with performance metrics
 */
function extractForecastPerformance(forecastReport) {
  if (!Array.isArray(forecastReport)) {
    console.warn('extractForecastPerformance received non-array data');
    return [];
  }
  
  return forecastReport.filter(row => row && row.category).map(row => ({
    category: row.category || row.product_category,
    mape: parseFloat(row.mape) || null,
    rmse: parseFloat(row.rmse) || null,
    mae: parseFloat(row.mae) || null,
    growth_rate: parseFloat(row.growth_rate) || 0
  }));
}

/**
 * Extract seller performance metrics
 * @param {Array} sellerClusters - Seller cluster data
 * @returns {Object} - Object with seller metrics
 */
function extractSellerMetrics(sellerClusters) {
  if (!Array.isArray(sellerClusters)) {
    console.warn('extractSellerMetrics received non-array data');
    return { clusterMetrics: {}, sellerCount: 0 };
  }
  
  // Calculate average metrics by cluster
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
        avg_delivery_days: 0,
        order_count: 0
      };
    }
    
    clusterMetrics[cluster].count += 1;
    clusterMetrics[cluster].total_sales += parseFloat(seller.total_sales) || 0;
    clusterMetrics[cluster].avg_processing_time += parseFloat(seller.avg_processing_time) || 0;
    clusterMetrics[cluster].avg_delivery_days += parseFloat(seller.avg_delivery_days) || 0;
    clusterMetrics[cluster].order_count += parseFloat(seller.order_count) || 0;
  });
  
  // Calculate averages
  Object.keys(clusterMetrics).forEach(cluster => {
    const metrics = clusterMetrics[cluster];
    if (metrics.count > 0) {
      metrics.avg_sales = metrics.total_sales / metrics.count;
      metrics.avg_processing_time = metrics.avg_processing_time / metrics.count;
      metrics.avg_delivery_days = metrics.avg_delivery_days / metrics.count;
      metrics.avg_order_count = metrics.order_count / metrics.count;
    }
  });
  
  return {
    clusterMetrics: clusterMetrics,
    sellerCount: sellerClusters.length
  };
}

/**
 * Calculate key performance indicators directly without API calls
 * @param {Array} demandData - Monthly demand data
 * @param {Array} sellerData - Seller performance data
 * @param {Array} forecastData - Forecast data
 * @returns {Object} - Object with calculated KPIs
 */
function calculateKPIsDirectly(demandData, sellerData, forecastData) {
  // Average processing time
  let totalProcessingTime = 0;
  let sellerCount = 0;
  
  if (Array.isArray(sellerData)) {
    sellerData.forEach(seller => {
      if (seller && seller.avg_processing_time !== undefined) {
        const processingTime = parseFloat(seller.avg_processing_time);
        if (!isNaN(processingTime)) {
          totalProcessingTime += processingTime;
          sellerCount++;
        }
      }
    });
  }
  
  const avgProcessingTime = sellerCount > 0 ? totalProcessingTime / sellerCount : 0;
  
  // Average forecast growth rate
  let totalGrowthRate = 0;
  let forecastCount = 0;
  
  if (Array.isArray(forecastData)) {
    forecastData.forEach(forecast => {
      if (forecast && forecast.growth_rate !== null && forecast.growth_rate !== undefined) {
        const growthRate = parseFloat(forecast.growth_rate);
        if (!isNaN(growthRate)) {
          totalGrowthRate += growthRate;
          forecastCount++;
        }
      }
    });
  }
  
  const avgGrowthRate = forecastCount > 0 ? totalGrowthRate / forecastCount : 0;
  
  // Total demand
  let totalDemand = 0;
  
  if (Array.isArray(demandData)) {
    demandData.forEach(row => {
      if (row) {
        const count = parseFloat(row.count || row.order_count || 0);
        if (!isNaN(count)) {
          totalDemand += count;
        }
      }
    });
  }
  
  // Calculate on-time delivery from seller data if available
  let onTimeDelivery = 85.0; // Default to industry benchmark
  
  if (Array.isArray(sellerData) && sellerData.some(seller => seller && seller.on_time_delivery !== undefined)) {
    let totalOnTime = 0;
    let onTimeCount = 0;
    
    sellerData.forEach(seller => {
      if (seller && seller.on_time_delivery !== undefined) {
        const onTime = parseFloat(seller.on_time_delivery);
        if (!isNaN(onTime)) {
          totalOnTime += onTime;
          onTimeCount++;
        }
      }
    });
    
    if (onTimeCount > 0) {
      onTimeDelivery = totalOnTime / onTimeCount;
    }
  }
  
  // Perfect order rate calculation - typically 80-90% of on-time delivery rate
  const perfectOrderRate = onTimeDelivery * 0.9;
  
  // Inventory turnover - industry benchmark for e-commerce
  const inventoryTurnover = 8.0;
  
  // Return calculated KPIs
  return {
    avg_processing_time: avgProcessingTime,
    forecast_growth: avgGrowthRate,
    total_demand: totalDemand,
    on_time_delivery: onTimeDelivery,
    perfect_order_rate: perfectOrderRate,
    inventory_turnover: inventoryTurnover,
    // Flag estimated values for transparency in the UI
    estimated_fields: ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
  };
}

/**
 * API function to fetch forecasts for a specific category
 * @param {string} category - Product category name
 * @returns {Promise<Object>} - Forecast data for the category
 */
export async function fetchCategoryForecast(category) {
  try {
    // Try to fetch from API
    try {
      const response = await axios.get(`${API_BASE_URL}/forecasts/${encodeURIComponent(category)}`);
      return response.data;
    } catch (apiError) {
      console.warn(`API not available for forecast data, loading from CSV with fallback to mock data`);
      
      // Fallback to CSV data
      const forecastReport = await loadCsvData('/data/forecast_report.csv', createMockForecastReport);
      const categoryForecast = forecastReport.find(f => f.category === category || f.product_category === category);
      
      if (categoryForecast) {
        return categoryForecast;
      }
      
      // If not found in CSV, generate mock data for this specific category
      console.warn(`No forecast data found for ${category}, generating mock data`);
      
      // Create consistent mock data for this category
      const mockForecastData = createMockForecastReport();
      const mockCategoryData = mockForecastData.find(f => f.category === category);
      
      if (mockCategoryData) {
        return mockCategoryData;
      }
      
      // Last resort - generate random data
      const mockForecast = {
        category: category,
        avg_historical_demand: Math.floor(500 + Math.random() * 1500),
        growth_rate: (Math.random() * 20 - 5).toFixed(2),
        mape: (5 + Math.random() * 15).toFixed(2),
        rmse: 100 + Math.random() * 200,
        mae: 80 + Math.random() * 150,
        arima_params: "(1,1,1)"
      };
      
      return mockForecast;
    }
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
    // Try to fetch from API
    try {
      const response = await axios.get(`${API_BASE_URL}/sellers/${encodeURIComponent(sellerId)}`);
      return response.data;
    } catch (apiError) {
      console.warn(`API not available for seller data, loading from CSV with fallback to mock data`);
      
      // Fallback to CSV data
      const sellerClusters = await loadCsvData('/data/seller_clusters.csv', createMockSellerClusters);
      const sellerDetails = sellerClusters.find(s => s.seller_id === sellerId);
      
      if (sellerDetails) {
        return sellerDetails;
      }
      
      // If not found in CSV, generate mock data for this specific seller
      console.warn(`No seller data found for ${sellerId}, generating mock data`);
      const mockSeller = {
        seller_id: sellerId,
        order_count: Math.floor(50 + Math.random() * 200),
        avg_processing_time: 2 + Math.random() * 3,
        avg_delivery_days: 4 + Math.random() * 4,
        total_sales: 5000 + Math.random() * 20000,
        prediction: Math.floor(Math.random() * 3)
      };
      
      return mockSeller;
    }
  } catch (error) {
    console.error(`Error fetching seller details for ${sellerId}:`, error);
    throw error;
  }
}