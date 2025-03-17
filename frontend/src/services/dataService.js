import axios from 'axios';
import Papa from 'papaparse';

// Base URL for API endpoints
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

/**
 * Enhanced function to parse CSV files with proper handling of data types
 * and special values.
 * 
 * @param {string} filePath - Path to the CSV file.
 * @param {Function} mockDataGenerator - Function to generate mock data if needed.
 * @returns {Promise<Array>} - Parsed CSV data as an array of objects.
 */
export async function loadCsvData(filePath, mockDataGenerator) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      console.warn(`CSV file not found: ${filePath}. Using mock data.`);
      return mockDataGenerator ? mockDataGenerator() : [];
    }
    const csvText = await response.text();
    return new Promise((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        delimitersToGuess: [',', '\t', '|', ';'],
        transformHeader: header => header.trim(),
        transform: (value, field) => {
          if (value === "" || value === undefined || value === null) {
            return null;
          }
          if (value === "N/A" || value === "n/a" || value === "NA") {
            return null;
          }
          if (field === "arima_params" && typeof value === "string") {
            return value.replace(/^["'](.+)["']$/, "$1");
          }
          return value;
        },
        complete: (results) => {
          if (results.errors && results.errors.length > 0) {
            console.error("CSV parsing errors:", results.errors);
            return reject(new Error("CSV parsing errors: " + JSON.stringify(results.errors)));
          }
          const processedData = results.data
            .filter(row => Object.keys(row).length > 1)
            .map(row => {
              const numericFields = [
                'avg_historical_demand', 'forecast_demand', 'min_forecast', 'max_forecast',
                'growth_rate', 'mape', 'rmse', 'mae', 'order_count', 'count',
                'avg_processing_time', 'avg_delivery_days', 'total_sales',
                'safety_stock', 'reorder_point', 'next_month_forecast',
                'on_time_delivery', 'perfect_order_rate', 'inventory_turnover',
                'lead_time_days', 'days_between_orders', 'avg_item_cost'
              ];
              for (const field of numericFields) {
                if (field in row && row[field] !== null) {
                  const parsed = parseFloat(row[field]);
                  if (!isNaN(parsed)) {
                    row[field] = parsed;
                  }
                }
              }
              if (row.date && typeof row.date === 'string') {
                try {
                  row.date = new Date(row.date);
                } catch (error) {
                  console.warn(`Could not parse date: ${row.date}`);
                }
              }
              if (!row.date && (row.order_year || row.year) && (row.order_month || row.month)) {
                const year = parseInt(row.order_year || row.year);
                const month = parseInt(row.order_month || row.month) - 1;
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
    throw error;
  }
}

/**
 * Generate consistent mock forecast report data.
 * The returned structure matches the real CSV data.
 * @returns {Array} - Mock forecast report data.
 */
function createMockForecastReport() {
  console.log("Generating mock forecast report data");
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  return categories.map(category => {
    const baseValue = category === "Electronics" ? 1540 : 
                      category === "Furniture" ? 980 :
                      category === "Clothing" ? 2100 :
                      category === "Books" ? 650 : 850;
    const variation = 0.1;
    const avgHistoricalDemand = Math.floor(baseValue * (1 + (Math.random() * 2 - 1) * variation));
    let growthRate, mape;
    switch(category) {
      case "Electronics":
        growthRate = 12.5;
        mape = 8.2;
        break;
      case "Furniture":
        growthRate = 5.3;
        mape = 15.4;
        break;
      case "Clothing":
        growthRate = 7.8;
        mape = 12.7;
        break;
      case "Books":
        growthRate = -2.3;
        mape = 18.9;
        break;
      case "Home Goods":
        growthRate = 3.2;
        mape = 10.5;
        break;
      default:
        growthRate = 0;
        mape = 10;
    }
    const forecastDemand = avgHistoricalDemand * (1 + growthRate / 100);
    const rmse = avgHistoricalDemand * (mape / 100) * 0.8;
    const mae = avgHistoricalDemand * (mape / 100) * 0.6;
    const forecastValues = [];
    for (let i = 1; i <= 6; i++) {
      const monthlyGrowth = growthRate / 100 * (i / 6);
      const noise = (Math.random() * 2 - 1) * (mape / 200);
      forecastValues.push(avgHistoricalDemand * (1 + monthlyGrowth + noise));
    }
    return {
      category,
      avg_historical_demand: avgHistoricalDemand,
      forecast_demand: forecastDemand,
      growth_rate: growthRate,
      mape,
      rmse,
      mae,
      arima_params: category === "Electronics" ? "(3,1,2)" :
                    category === "Furniture" ? "(2,1,2)" :
                    category === "Clothing" ? "(2,1,1)" :
                    category === "Books" ? "(1,1,1)" : "(2,1,0)",
      forecast_values: forecastValues,
      data_quality: category === "Books" ? "Limited" : "Sufficient",
      has_visualization: true
    };
  });
}

/**
 * Generate consistent mock monthly demand data.
 * @returns {Array} - Mock monthly demand data for past 12 months.
 */
function createMockMonthlyDemand() {
  console.log("Generating mock monthly demand data");
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  const results = [];
  categories.forEach(category => {
    const baseValue = category === "Electronics" ? 1540 : 
                      category === "Furniture" ? 980 :
                      category === "Clothing" ? 2100 : 
                      category === "Books" ? 650 : 850;
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = today.getMonth();
    for (let i = 0; i < 12; i++) {
      const monthOffset = (currentMonth - i) % 12;
      const month = monthOffset >= 0 ? monthOffset + 1 : monthOffset + 13;
      const year = currentYear - Math.floor((i - currentMonth) / 12);
      let seasonality = 0;
      if (category === "Electronics") {
        seasonality = (month >= 11 || month <= 1) ? 0.25 : (month >= 8 && month <= 10) ? 0.1 : -0.05;
      } else if (category === "Clothing") {
        seasonality = (month >= 4 && month <= 8) ? 0.2 : (month >= 2 && month <= 3) ? 0.1 : (month >= 9 && month <= 10) ? 0.05 : -0.1;
      } else if (category === "Books") {
        seasonality = (month >= 8 && month <= 9) ? 0.15 : (month === 12) ? 0.1 : 0;
      } else if (category === "Furniture") {
        seasonality = (month === 1) ? 0.2 : (month >= 6 && month <= 8) ? 0.15 : 0;
      } else {
        seasonality = (month >= 11 && month <= 12) ? 0.2 : (month >= 5 && month <= 7) ? 0.1 : 0;
      }
      const growthRate = category === "Electronics" ? 0.125 : 
                         category === "Furniture" ? 0.053 :
                         category === "Clothing" ? 0.078 :
                         category === "Books" ? -0.023 : 0.032;
      const monthsFromNow = i;
      const trend = -growthRate * monthsFromNow / 12;
      const noise = (Math.random() * 2 - 1) * 0.05;
      const count = Math.max(10, Math.round(baseValue * (1 + seasonality + trend + noise)));
      results.push({
        product_category_name: category,
        order_year: year,
        order_month: month,
        count,
        date: new Date(year, month - 1, 1)
      });
    }
  });
  return results;
}

/**
 * Generate mock seller clusters data.
 * @returns {Array} - Mock seller cluster data.
 */
function createMockSellerClusters() {
  console.log("Generating mock seller clusters data");
  const sellers = [];
  const clusters = [0, 1, 2];
  for (let i = 0; i < 50; i++) {
    const cluster = clusters[Math.floor(Math.random() * clusters.length)];
    let processingTime, deliveryDays, totalSales, onTimeRate, orderCount;
    if (cluster === 0) {
      processingTime = 1 + Math.random() * 1.5;
      deliveryDays = 3 + Math.random() * 2;
      orderCount = 100 + Math.random() * 200;
      totalSales = orderCount * (200 + Math.random() * 200);
      onTimeRate = 95 + Math.random() * 5;
    } else if (cluster === 1) {
      processingTime = 2.5 + Math.random() * 2;
      deliveryDays = 5 + Math.random() * 2;
      orderCount = 50 + Math.random() * 100;
      totalSales = orderCount * (150 + Math.random() * 150);
      onTimeRate = 80 + Math.random() * 15;
    } else {
      processingTime = 4 + Math.random() * 3;
      deliveryDays = 7 + Math.random() * 4;
      orderCount = 10 + Math.random() * 40;
      totalSales = orderCount * (100 + Math.random() * 100);
      onTimeRate = 60 + Math.random() * 20;
    }
    sellers.push({
      seller_id: `s${1000 + i}`,
      order_count: Math.floor(orderCount),
      avg_processing_time: processingTime,
      avg_delivery_days: deliveryDays,
      total_sales: totalSales,
      prediction: cluster,
      on_time_delivery_rate: onTimeRate,
      shipping_costs: totalSales * (0.05 + Math.random() * 0.1),
      avg_order_value: totalSales / Math.floor(orderCount)
    });
  }
  return sellers;
}

/**
 * Generate mock reorder recommendations data with enhanced supply chain metrics.
 * @returns {Array} - Mock reorder recommendations data.
 */
function createMockReorderRecommendations() {
  console.log("Generating mock reorder recommendations data");
  const categories = ["Electronics", "Furniture", "Clothing", "Books", "Home Goods"];
  return categories.map(category => {
    const baseValue = category === "Electronics" ? 1540 : 
                      category === "Furniture" ? 980 :
                      category === "Clothing" ? 2100 :
                      category === "Books" ? 650 : 850;
    const safetyFactor = category === "Electronics" ? 0.5 : 
                         category === "Furniture" ? 0.4 :
                         category === "Clothing" ? 0.55 :
                         category === "Books" ? 0.3 : 0.45;
    const safetyStock = Math.round(baseValue * safetyFactor);
    const leadTime = category === "Electronics" ? 5 : 
                     category === "Furniture" ? 10 :
                     category === "Clothing" ? 7 :
                     category === "Books" ? 4 : 6;
    const leadTimeFraction = leadTime / 30;
    const reorderPoint = Math.round((baseValue * leadTimeFraction) + safetyStock);
    const growthRate = category === "Electronics" ? 0.125 : 
                       category === "Furniture" ? 0.053 :
                       category === "Clothing" ? 0.078 :
                       category === "Books" ? -0.023 : 0.032;
    const nextMonthForecast = Math.round(baseValue * (1 + growthRate));
    const avgItemCost = category === "Electronics" ? 150 : 
                        category === "Furniture" ? 350 :
                        category === "Clothing" ? 45 :
                        category === "Books" ? 18 : 30;
    const annualDemand = baseValue * 12;
    const orderCost = 50;
    const holdingCostPct = 0.2;
    const holdingCost = avgItemCost * holdingCostPct;
    const eoq = Math.sqrt((2 * annualDemand * orderCost) / holdingCost);
    const orderFrequency = annualDemand / eoq;
    const daysBetweenOrders = Math.round(365 / orderFrequency);
    return {
      product_category: category,
      category,
      avg_monthly_demand: baseValue,
      safety_stock: safetyStock,
      reorder_point: reorderPoint,
      next_month_forecast: nextMonthForecast,
      forecast_demand: nextMonthForecast,
      growth_rate: growthRate * 100,
      lead_time_days: leadTime,
      days_between_orders: daysBetweenOrders,
      avg_item_cost: avgItemCost
    };
  });
}

/**
 * Generate mock state metrics data.
 * @returns {Array} - Mock state metrics data.
 */
function createMockStateMetrics() {
  console.log("Generating mock state metrics data");
  const states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"];
  return states.map(state => {
    let orderMultiplier, processingTimeModifier, deliveryDaysModifier, salesModifier, onTimeRateModifier;
    switch(state) {
      case "CA":
        orderMultiplier = 1.5;
        processingTimeModifier = 0.9;
        deliveryDaysModifier = 1.0;
        salesModifier = 1.4;
        onTimeRateModifier = 1.05;
        break;
      case "TX":
        orderMultiplier = 1.3;
        processingTimeModifier = 0.95;
        deliveryDaysModifier = 0.9;
        salesModifier = 1.2;
        onTimeRateModifier = 1.02;
        break;
      case "NY":
        orderMultiplier = 1.4;
        processingTimeModifier = 1.1;
        deliveryDaysModifier = 1.1;
        salesModifier = 1.5;
        onTimeRateModifier = 0.98;
        break;
      case "FL":
        orderMultiplier = 1.2;
        processingTimeModifier = 1.0;
        deliveryDaysModifier = 0.95;
        salesModifier = 1.1;
        onTimeRateModifier = 1.0;
        break;
      default:
        orderMultiplier = 0.8 + Math.random() * 0.4;
        processingTimeModifier = 0.9 + Math.random() * 0.3;
        deliveryDaysModifier = 0.9 + Math.random() * 0.3;
        salesModifier = 0.8 + Math.random() * 0.4;
        onTimeRateModifier = 0.95 + Math.random() * 0.1;
    }
    const baseOrders = 500;
    const orderCount = Math.floor(baseOrders * orderMultiplier);
    const avgProcessingTime = (2 + Math.random()) * processingTimeModifier;
    const avgDeliveryDays = (5 + Math.random() * 2) * deliveryDaysModifier;
    const totalSales = (50000 + Math.random() * 50000) * salesModifier;
    const onTimeRate = (85 + Math.random() * 10) * onTimeRateModifier;
    return {
      customer_state: state,
      order_count: orderCount,
      avg_processing_time: avgProcessingTime,
      avg_delivery_days: avgDeliveryDays,
      total_sales: totalSales,
      on_time_delivery_rate: Math.min(100, onTimeRate)
    };
  });
}

/**
 * Get top categories by total demand from monthly demand data.
 * @param {Array} data - Monthly demand data.
 * @param {Number} limit - Maximum number of categories to return.
 * @returns {Array} - Array of category names sorted by total demand.
 */
function getTopCategories(data, limit = 15) {
  if (!Array.isArray(data)) {
    console.warn('getTopCategories received non-array data');
    return [];
  }
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
  return Object.entries(categoryTotals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(entry => entry[0]);
}

/**
 * Group demand data by category.
 * @param {Array} data - Monthly demand data.
 * @returns {Object} - Object with category names as keys and arrays of data as values.
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
 * Extract forecast performance metrics.
 * @param {Array} forecastReport - Forecast report data.
 * @returns {Array} - Array with performance metrics.
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
    growth_rate: parseFloat(row.growth_rate) || 0,
    data_quality: row.data_quality || 'Unknown'
  }));
}

/**
 * Extract seller performance metrics.
 * This function calculates average metrics for seller clusters.
 * @param {Array} sellerClusters - Seller cluster data.
 * @returns {Object} - Object with seller metrics.
 */
function extractSellerMetrics(sellerClusters) {
  if (!Array.isArray(sellerClusters)) {
    console.warn('extractSellerMetrics received non-array data');
    return { clusterMetrics: {}, sellerCount: 0 };
  }
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
    clusterMetrics[cluster].total_sales += parseFloat(seller.total_sales) || 0;
    clusterMetrics[cluster].avg_processing_time += parseFloat(seller.avg_processing_time) || 0;
    clusterMetrics[cluster].avg_delivery_days += parseFloat(seller.avg_delivery_days) || 0;
  });
  Object.keys(clusterMetrics).forEach(cluster => {
    const metrics = clusterMetrics[cluster];
    if (metrics.count > 0) {
      metrics.avg_sales = metrics.total_sales / metrics.count;
      metrics.avg_processing_time = metrics.avg_processing_time / metrics.count;
      metrics.avg_delivery_days = metrics.avg_delivery_days / metrics.count;
    }
  });
  return { clusterMetrics, sellerCount: sellerClusters.length };
}

/**
 * Calculate key performance indicators directly without API calls.
 * @param {Array} demandData - Monthly demand data.
 * @param {Array} sellerData - Seller performance data.
 * @param {Array} forecastData - Forecast data.
 * @param {Array} performanceSummary - Performance summary data.
 * @returns {Object} - Object with calculated KPIs.
 */
function calculateKPIsDirectly(demandData, sellerData, forecastData, performanceSummary) {
  if (Array.isArray(performanceSummary) && performanceSummary.length > 0) {
    const summaryKPIs = {};
    performanceSummary.forEach(row => {
      if (!row || !row.metric || !row.value) return;
      let value = row.value;
      if (typeof value === 'string') {
        if (value.startsWith('$')) {
          value = parseFloat(value.substring(1).replace(/,/g, ''));
        } else if (value.endsWith('%')) {
          value = parseFloat(value.substring(0, value.length - 1));
        } else if (!isNaN(parseFloat(value))) {
          value = parseFloat(value);
        }
      }
      if (row.metric === 'Average Processing Time') {
        summaryKPIs.avg_processing_time = value;
      } else if (row.metric === 'Average Delivery Days') {
        summaryKPIs.avg_delivery_days = value;
      } else if (row.metric === 'On-Time Delivery Rate') {
        summaryKPIs.on_time_delivery = value;
      } else if (row.metric === 'Total Orders') {
        summaryKPIs.total_orders = value;
      } else if (row.metric === 'Total Products') {
        summaryKPIs.total_products = value;
      } else if (row.metric === 'Average Order Value') {
        summaryKPIs.average_order_value = value;
      } else if (row.metric === 'Forecast Accuracy (Avg MAPE)') {
        summaryKPIs.forecast_accuracy = value;
      }
    });
    if (Object.keys(summaryKPIs).length >= 3) {
      return summaryKPIs;
    }
  }
  let totalProcessingTime = 0;
  let sellerCount = 0;
  if (Array.isArray(sellerData)) {
    sellerData.forEach(seller => {
      if (seller && seller.avg_processing_time !== undefined) {
        const pt = parseFloat(seller.avg_processing_time);
        if (!isNaN(pt)) {
          totalProcessingTime += pt;
          sellerCount++;
        }
      }
    });
  }
  const avgProcessingTime = sellerCount > 0 ? totalProcessingTime / sellerCount : 0;
  let totalGrowthRate = 0;
  let forecastCount = 0;
  if (Array.isArray(forecastData)) {
    forecastData.forEach(forecast => {
      if (forecast && forecast.growth_rate !== null && forecast.growth_rate !== undefined) {
        const gr = parseFloat(forecast.growth_rate);
        if (!isNaN(gr)) {
          totalGrowthRate += gr;
          forecastCount++;
        }
      }
    });
  }
  const avgGrowthRate = forecastCount > 0 ? totalGrowthRate / forecastCount : 0;
  let totalDemand = 0;
  if (Array.isArray(demandData)) {
    demandData.forEach(row => {
      if (row) {
        const cnt = parseFloat(row.count || row.order_count || 0);
        if (!isNaN(cnt)) {
          totalDemand += cnt;
        }
      }
    });
  }
  let onTimeDelivery = 85.0;
  let onTimeCount = 0;
  if (Array.isArray(sellerData)) {
    sellerData.forEach(seller => {
      if (seller && seller.on_time_delivery_rate !== undefined) {
        const rate = parseFloat(seller.on_time_delivery_rate);
        if (!isNaN(rate)) {
          onTimeDelivery += rate;
          onTimeCount++;
        }
      }
    });
    if (onTimeCount > 0) {
      onTimeDelivery = onTimeDelivery / onTimeCount;
    }
  }
  const perfectOrderRate = onTimeDelivery * 0.9;
  const inventoryTurnover = 8.0;
  let avgOrderValue = 0;
  if (Array.isArray(sellerData) && sellerData.length > 0) {
    const totalSales = sellerData.reduce((sum, seller) => sum + (parseFloat(seller.total_sales) || 0), 0);
    const totalOrders = sellerData.reduce((sum, seller) => sum + (parseFloat(seller.order_count) || 0), 0);
    if (totalOrders > 0) {
      avgOrderValue = totalSales / totalOrders;
    }
  }
  let forecastAccuracy = 0;
  if (Array.isArray(forecastData) && forecastData.length > 0) {
    const totalMape = forecastData.reduce((sum, forecast) => {
      const m = parseFloat(forecast.mape || 0);
      return sum + (isNaN(m) ? 0 : m);
    }, 0);
    if (forecastData.length > 0) {
      forecastAccuracy = 100 - Math.min(100, totalMape / forecastData.length);
    }
  }
  return {
    avg_processing_time: avgProcessingTime,
    forecast_growth: avgGrowthRate,
    total_demand: totalDemand,
    on_time_delivery: onTimeDelivery,
    perfect_order_rate: perfectOrderRate,
    inventory_turnover: inventoryTurnover,
    average_order_value: avgOrderValue,
    forecast_accuracy: forecastAccuracy,
    estimated_fields: ['on_time_delivery', 'perfect_order_rate', 'inventory_turnover']
  };
}

/**
 * API function to fetch forecasts for a specific category.
 * @param {string} category - Product category name.
 * @returns {Promise<Object>} - Forecast data for the category.
 */
export async function fetchCategoryForecast(category) {
  try {
    try {
      const response = await axios.get(`${API_BASE_URL}/forecasts/${encodeURIComponent(category)}`);
      return response.data;
    } catch (apiError) {
      console.warn(`API not available for forecast data, loading from CSV fallback`);
      const forecastReport = await loadCsvData('/data/forecast_report.csv', createMockForecastReport);
      const categoryForecast = forecastReport.find(f => f.category === category || f.product_category === category);
      if (categoryForecast) {
        return categoryForecast;
      }
      console.warn(`No forecast data found for ${category}, generating mock data`);
      const mockForecastData = createMockForecastReport();
      const mockCategoryData = mockForecastData.find(f => f.category === category);
      if (mockCategoryData) {
        return mockCategoryData;
      }
      const mockForecast = {
        category,
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
 * API function to fetch seller performance details.
 * @param {string} sellerId - Seller ID.
 * @returns {Promise<Object>} - Seller performance details.
 */
export async function fetchSellerDetails(sellerId) {
  try {
    try {
      const response = await axios.get(`${API_BASE_URL}/sellers/${encodeURIComponent(sellerId)}`);
      return response.data;
    } catch (apiError) {
      console.warn(`API not available for seller data, loading from CSV fallback`);
      const sellerClusters = await loadCsvData('/data/seller_clusters.csv', createMockSellerClusters);
      const sellerDetails = sellerClusters.find(s => s.seller_id === sellerId);
      if (sellerDetails) {
        return sellerDetails;
      }
      console.warn(`No seller data found for ${sellerId}, generating mock data`);
      const clusterWeights = [0.2, 0.5, 0.3];
      const randomVal = Math.random();
      let sellerCluster;
      if (randomVal < clusterWeights[0]) {
        sellerCluster = 0;
      } else if (randomVal < clusterWeights[0] + clusterWeights[1]) {
        sellerCluster = 1;
      } else {
        sellerCluster = 2;
      }
      let processingTime, deliveryDays, totalSales, onTimeRate, orderCount;
      if (sellerCluster === 0) {
        processingTime = 1 + Math.random() * 1.5;
        deliveryDays = 3 + Math.random() * 2;
        orderCount = 100 + Math.random() * 200;
        totalSales = orderCount * (200 + Math.random() * 200);
        onTimeRate = 95 + Math.random() * 5;
      } else if (sellerCluster === 1) {
        processingTime = 2.5 + Math.random() * 2;
        deliveryDays = 5 + Math.random() * 2;
        orderCount = 50 + Math.random() * 100;
        totalSales = orderCount * (150 + Math.random() * 150);
        onTimeRate = 80 + Math.random() * 15;
      } else {
        processingTime = 4 + Math.random() * 3;
        deliveryDays = 7 + Math.random() * 4;
        orderCount = 10 + Math.random() * 40;
        totalSales = orderCount * (100 + Math.random() * 100);
        onTimeRate = 60 + Math.random() * 20;
      }
      const mockSeller = {
        seller_id: sellerId,
        order_count: Math.floor(orderCount),
        avg_processing_time: processingTime,
        avg_delivery_days: deliveryDays,
        total_sales: totalSales,
        prediction: sellerCluster,
        on_time_delivery_rate: onTimeRate,
        shipping_costs: totalSales * (0.05 + Math.random() * 0.1),
        avg_order_value: totalSales / Math.floor(orderCount)
      };
      return mockSeller;
    }
  } catch (error) {
    console.error(`Error fetching seller details for ${sellerId}:`, error);
    throw error;
  }
}

/**
 * Run a supply chain analysis with given parameters.
 * @param {Object} params - Analysis parameters.
 * @returns {Promise<Object>} - Analysis results.
 */
export async function runSupplyChainAnalysis(params) {
  try {
    const response = await axios.post(`${API_BASE_URL}/run-analysis`, params);
    return response.data;
  } catch (error) {
    console.error('Error running supply chain analysis:', error);
    throw error;
  }
}

/**
 * Export raw data files for downloading.
 * @param {String} filename - File to download.
 * @returns {Promise<Blob>} - File blob for download.
 */
export async function exportDataFile(filename) {
  try {
    const response = await fetch(`/data/${filename}`);
    if (!response.ok) {
      throw new Error(`File not found: ${filename}`);
    }
    return await response.blob();
  } catch (error) {
    console.error(`Error exporting file ${filename}:`, error);
    throw error;
  }
}

/**
 * Load dashboard data from CSV files with fallback to mock data.
 * @returns {Promise<Object>} - Object containing all dashboard data.
 */
export async function loadDashboardData() {
  try {
    const [
      monthlyDemand,
      forecastReport,
      sellerClusters,
      reorderRecommendations,
      stateMetrics,
      performanceSummary
    ] = await Promise.all([
      loadCsvData('/data/monthly_demand.csv', createMockMonthlyDemand),
      loadCsvData('/data/forecast_report.csv', createMockForecastReport),
      loadCsvData('/data/seller_clusters.csv', createMockSellerClusters),
      loadCsvData('/data/reorder_recommendations.csv', createMockReorderRecommendations),
      loadCsvData('/data/state_metrics.csv', createMockStateMetrics),
      loadCsvData('/data/performance_summary.csv', () => [])
    ]);
    const processedDemand = processDemandData(monthlyDemand);
    const topCategories = getTopCategories(monthlyDemand);
    const kpis = calculateKPIsDirectly(monthlyDemand, sellerClusters, forecastReport, performanceSummary);
    let topCategoryByState = [];
    try {
      topCategoryByState = await loadCsvData('/data/top_category_by_state.csv', () => []);
    } catch (error) {
      console.warn('Could not load top category by state data');
    }
    let clusterInterpretation = [];
    try {
      clusterInterpretation = await loadCsvData('/data/cluster_interpretation.csv', () => []);
    } catch (error) {
      console.warn('Could not load cluster interpretation data');
    }
    return {
      demandData: processedDemand,
      categories: {
        topCategories,
        categoryData: groupByCategory(monthlyDemand)
      },
      forecasts: {
        forecastReport,
        performanceMetrics: extractForecastPerformance(forecastReport)
      },
      sellerPerformance: {
        clusters: sellerClusters,
        metrics: extractSellerMetrics(sellerClusters),
        interpretation: clusterInterpretation
      },
      geography: {
        stateMetrics,
        topCategoryByState
      },
      recommendations: {
        inventory: reorderRecommendations
      },
      kpis,
      performance: performanceSummary
    };
  } catch (error) {
    console.error("Error loading dashboard data:", error);
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
      kpis: calculateKPIsDirectly(mockMonthlyDemand, mockSellerClusters, mockForecastReport, [])
    };
  }
}

/**
 * Process demand data for time series visualization.
 * @param {Array} data - Raw monthly demand data.
 * @returns {Array} - Processed demand data.
 */
function processDemandData(data) {
  if (!Array.isArray(data)) {
    console.warn('processDemandData received non-array data');
    return [];
  }
  const processedData = data.map(row => {
    if (!row) return null;
    const processedRow = { ...row };
    if (!processedRow.date) {
      const year = processedRow.year || processedRow.order_year;
      const month = processedRow.month || processedRow.order_month;
      if (year && month) {
        processedRow.date = new Date(parseInt(year), parseInt(month) - 1, 1);
      }
    } else if (typeof processedRow.date === 'string') {
      processedRow.date = new Date(processedRow.date);
    }
    if (!processedRow.count && processedRow.order_count) {
      processedRow.count = processedRow.order_count;
    }
    return processedRow;
  }).filter(row => row !== null);
  return processedData;
}

export default {
  loadDashboardData,
  fetchCategoryForecast,
  fetchSellerDetails,
  runSupplyChainAnalysis,
  exportDataFile
};
