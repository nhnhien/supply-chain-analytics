// // frontend/src/services/mongoDbService.js
// import axios from 'axios';

// // Base URL for API endpoints
// const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

// /**
//  * Get a list of all analysis runs stored in MongoDB
//  * @returns {Promise<Array>} - List of analysis runs
//  */
// export async function getAnalysisRuns() {
//   try {
//     const response = await axios.get(`${API_BASE_URL}/mongo/runs`);
//     return response.data;
//   } catch (error) {
//     console.error('Error fetching analysis runs:', error);
//     throw error;
//   }
// }

// /**
//  * Get detailed data for a specific analysis run
//  * @param {string} runId - Analysis run ID
//  * @returns {Promise<Object>} - Analysis run data
//  */
// export async function getRunData(runId) {
//   try {
//     const response = await axios.get(`${API_BASE_URL}/mongo/run/${runId}`);
//     return response.data;
//   } catch (error) {
//     console.error(`Error fetching data for run ${runId}:`, error);
//     throw error;
//   }
// }

// /**
//  * Get the latest analysis data from MongoDB
//  * @returns {Promise<Object>} - Latest analysis data
//  */
// export async function getLatestData() {
//   try {
//     const response = await axios.get(`${API_BASE_URL}/mongo/latest`);
//     return response.data;
//   } catch (error) {
//     console.error('Error fetching latest data:', error);
//     throw error;
//   }
// }

// /**
//  * Get forecast data for a specific category from MongoDB
//  * @param {string} category - Product category name
//  * @returns {Promise<Object>} - Forecast data for the category
//  */
// export async function getCategoryForecast(category) {
//   try {
//     const response = await axios.get(`${API_BASE_URL}/mongo/forecasts/${encodeURIComponent(category)}`);
//     return response.data;
//   } catch (error) {
//     console.error(`Error fetching forecast for ${category}:`, error);
//     throw error;
//   }
// }

// /**
//  * Delete a specific analysis run from MongoDB
//  * @param {string} runId - Analysis run ID to delete
//  * @returns {Promise<Object>} - Deletion result
//  */
// export async function deleteRun(runId) {
//   try {
//     const response = await axios.delete(`${API_BASE_URL}/mongo/run/${runId}`);
//     return response.data;
//   } catch (error) {
//     console.error(`Error deleting run ${runId}:`, error);
//     throw error;
//   }
// }

// export default {
//   getAnalysisRuns,
//   getRunData,
//   getLatestData,
//   getCategoryForecast,
//   deleteRun
// };