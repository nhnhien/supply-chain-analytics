// import React, { useState, useEffect } from 'react';
// import { 
//   Box, Typography, Paper, Grid, Table, TableBody, TableCell, TableContainer, 
//   TableHead, TableRow, Button, Chip, Alert, CircularProgress, Card, CardContent,
//   Accordion, AccordionSummary, AccordionDetails, Divider, IconButton, Tooltip,
//   Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle
// } from '@mui/material';
// import {
//   ExpandMore as ExpandMoreIcon,
//   Refresh as RefreshIcon,
//   Delete as DeleteIcon,
//   CloudDownload as CloudDownloadIcon,
//   Storage as StorageIcon,
//   History as HistoryIcon
// } from '@mui/icons-material';
// import { getAnalysisRuns, getRunData, deleteRun } from '../services/mongoDbService';

// /**
//  * MongoDB Data Explorer Page Component
//  * Allows users to browse and manage analysis runs stored in MongoDB
//  */
// const MongoDBPage = () => {
//   const [analysisRuns, setAnalysisRuns] = useState([]);
//   const [selectedRun, setSelectedRun] = useState(null);
//   const [runData, setRunData] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
//   const [expandedSection, setExpandedSection] = useState('metadata');
//   const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
//   const [runToDelete, setRunToDelete] = useState(null);
//   const [deleteLoading, setDeleteLoading] = useState(false);
  
//   // Fetch analysis runs on component mount
//   useEffect(() => {
//     fetchAnalysisRuns();
//   }, []);
  
//   // Fetch run data when selected run changes
//   useEffect(() => {
//     if (selectedRun) {
//       fetchRunData(selectedRun);
//     } else {
//       setRunData(null);
//     }
//   }, [selectedRun]);
  
//   // Fetch list of analysis runs
//   const fetchAnalysisRuns = async () => {
//     setLoading(true);
//     setError(null);
//     try {
//       const runs = await getAnalysisRuns();
//       setAnalysisRuns(runs);
//       // Select the most recent run by default
//       if (runs.length > 0 && !selectedRun) {
//         setSelectedRun(runs[0].run_id);
//       }
//     } catch (err) {
//       console.error('Error fetching analysis runs:', err);
//       setError('Failed to load analysis runs. MongoDB may not be available.');
//     } finally {
//       setLoading(false);
//     }
//   };
  
//   // Fetch data for a specific run
//   const fetchRunData = async (runId) => {
//     setLoading(true);
//     setError(null);
//     try {
//       const data = await getRunData(runId);
//       setRunData(data);
//     } catch (err) {
//       console.error(`Error fetching data for run ${runId}:`, err);
//       setError(`Failed to load data for run ${runId}`);
//       setRunData(null);
//     } finally {
//       setLoading(false);
//     }
//   };
  
//   // Handle delete confirmation
//   const handleDeleteConfirm = async () => {
//     if (!runToDelete) return;
    
//     setDeleteLoading(true);
//     try {
//       await deleteRun(runToDelete);
//       // Refresh the runs list
//       fetchAnalysisRuns();
//       // If the deleted run was selected, clear selection
//       if (selectedRun === runToDelete) {
//         setSelectedRun(null);
//         setRunData(null);
//       }
//       setDeleteDialogOpen(false);
//     } catch (err) {
//       console.error(`Error deleting run ${runToDelete}:`, err);
//       setError(`Failed to delete run ${runToDelete}`);
//     } finally {
//       setDeleteLoading(false);
//       setRunToDelete(null);
//     }
//   };
  
//   // Handle opening delete dialog
//   const handleDeleteClick = (runId, event) => {
//     event.stopPropagation();
//     setRunToDelete(runId);
//     setDeleteDialogOpen(true);
//   };
  
//   // Format date for display
//   const formatDate = (dateString) => {
//     if (!dateString) return 'N/A';
//     return new Date(dateString).toLocaleString();
//   };
  
//   // Get count of records in a collection
//   const getCollectionCount = (collection) => {
//     if (!runData || !runData[collection]) return 0;
//     return runData[collection].length;
//   };
  
//   // Toggle expanded section
//   const handleAccordionChange = (section) => (event, isExpanded) => {
//     setExpandedSection(isExpanded ? section : false);
//   };
  
//   // Render loading state
//   if (loading && !runData && !analysisRuns.length) {
//     return (
//       <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
//         <CircularProgress />
//       </Box>
//     );
//   }
  
//   return (
//     <Box>
//       <Typography variant="h4" gutterBottom>
//         MongoDB Data Explorer
//       </Typography>
      
//       {error && (
//         <Alert severity="error" sx={{ mb: 3 }}>
//           {error}
//         </Alert>
//       )}
      
//       <Grid container spacing={3}>
//         {/* Analysis Runs List */}
//         <Grid item xs={12} md={4}>
//           <Paper elevation={2} sx={{ p: 2, height: '75vh', display: 'flex', flexDirection: 'column' }}>
//             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
//               <Typography variant="h6">Analysis Runs</Typography>
//               <Tooltip title="Refresh runs">
//                 <IconButton onClick={fetchAnalysisRuns} disabled={loading}>
//                   <RefreshIcon />
//                 </IconButton>
//               </Tooltip>
//             </Box>
            
//             {loading && !analysisRuns.length ? (
//               <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
//                 <CircularProgress />
//               </Box>
//             ) : analysisRuns.length === 0 ? (
//               <Alert severity="info">
//                 No analysis runs found in MongoDB. Run an analysis with the <code>--use-mongodb</code> flag to store results.
//               </Alert>
//             ) : (
//               <TableContainer sx={{ flexGrow: 1, overflow: 'auto' }}>
//                 <Table stickyHeader size="small">
//                   <TableHead>
//                     <TableRow>
//                       <TableCell>Run ID</TableCell>
//                       <TableCell>Timestamp</TableCell>
//                       <TableCell align="right">Actions</TableCell>
//                     </TableRow>
//                   </TableHead>
//                   <TableBody>
//                     {analysisRuns.map((run) => (
//                       <TableRow 
//                         key={run.run_id}
//                         hover
//                         selected={selectedRun === run.run_id}
//                         onClick={() => setSelectedRun(run.run_id)}
//                         sx={{ cursor: 'pointer' }}
//                       >
//                         <TableCell>
//                           <Tooltip title="Run ID">
//                             <Chip
//                               size="small"
//                               icon={<StorageIcon fontSize="small" />}
//                               label={run.run_id}
//                               color={selectedRun === run.run_id ? "primary" : "default"}
//                               variant={selectedRun === run.run_id ? "filled" : "outlined"}
//                             />
//                           </Tooltip>
//                         </TableCell>
//                         <TableCell>
//                           <Tooltip title="Timestamp">
//                             <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                               <HistoryIcon fontSize="small" sx={{ mr: 0.5, opacity: 0.6 }} />
//                               <Typography variant="body2">
//                                 {formatDate(run.timestamp || run.stored_at)}
//                               </Typography>
//                             </Box>
//                           </Tooltip>
//                         </TableCell>
//                         <TableCell align="right">
//                           <Tooltip title="Delete Run">
//                             <IconButton 
//                               size="small" 
//                               color="error"
//                               onClick={(e) => handleDeleteClick(run.run_id, e)}
//                             >
//                               <DeleteIcon fontSize="small" />
//                             </IconButton>
//                           </Tooltip>
//                         </TableCell>
//                       </TableRow>
//                     ))}
//                   </TableBody>
//                 </Table>
//               </TableContainer>
//             )}
//           </Paper>
//         </Grid>
        
//         {/* Run Details */}
//         <Grid item xs={12} md={8}>
//           <Paper elevation={2} sx={{ p: 2, height: '75vh', overflow: 'auto' }}>
//             {!selectedRun ? (
//               <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
//                 <Typography color="text.secondary">Select a run to view details</Typography>
//               </Box>
//             ) : loading ? (
//               <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
//                 <CircularProgress />
//               </Box>
//             ) : !runData ? (
//               <Alert severity="error">
//                 Failed to load run data. The run may have been deleted or MongoDB is unavailable.
//               </Alert>
//             ) : (
//               <Box>
//                 <Typography variant="h6" gutterBottom>
//                   Run Details: {selectedRun}
//                 </Typography>
                
//                 {/* Collection Statistics */}
//                 <Grid container spacing={2} sx={{ mb: 3 }}>
//                   <Grid item xs={6} md={3}>
//                     <Card variant="outlined">
//                       <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
//                         <Typography variant="body2" color="text.secondary">
//                           Demand Records
//                         </Typography>
//                         <Typography variant="h6">
//                           {getCollectionCount('demandData')}
//                         </Typography>
//                       </CardContent>
//                     </Card>
//                   </Grid>
//                   <Grid item xs={6} md={3}>
//                     <Card variant="outlined">
//                       <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
//                         <Typography variant="body2" color="text.secondary">
//                           Forecasts
//                         </Typography>
//                         <Typography variant="h6">
//                           {getCollectionCount('forecasts')}
//                         </Typography>
//                       </CardContent>
//                     </Card>
//                   </Grid>
//                   <Grid item xs={6} md={3}>
//                     <Card variant="outlined">
//                       <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
//                         <Typography variant="body2" color="text.secondary">
//                           Suppliers
//                         </Typography>
//                         <Typography variant="h6">
//                           {getCollectionCount('supplierClusters')}
//                         </Typography>
//                       </CardContent>
//                     </Card>
//                   </Grid>
//                   <Grid item xs={6} md={3}>
//                     <Card variant="outlined">
//                       <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
//                         <Typography variant="body2" color="text.secondary">
//                           Inventory Recs
//                         </Typography>
//                         <Typography variant="h6">
//                           {getCollectionCount('inventoryRecommendations')}
//                         </Typography>
//                       </CardContent>
//                     </Card>
//                   </Grid>
//                 </Grid>
                
//                 {/* Metadata Accordion */}
//                 <Accordion 
//                   expanded={expandedSection === 'metadata'} 
//                   onChange={handleAccordionChange('metadata')}
//                 >
//                   <AccordionSummary expandIcon={<ExpandMoreIcon />}>
//                     <Typography variant="subtitle1">Metadata</Typography>
//                   </AccordionSummary>
//                   <AccordionDetails>
//                     {runData.metadata ? (
//                       <Box>
//                         <Typography variant="subtitle2" gutterBottom>Parameters</Typography>
//                         <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
//                           <Table size="small">
//                             <TableBody>
//                               {runData.metadata.parameters && Object.entries(runData.metadata.parameters).map(([key, value]) => (
//                                 <TableRow key={key}>
//                                   <TableCell component="th" scope="row">{key}</TableCell>
//                                   <TableCell align="right">
//                                     {typeof value === 'boolean' ? 
//                                       (value ? 'Yes' : 'No') : 
//                                       value.toString()}
//                                   </TableCell>
//                                 </TableRow>
//                               ))}
//                             </TableBody>
//                           </Table>
//                         </TableContainer>
                        
//                         <Typography variant="subtitle2" gutterBottom>Environment</Typography>
//                         <TableContainer component={Paper} variant="outlined">
//                           <Table size="small">
//                             <TableBody>
//                               {runData.metadata.execution_environment && Object.entries(runData.metadata.execution_environment).map(([key, value]) => (
//                                 <TableRow key={key}>
//                                   <TableCell component="th" scope="row">{key}</TableCell>
//                                   <TableCell align="right">{value}</TableCell>
//                                 </TableRow>
//                               ))}
//                             </TableBody>
//                           </Table>
//                         </TableContainer>
//                       </Box>
//                     ) : (
//                       <Typography color="text.secondary">No metadata available</Typography>
//                     )}
//                   </AccordionDetails>
//                 </Accordion>
                
//                 {/* Demand Data Accordion */}
//                 <Accordion 
//                   expanded={expandedSection === 'demand'} 
//                   onChange={handleAccordionChange('demand')}
//                 >
//                   <AccordionSummary expandIcon={<ExpandMoreIcon />}>
//                     <Typography variant="subtitle1">Demand Data</Typography>
//                   </AccordionSummary>
//                   <AccordionDetails>
//                     {runData.demandData && runData.demandData.length > 0 ? (
//                       <TableContainer sx={{ maxHeight: 300 }}>
//                         <Table size="small" stickyHeader>
//                           <TableHead>
//                             <TableRow>
//                               <TableCell>Category</TableCell>
//                               <TableCell>Date</TableCell>
//                               <TableCell align="right">Count</TableCell>
//                             </TableRow>
//                           </TableHead>
//                           <TableBody>
//                             {runData.demandData.slice(0, 100).map((record, index) => (
//                               <TableRow key={index}>
//                                 <TableCell>{record.product_category_name}</TableCell>
//                                 <TableCell>{formatDate(record.date)}</TableCell>
//                                 <TableCell align="right">{record.count}</TableCell>
//                               </TableRow>
//                             ))}
//                           </TableBody>
//                         </Table>
//                       </TableContainer>
//                     ) : (
//                       <Typography color="text.secondary">No demand data available</Typography>
//                     )}
//                   </AccordionDetails>
//                 </Accordion>
                
//                 {/* Forecasts Accordion */}
//                 <Accordion 
//                   expanded={expandedSection === 'forecasts'} 
//                   onChange={handleAccordionChange('forecasts')}
//                 >
//                   <AccordionSummary expandIcon={<ExpandMoreIcon />}>
//                     <Typography variant="subtitle1">Forecasts</Typography>
//                   </AccordionSummary>
//                   <AccordionDetails>
//                     {runData.forecasts && runData.forecasts.length > 0 ? (
//                       <TableContainer sx={{ maxHeight: 300 }}>
//                         <Table size="small" stickyHeader>
//                           <TableHead>
//                             <TableRow>
//                               <TableCell>Category</TableCell>
//                               <TableCell align="right">Growth Rate</TableCell>
//                               <TableCell align="right">Forecast Values</TableCell>
//                             </TableRow>
//                           </TableHead>
//                           <TableBody>
//                             {runData.forecasts.map((forecast, index) => (
//                               <TableRow key={index}>
//                                 <TableCell>{forecast.category}</TableCell>
//                                 <TableCell align="right">
//                                   {forecast.growth_rate != null ? `${forecast.growth_rate.toFixed(2)}%` : 'N/A'}
//                                 </TableCell>
//                                 <TableCell align="right">
//                                   {forecast.forecast_values ? (
//                                     <Tooltip title={forecast.forecast_values.map((v, i) => `Month ${i+1}: ${v.toFixed(1)}`).join('\n')}>
//                                       <Button size="small" variant="outlined">View Values</Button>
//                                     </Tooltip>
//                                   ) : 'N/A'}
//                                 </TableCell>
//                               </TableRow>
//                             ))}
//                           </TableBody>
//                         </Table>
//                       </TableContainer>
//                     ) : (
//                       <Typography color="text.secondary">No forecast data available</Typography>
//                     )}
//                   </AccordionDetails>
//                 </Accordion>
                
//                 {/* Supplier Clusters Accordion */}
//                 <Accordion 
//                   expanded={expandedSection === 'suppliers'} 
//                   onChange={handleAccordionChange('suppliers')}
//                 >
//                   <AccordionSummary expandIcon={<ExpandMoreIcon />}>
//                     <Typography variant="subtitle1">Supplier Clusters</Typography>
//                   </AccordionSummary>
//                   <AccordionDetails>
//                     {runData.supplierClusters && runData.supplierClusters.length > 0 ? (
//                       <TableContainer sx={{ maxHeight: 300 }}>
//                         <Table size="small" stickyHeader>
//                           <TableHead>
//                             <TableRow>
//                               <TableCell>Seller ID</TableCell>
//                               <TableCell align="right">Cluster</TableCell>
//                               <TableCell align="right">Orders</TableCell>
//                               <TableCell align="right">Processing Time</TableCell>
//                             </TableRow>
//                           </TableHead>
//                           <TableBody>
//                             {runData.supplierClusters.slice(0, 100).map((supplier, index) => (
//                               <TableRow key={index}>
//                                 <TableCell>{supplier.seller_id}</TableCell>
//                                 <TableCell align="right">
//                                   <Chip 
//                                     size="small"
//                                     label={supplier.prediction === 0 ? 'High' : supplier.prediction === 1 ? 'Medium' : 'Low'}
//                                     color={supplier.prediction === 0 ? 'success' : supplier.prediction === 1 ? 'warning' : 'error'}
//                                   />
//                                 </TableCell>
//                                 <TableCell align="right">{supplier.order_count}</TableCell>
//                                 <TableCell align="right">{supplier.avg_processing_time?.toFixed(1)} days</TableCell>
//                               </TableRow>
//                             ))}
//                           </TableBody>
//                         </Table>
//                       </TableContainer>
//                     ) : (
//                       <Typography color="text.secondary">No supplier data available</Typography>
//                     )}
//                   </AccordionDetails>
//                 </Accordion>
                
//                 {/* Inventory Recommendations Accordion */}
//                 <Accordion 
//                   expanded={expandedSection === 'inventory'} 
//                   onChange={handleAccordionChange('inventory')}
//                 >
//                   <AccordionSummary expandIcon={<ExpandMoreIcon />}>
//                     <Typography variant="subtitle1">Inventory Recommendations</Typography>
//                   </AccordionSummary>
//                   <AccordionDetails>
//                     {runData.inventoryRecommendations && runData.inventoryRecommendations.length > 0 ? (
//                       <TableContainer sx={{ maxHeight: 300 }}>
//                         <Table size="small" stickyHeader>
//                           <TableHead>
//                             <TableRow>
//                               <TableCell>Category</TableCell>
//                               <TableCell align="right">Safety Stock</TableCell>
//                               <TableCell align="right">Reorder Point</TableCell>
//                               <TableCell align="right">Growth Rate</TableCell>
//                             </TableRow>
//                           </TableHead>
//                           <TableBody>
//                             {runData.inventoryRecommendations.map((rec, index) => (
//                               <TableRow key={index}>
//                                 <TableCell>{rec.product_category || rec.category}</TableCell>
//                                 <TableCell align="right">{Math.round(rec.safety_stock)}</TableCell>
//                                 <TableCell align="right">{Math.round(rec.reorder_point)}</TableCell>
//                                 <TableCell align="right">{rec.growth_rate?.toFixed(2)}%</TableCell>
//                               </TableRow>
//                             ))}
//                           </TableBody>
//                         </Table>
//                       </TableContainer>
//                     ) : (
//                       <Typography color="text.secondary">No inventory recommendations available</Typography>
//                     )}
//                   </AccordionDetails>
//                 </Accordion>
//               </Box>
//             )}
//           </Paper>
//         </Grid>
//       </Grid>
      
//       {/* Delete Confirmation Dialog */}
//       <Dialog
//         open={deleteDialogOpen}
//         onClose={() => setDeleteDialogOpen(false)}
//       >
//         <DialogTitle>Confirm Deletion</DialogTitle>
//         <DialogContent>
//           <DialogContentText>
//             Are you sure you want to delete run <strong>{runToDelete}</strong>? This action cannot be undone.
//           </DialogContentText>
//         </DialogContent>
//         <DialogActions>
//           <Button onClick={() => setDeleteDialogOpen(false)} disabled={deleteLoading}>
//             Cancel
//           </Button>
//           <Button 
//             onClick={handleDeleteConfirm} 
//             color="error" 
//             variant="contained"
//             disabled={deleteLoading}
//             startIcon={deleteLoading ? <CircularProgress size={20} /> : <DeleteIcon />}
//           >
//             {deleteLoading ? 'Deleting...' : 'Delete'}
//           </Button>
//         </DialogActions>
//       </Dialog>
//     </Box>
//   );
// };

// export default MongoDBPage;