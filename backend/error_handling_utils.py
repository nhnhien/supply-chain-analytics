# import logging

# # Create a logger for the ARIMA forecasting module
# logger = logging.getLogger('supply_chain_analytics.arima_forecasting')

# # If no handlers have been configured, add a default handler
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# def handle_forecasting_error(method_name, error, category=None, default_return=None, log_level='error'):
#     """
#     Centralized error handling function for forecasting operations.
    
#     Args:
#         method_name: Name of the method where the error occurred
#         error: The exception that was raised
#         category: Optional category name for context (e.g., product category)
#         default_return: Value to return when an error occurs
#         log_level: Logging level to use ('error', 'warning', or 'info')
        
#     Returns:
#         The default_return value
#     """
#     category_info = f" for category '{category}'" if category else ""
#     message = f"Error in {method_name}{category_info}: {str(error)}"
    
#     if log_level.lower() == 'error':
#         logger.error(message)
#     elif log_level.lower() == 'warning':
#         logger.warning(message)
#     else:
#         logger.info(message)
        
#     return default_return