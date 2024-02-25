import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import boto3
import os

from slack import sendErrorToSlack

from movingAverage import moving_average_forecast
from exponentialAverage import exponential_smoothing_forecast
from linearRegression import linear_regression_forecast

def forecast_quantity(data, window_size, future_period, method, number_of_days):
    forecasts = []
    dates = []

    curr_date = data.iloc[-1, 0]
    
    curr_data = data[0:window_size]['quantity']
    
    if method == 'moving_avg':
        forecast = moving_average_forecast(curr_data, window_size)
    elif method == 'exponential_avg':
        forecast = exponential_smoothing_forecast(curr_data)
    elif method == 'linear_regression':
        forecast_data = linear_regression_forecast(curr_data, future_period)
        for _ in range(future_period):
            curr_date += timedelta(days=1)
            dates.append(curr_date.strftime("%Y-%m-%d"))
        return {'date': dates, 'quantity': forecast_data}
    else:
        return ValueError("Invalid method. Please choose 'moving_avg', 'exponential_avg', or 'linear_regression'.")
    
    
    # Append historical forecast        
    curr_date += timedelta(days=number_of_days)
    dates.append(curr_date.strftime("%Y-%m-%d"))
    
    forecasts.extend([forecast])

    # Generate forecasts for future periods
    for idx in range(1, future_period):
        curr_data = pd.Series(list(curr_data[idx:]) + forecasts[-window_size:], name = "quantity", dtype = 'int64')
        if method == 'moving_avg':
            forecast_next = moving_average_forecast(curr_data, window_size)
        else: #method == 'exponential_avg':
            forecast_next = exponential_smoothing_forecast(curr_data)
            
        # Append historical forecast
        
        dates.append(curr_date.strftime("%Y-%m-%d"))
        curr_date += timedelta(days=number_of_days)
        forecasts.append(forecast_next)

    return {'date': dates, 'quantity': forecasts}