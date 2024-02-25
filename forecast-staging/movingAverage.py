import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import boto3
import os

def moving_average_forecast(data, window_size):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a Pandas DataFrame.")
    
    rolling_mean = data.rolling(window=window_size).mean()

    if pd.notna(rolling_mean.values[-1]):
        # Round and return the value if it's not NaN
        return round(rolling_mean.values[-1])
    else:
        # Handle NaN case (e.g., return a default value)
        return 0  # Replace 0 with your desired default value