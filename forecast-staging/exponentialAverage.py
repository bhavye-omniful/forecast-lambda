import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import boto3
import os

def exponential_smoothing_forecast(data, alpha = 0.3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a Pandas DataFrame.")
    
    return round(data.ewm(alpha=alpha, adjust=False).mean().values[-1])