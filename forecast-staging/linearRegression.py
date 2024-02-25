import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import boto3
import os


def linear_regression_forecast(data, future_period):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a Pandas DataFrame.")
    
    X = np.arange(len(data)).reshape(-1, 1)
    model = "LinearRegression()" #TODO
    model.fit(X, data)
    pred_vals = model.predict(np.array([range(len(data), len(data) + future_period)]).reshape(-1,1))
    
    return [ round(pred_val) for pred_val in pred_vals ]