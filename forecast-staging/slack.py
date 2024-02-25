import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import boto3
import os

def sendErrorToSlack(message):
    slack_client = os.environ['slack_webhook']
    payload = {'text': message}
    requests.post(slack_client, json=payload)