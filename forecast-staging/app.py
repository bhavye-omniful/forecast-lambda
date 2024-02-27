import json
from datetime import datetime, timedelta
import uuid
from io import BytesIO
import sys
import pandas as pd
import numpy as np
import boto3
import os
from slack import sendErrorToSlack

# Testing event
# {
#   "method": "value1",
#   "future_period": "value2",
#   "window_size": "value3",
#   "data_url" : "s3 url of csv"
# }

try:
    access_key = os.environ['access_key']
    secret_key = os.environ['secret_key']
    queue_url = os.environ['queue_url']
    bucket_name = os.environ['bucket_name']
    s3_path = os.environ['s3_path']  # analytics/forecasts/downloads

    sqs = boto3.client('sqs', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

except KeyError as e:
    # Handle the case where the environment variable is not set
    print(f"Error: An unexpected error occurred: {e}")
    sendErrorToSlack(json.dumps({
        "Error" : f"{e}"
    }))
    sys.exit(f"Terminating Lambda execution due to error : {e}")
    # You might want to exit the script, log the error, or provide a default value

except Exception as e:
    # Handle other exceptions that might occur
    print(f"Error: An unexpected error occurred: {e}")
    sendErrorToSlack(json.dumps({
        "Error" : f"{e}"
    }))
    sys.exit(f"Terminating Lambda execution due to error : {e}")

def lambda_handler(event, context):
    # print("SQS Event : ", event)
    print("Event Record : ", event)
    print("Event Context : ", context)

    # event = json.loads(event)    # TODO 
    # print("Byte array event : ", event) 
    
    try: 
        # requesId = 
        event = json.loads(event["Records"][0]["body"])
        print("SQS Event : ", event)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sendErrorToSlack(json.dumps({
            "Error" : f"{e}"
        }))
        return
    
    
    # print(type(event))

    try:
        method = event["method"] # Method for forecasting: 'moving_avg', 'exponential_avg', or 'linear_regression'
        future_period = int(event["future_period"]) # In days
        window_size = int(event["window_size"]) # In days (on the basis of how much previous data we have to pick)
        url = event["data_url"] #Url of data stored in s3
        interval = event["interval"] # day, week, month, year 

    except KeyError as e:
        # Handle the case where the environment variable is not set
        print(f"Error: An unexpected error occurred: {e}")
        sendErrorToSlack(json.dumps({
            "Error" : f"{e}"
        }))
        return
        # You might want to exit the script, log the error, or provide a default value

    except Exception as e:
        # Handle other exceptions that might occur
        print(f"Error: An unexpected error occurred: {e}")
        sendErrorToSlack(json.dumps({
            "Error" : f"{e}"
        }))
        return

    # print("window size -->",type(window_size))
    # print("future period -->",type(future_period))

    try : 
        df_data = pd.read_csv(url, encoding="utf-16")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sendErrorToSlack(json.dumps({
            "Error" : f"{e}"
        }))
        return 

    # df_data = pd.read_csv(url)

    # df_data = pd.read_csv(StringIO(csv_content))

    df_data.info()

    #     Column           Non-Null Count  Dtype 
    #     ---------        --------------  ----- 
    # 0   seller_sku_code  925 non-null    object
    # 1   order_date       925 non-null    object
    # 2   quantity         925 non-null    int64 
    # 3   curr_inv         925 non-null    int64

    df_data['order_date'] = pd.to_datetime(df_data["order_date"],format="%Y-%m-%d")   # converting to appropriate date format
    df_data['order_date'] = df_data['order_date'].dt.date    # conveting to datetime from object
    df_data['quantity'] = df_data['quantity'].fillna(0)  # filling empty quantity items with 0
    df_data['curr_inventory'] = df_data['curr_inventory'].astype(int)
    # df_temp = df.groupby(['seller_sku_code'])['ordered_quantity'].sum().reset_index()
    # df_temp.sort_values(by = 'ordered_quantity', ascending = False)

    number_of_days = 0
    today = datetime.today()
    if interval == 'day' : 
        number_of_days = 1
        today = today.strftime('%Y-%m-%d')   # YYYY-MM-DD
    elif interval == 'week' :
        number_of_days = 7
        today = today - timedelta(days=today.weekday() + 1)
        today = today.strftime('%Y-%m-%d')
    elif interval == 'month' : 
        number_of_days = 30     
        today = today.strftime('%Y-%m-01')   # YYYY-MM-01
    elif interval == 'year' :
        number_of_days = 365
        today = today.strftime('%Y-01-01')   # YYYY-01-01

    dates = pd.date_range(end=today, periods=window_size, freq=f'{number_of_days}D')
    date_range = pd.DataFrame(dates, columns=['date'])
    date_range['date'] = pd.to_datetime(date_range['date']).dt.date

    forecastingResult = pd.DataFrame()

    skus = df_data['seller_sku_code'].unique()

    for sku in skus :
        # print("sku ==>", sku)
        
        df = df_data[df_data['seller_sku_code'].astype(str) == str(sku)] 
        curr_inv = df_data['curr_inventory'].iloc[0]
        df = df.drop('curr_inventory', axis=1)
        df = df.drop('seller_sku_code', axis=1)

        df.columns = ['date', 'quantity']

        df = date_range.merge(df, on='date', how='left')
        df['quantity'] = df['quantity'].fillna(0) 

        # print(df)
        # df.plot(x='order_created_at', y='ordered_quantity')

        if method == "mean":
            sum = df['quantity'].sum()
            df_forecast = pd.DataFrame()
            df_forecast["seller_sku_code"] = [sku]
            df_forecast["mean"] = [sum/window_size]
            forecastingResult = pd.concat([forecastingResult, df_forecast], ignore_index=True)
            continue

        # Forecast
        forecast = forecast_quantity(
            data = df, 
            window_size = window_size, # previous history data 7d, 10d etc
            future_period = future_period, # future forecase next 2d, 3d etc
            method = method, # 'moving_avg', 'exponential_avg', 'linear_regression'
            number_of_days = number_of_days # day, week, month, year
        )

        try:
            df_forecast = pd.DataFrame(forecast).T
            df_forecast = df_forecast.reset_index()
            df_forecast.columns = df_forecast.iloc[0]
            df_forecast = df_forecast.drop(df_forecast.columns[0], axis=1)
            df_forecast = df_forecast[1:]
        
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
            sendErrorToSlack(json.dumps({
                "Error" : f"{e}"
            }))
            return 

        row_sum = df_forecast.sum(axis=1)
        avg = row_sum/(future_period*number_of_days)
        df_forecast['days_on_hand'] = 0
        if not avg.eq(0).all():
           df_forecast['days_on_hand'] = int(curr_inv/avg)

        df_forecast.insert(0, "seller_sku_code", sku)

        # Concatenate the DataFrames
        forecastingResult = pd.concat([forecastingResult, df_forecast], ignore_index=True)
    
    # json_string = json.dumps(forecastingResults)
    # byte_array = json_string.encode('utf-8') # Convert the JSON string to a byte array

    try:
        csv_buffer = BytesIO()
        forecastingResult.to_csv(csv_buffer, index=False)

        # print(csv_buffer.getvalue())

        csv_buffer.seek(0)

        unique_filename = f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.csv"

        s3.upload_fileobj(csv_buffer, bucket_name, s3_path + "/" + unique_filename)
        forecast_csv_result = f"https://{bucket_name}.s3.eu-central-1.amazonaws.com/{s3_path}/{unique_filename}"

    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sendErrorToSlack(json.dumps({
            "Error" : f"{e}"
        }))
        return 

    json_result = {
        "url" : forecast_csv_result,
        "user_details" : event['user_details']
    } 

    response = sqs.send_message(
        QueueUrl=queue_url,
        DelaySeconds=10,
        MessageBody=(
            json.dumps(json_result)
        )
    )
    
    # print(forecastingResults)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "sqsResponse": response,
        }),
    }


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
        
        curr_date += timedelta(days=number_of_days)
        dates.append(curr_date.strftime("%Y-%m-%d"))
        forecasts.append(forecast_next)

    return {'date': dates, 'quantity': forecasts}

def linear_regression_forecast(data, future_period):
    X = np.arange(len(data)).reshape(-1, 1)
    model = "LinearRegression()" #TODO
    model.fit(X, data)
    pred_vals = model.predict(np.array([range(len(data), len(data) + future_period)]).reshape(-1,1))
    
    return [ round(pred_val) for pred_val in pred_vals ]

def exponential_smoothing_forecast(data, alpha = 0.3):    
    return round(data.ewm(alpha=alpha, adjust=False).mean().values[-1])

def moving_average_forecast(data, window_size):
    rolling_mean = data.rolling(window=window_size).mean()

    if pd.notna(rolling_mean.values[-1]):
        # Round and return the value if it's not NaN
        return round(rolling_mean.values[-1])
    else:
        # Handle NaN case (e.g., return a default value)
        return 0  # Replace 0 with your desired default value
    

# ----------------------------------------------------------------
# jsonEvent = {
#   "method": "moving_avg",
#   "future_period": "7",
#   "window_size": "30",
#   "data_url" : "https://omniful-bucket.s3.eu-central-1.amazonaws.com/zrafhDataLambda.csv",
#   "skus" : [
#     "856553007122"
#   ]
# }

# jsonEvent = {'Records': [{'messageId': '5142a061-2915-40e6-91ad-59d6c2c97441', 'receiptHandle': 'AQEBpmUqBvXD0iDpVktMyxKvaIwwmMBCcpogPbeyTA12+TKc6tgsvZtO7dh8vNrkmd2IkVO4OvIJZd9DAHerNC26OMe+FNhbfcxRYJgTIHfLPjB8xU43v1jYlaDStn88CXJH4hh59HcrwhkfcCdfzC9KmRo3rGWCZAMtc3jSMXCbm31CNoSEEEuJtN6mL0Io3EbMzcMHtvtYKc95y12kAvOYuSv9mwb7yjCvZFwlo+b/JQ+2HrCr34wxmKE+KDKraff3BeLU/LZjTc2cs0CrNi34Q3ISa+661PeeLY/9GbtGGBfTI5pd/1ILv3VJpiaMpTdCsxWxIlYaA/1EB69z6ztqkqUEB2CQtsbVkcMoZsR9CUudZWDRKBlws705fclD6sOdDMQ2aTZS8UUITpgrRq5HzQ==', 'body': '{"seller_id":"2","method":"moving_avg","future_period":"7","window_size":2,"interval":"week","skus":null,"data_url":"https://omniful-bucket.s3.amazonaws.com/analytics/reports/downloads/_1_20240223030119.csv","user_details":{"tenant_id":"1","user_email":"bhavye.goel2002@gmail.com","time_zone":"","user_id":"","user_name":"bhavye","self_tenant":false}}', 'attributes': {'ApproximateReceiveCount': '1', 'SentTimestamp': '1708637481690', 'SenderId': 'AIDAYS2NVPSRBBLCL4I54', 'ApproximateFirstReceiveTimestamp': '1708637481695'}, 'messageAttributes': {'X-Omniful-Request-Id': {'stringValue': '7b3ce645-e592-4eba-a11a-597c8502b15b', 'stringListValues': [], 'binaryListValues': [], 'dataType': 'String'}}, 'md5OfMessageAttributes': '0aa26e29b4f22dcd05bcb1ba4ea55b27', 'md5OfBody': 'dce46df4cdf558cf2fc816d56559b44f', 'eventSource': 'aws:sqs', 'eventSourceARN': 'arn:aws:sqs:eu-central-1:590184021154:lambda-queue', 'awsRegion': 'eu-central-1'}]}


# temp = lambda_handler(jsonEvent, context="")
# print(temp)
# ----------------------------------------------------------------


# bucket_name = 'omniful-bucket'
# csv_file = 'zrafhDataLambda.csv'

# s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
# s3 = boto3.client('s3')
# bucket = s3.get_object(Bucket=bucket_name, Key=csv_file)

# csv_content = bucket['Body'].read().decode('utf-8')
