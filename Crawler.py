import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
import requests
import csv
import time
from datetime import datetime

class BusCrawler:
    '''
    Crawler City Hall Exit B
    '''
    def __init__(self, api_key, urls):
        self.api_key = api_key
        self.urls = urls
        self.headers = {
            "AccountKey": self.api_key,
            "accept": "application/json"
        }
        self.csv_columns = ["CollectionTime", "BusStopCode", "ServiceNo", "Operator", 
                            "BusNo", "EstimatedArrival", "Latitude", 
                            "Longitude", "Load", "Type", "Feature"]
        self.last_data = None

    def fetch_data(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"请求失败，状态码: {response.status_code} for URL: {url}")
            return None

    def parse_bus_data(self, data, bus_stop_code, current_time):
        current_data = []
        current_data_for_comparison = []
        
        services = data.get('Services', [])
        for service in services:
            bus_info = [
                {"BusNo": "NextBus", "Data": service.get('NextBus', {})},
                {"BusNo": "NextBus2", "Data": service.get('NextBus2', {})},
                {"BusNo": "NextBus3", "Data": service.get('NextBus3', {})}
            ]

            for bus in bus_info:
                bus_data = bus['Data']
                if not bus_data:
                    continue

                row = {
                    "CollectionTime": current_time,
                    "BusStopCode": bus_stop_code,
                    "ServiceNo": service['ServiceNo'],
                    "Operator": service['Operator'],
                    "BusNo": bus["BusNo"],
                    "EstimatedArrival": bus_data.get('EstimatedArrival'),
                    "Latitude": bus_data.get('Latitude'),
                    "Longitude": bus_data.get('Longitude'),
                    "Load": bus_data.get('Load'),
                    "Type": bus_data.get('Type'),
                    "Feature": bus_data.get('Feature')
                }
                current_data.append(row)

                row_for_comparison = row.copy()
                del row_for_comparison["CollectionTime"]
                current_data_for_comparison.append(row_for_comparison)
        
        return current_data, current_data_for_comparison

    def preprocess(self, df):
            #preprocess
            df = pd.DataFrame(df)
            df.replace(["", "NaN", "None", "N/A"], np.nan, inplace=True)
            df.dropna(inplace=True)
            df['CollectionTime'] = pd.to_datetime(df['CollectionTime'])
            df['BusStopCode'] = df['BusStopCode'].astype(int)
            df['ServiceNo'] = df['ServiceNo'].map({
                                                                        '124':'1',
                                                                        '145':'2',
                                                                        '166':'3',
                                                                        '174':'4',
                                                                        '174e':'4',
                                                                        '195':'5',
                                                                        '195A':'5',
                                                                        '197':'6',
                                                                        '32':'7',
                                                                        '51':'8',
                                                                        '61':'9',
                                                                        '63':'10',
                                                                        '63A':'10',
                                                                        '80':'11',
                                                                        '851':'12',
                                                                        '851e':'12',
                                                                        '961':'13',
                                                                        '961M':'13'
                                                                    })
            df['ServiceNo'] = df['ServiceNo'].astype(int)
            df.rename(columns={'BusNo': 'Order'}, inplace=True)
            df['Order'] = df['Order'].astype(str)
            df['EstimatedArrival'] = pd.to_datetime(df['EstimatedArrival'])
            df['EstimatedArrival'] = pd.to_datetime(df['EstimatedArrival']).dt.tz_localize(None)
            df['Latitude'] = df['Latitude'].astype(float)
            df['Longitude'] = df['Longitude'].astype(float)
            df['Type'] = df['Type'].map({'SD': 1,
                                        'DD': 2
                                        })
            df['Type'] = df['Type'].astype(int)

            df['Load'] = df['Load'].map({'LSD': 1, 
                                        'SEA': 2, 
                                        'SDA': 3
                                        })
            df['Load'] = df['Load'].astype(int)

            df['Order'] = df['Order'].map({'NextBus': 1, 
                                        'NextBus2': 2, 
                                        'NextBus3': 3
                                        })
            df['Order'] = df['Order'].astype(int)

            df['VehCode'] = df['ServiceNo'].astype(str) + '00000' + df['Order'].astype(str)

            return df

    def collect(self):
        current_data = []
        current_data_for_comparison = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for url in self.urls:
            data = self.fetch_data(url)
            if data:
                bus_stop_code = url.split("BusStopCode=")[-1]
                parsed_data, parsed_data_for_comparison = self.parse_bus_data(
                    data, bus_stop_code, current_time
                )
                current_data.extend(parsed_data)
                current_data_for_comparison.extend(parsed_data_for_comparison)

        current_data = sorted(
            current_data, 
            key=lambda x: x["EstimatedArrival"] if x["EstimatedArrival"] else ""
        )

        if current_data_for_comparison != self.last_data:
            current_data = self.preprocess(current_data)
            print(f"Collect Time:{current_time}")
            self.last_data = current_data_for_comparison
        else:
            print("No New Data")

        return current_data
    
'''
api_key = "LhPnk7kfTDqb849G9KuhqA=="
urls = [
    "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=04167",
    "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=04168"
]
collector = BusCrawler(api_key, urls)
df = collector.collect()
print(df)
df.to_csv('1.csv')
'''