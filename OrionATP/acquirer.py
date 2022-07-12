import requests
from datetime import datetime
import csv
import pandas as pd
from pandas.io.json import json_normalize
import json

#converter script
api = open("api.txt", "r")
api_key = api.readline()

try: #check if api response is a json
    raw_df = pd.read_json(api_key)
except:
    print("File not in json format")
    try:
        raw_df = pd.read_csv(api_key)
    except:
        pass
    else:
        raw_df = pd.read_csv(api_key)
else:
    raw_df = pd.read_json(api_key)

time = api.readline().strip() #read from the api file, time will be first a string representation of the name of the "time" column in the API call
value = api.readline().strip() #read from the api file, value will be first a string representation of the name of the "value" column in the API call
print(time, value, raw_df)
#might need try / except blocks but you can check online for advice on converters
#the difficult thing will be finding dates / times and the one metric you care about in the timeseries, then making it into a 2 column df

standardized_time = raw_df[[value,time]] #remember that datetime needs to be changed to a variable that represents the time column
original_time = pd.DataFrame({"timestamp":standardized_time[time], "value": standardized_time[value]}) #a copy of the raw df for publisher purposes

selected = standardized_time.iloc[[-5, -4, -3, -2, -1]]
#standardized_time["datetime"] = standardized_time[time].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
standardized_time[time] = pd.to_timedelta(standardized_time[time])
standardized_time[time] = standardized_time[time].dt.total_seconds()

final_df = pd.DataFrame({"timestamp":standardized_time[time], "value": standardized_time[value]}) #this line will need to be edited to change datetime and close, but is important
print(final_df)
original_time["seconds"] = final_df["timestamp"]
#lastly the final dataframe needs to be sent to trainer.py to train the model
