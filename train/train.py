import xgboost as xgb
import pandas as pd
import numpy as np
import os

# Training script for New York City taxi trip data
#
# Author: Tianyu Li
# Created on Feb 13th, 2019

# Load csv files into Pandas
rawdata = pd.read_csv("train.csv")

# Data cleaning helper function for NYC taxi data
#
# Inputs:
# data:pandas data frame to be cleaned)
def clean_data(data):
    # Load csv file into Pandas
    data = pd.read_csv("train.csv")
    
    # Data cleaning
    # Convert id to integer
    data.id = data.id.apply(lambda x: int(str(x)[2:]))
    
    # Convert pickup/dropoff time to unix timestamp
    data.pickup_datetime = data.pickup_datetime.apply(pd.Timestamp)
    data.pickup_datetime = data.pickup_datetime.astype(np.int64) // 10 ** 9
    
    data.dropoff_datetime = data.dropoff_datetime.apply(pd.Timestamp)
    data.dropoff_datetime = data.dropoff_datetime.astype(np.int64) // 10 ** 9
    
    # Convert the flag to 0 and 1 where 0 = Yes and 1 = No
    data.store_and_fwd_flag = data.store_and_fwd_flag.map(dict(Y = 1, N = 0))
    
    return data

# Clean data
cleaned_data = clean_data(rawdata)

# Load cleaned data to xgboost regressor
traindata = xgb.DMatrix(data = cleaned_data.iloc[:100000, :-1], label = cleaned_data.iloc[:100000, -1])
testdata = xgb.DMatrix(data = cleaned_data.iloc[100000:, :-1], label = cleaned_data.iloc[100000:, -1])

# Setting Parameters
param = {'max_depth' : 2, 'eta' : 1, 'silent' : 1, 'objective' : 'reg:linear'}
watchlist = [(testdata, 'eval'), (traindata, 'train')]

# Training
num_round = 3
bst = xgb.train(param, traindata, num_round, watchlist)
print("Start Training")

# Prediction
pred = bst.predict(testdata)
real = testdata.get_label()
print("Prediction result is ", pred)
print("Real result is ", real)
print("MSE is ", np.square(pred - real).mean())

# Plot
os.environ["PATH"] += os.pathsep + '/usr/local/bin/'
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees = 2)








