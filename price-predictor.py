from alpha_vantage.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('API_KEY')

config = {
    "alpha_vantage": {
        "key": api_key,
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",  # CHANGE THIS COLOR LATER??
        "color_train": "#3D9970",  # CHANGE THIS COLOR LATER??
        "color_val": "#0074D9",  # CHANGE THIS COLOR LATER??
        "color_pred_train": "#3D9970",  # CHANGE THIS COLOR LATER??
        "color_pred_val": "#0074D9",  # CHANGE THIS COLOR LATER??
        "color_pred_test": "#FF4136",  # CHANGE THIS COLOR LATER??
    },
    "model": {
        "input_size": 1, # the closing price is the feature we can use here
        "num_lstm_layers": 2,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "schedular_step_size": 40,
    }
}

"""
PULL DATA FROM API
------------------
Pulling stock price data from Alpha Vantage's API and graphing the adjusted closing price of the specified stock over time.
-----------------------------------------------------------------------------------------------------------------------------
"""
def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "From " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range

# Graphing historical price chart
fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])

xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]   # adjusting the x axis in matplotlib
x = np.arange(0, len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily adj close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.show()

"""
NORMALIZE DATA
--------------
LSTM algorithms use something called "gradient descent" as the algo's optimization technique. This means that we must
normalize our data to avoid skewing the model in unpredictable ways. This will help increase the accuracy of our LSTM model,
and help the "gradient descent" technique run smoothly. By scaling the input data by the same scale and reducing the 
variance, means that the neural network resources will not be wasted on normalizing tasks. LSTMs are VERY sensitive to the 
scale of the input data, and for this reason, the normalization process is crucial when using this algo.
The normalization process takes place below in the class "Normalizer". Here we are rescaling the data to have a mean of 0 &
a standard deviation of 1.
-----------------------------------------------------------------------------------------------------------------------------
"""

class Normalizer():
    def __init__():
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x
    
    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
# Normalize
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

"""
SPLIT DATASETS
--------------
LSTM is a "Supervised Machine Learning", as such, we need to create a training data set and a validation data set by 
splitting our normalized data set into two different sets. We will train the model to try and predict the 21st day's adj 
closing price based on day 20's closing price. 
-----------------------------------------------------------------------------------------------------------------------------
"""

def prepare_data_x(x, window_size):
    # Windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # use the next day as the label
    output = x[window_size]
    return output

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# SPLIT DATASET
split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

# PLOT DATA
to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)


