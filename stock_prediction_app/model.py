"""
Main method will bring in yfinance stock data and train an LSTM on 
80% of historical stock data and use the latest 20% for testing. 
Output is a dictionary with plotly data which will be used as input
to a javascript plotly plot. 
"""
# Import packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import time
import math
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import plotly
os.getcwd()

# ----------------
# Helper functions
# ----------------

#Function to return if stock name is valid and return the short name of it
def confirmStock(stockName):
      stock=yf.Ticker(stockName)
      returnName=np.nan
      try:
            returnName=stock.info['shortName']
      except:
            pass
      return returnName


def get_stock_history(ticker,history='max'):
    data = yf.Ticker(ticker).history(period=history)
    
    data = data.reset_index()
    for i in ['Open', 'High', 'Close', 'Low']:
        data[i]  =  data[i].astype('float64')

    return data

def scale_stock_prices(data):
    price = data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

    return price

def create_torch_tensors(x):
    torch_tensor = torch.from_numpy(x).type(torch.Tensor)
    return torch_tensor

def split_data(stock, lookback=20):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = create_torch_tensors(data[:train_set_size,:-1,:])
    y_train = create_torch_tensors(data[:train_set_size,-1,:])
    
    x_test = create_torch_tensors(data[train_set_size:,:-1])
    y_test = create_torch_tensors(data[train_set_size:,-1,:])
    
    return [x_train, y_train, x_test, y_test]

# ---------------------
# LSTM 
# ---------------------
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# ---------------------
# GRU 
# ---------------------
class GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# -----------
# Main method
# -----------
def main(ticker,model='LSTM',lookback=20,num_epochs=100):
    """
    Given a ticker this main method will train an LSTM on the oldest 80% of stock data predicting
    closing price, and then test the model on the latest 20% of stock data.

    Returns:
        list of plotly fig data
    """

    data = get_stock_history(ticker)

    price = data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    vals=scaler.fit_transform(price['Close'].values.reshape(-1,1))
    price2=pd.DataFrame()
    price2['Close']=vals.reshape(-1)

    x_train, y_train, x_test, y_test = split_data(price2, lookback)

    assert model in ('LSTM','GRU'), "Please select model of type LSTM or GRU"
    if model=='LSTM':
        model = LSTM()
    else:
        model = GRU()

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        #print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time

    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price2)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price2)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price2)-1, :] = y_test_pred

    original = scaler.inverse_transform(price2['Close'].values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    d_out = {}
    d_out['train'] = {'x': list(data['Date'].dt.date.astype(str).values),'y': list(result[0].values)}
    d_out['test'] = {'x': list(data['Date'].dt.date.astype(str).values),'y': list(result[1].values)}
    d_out['actual'] = {'x': list(data['Date'].dt.date.astype(str).values),'y': list(result[2].values)}
        
    return d_out

def plotThis(trainX,trainY,testX,testY,actualX,actualY,modelType="LSTM"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=trainX, y=trainY,
                        mode='lines',
                        name='Train prediction')))
    fig.add_trace(go.Scatter(x=testX, y=testY,
                        mode='lines',
                        name='Test prediction'))
    fig.add_trace(go.Scatter(go.Scatter(x=actualX, y=actualY,
                        mode='lines',
                        name='Actual Value')))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template = 'plotly_dark'
    )

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text=f'Results ({modelType})',
                                font=dict(family='Rockwell',
                                            size=26,
                                            color='white'),
                                showarrow=False))
    fig.update_layout(annotations=annotations)
    #fig.show()
    plotlyDiv=plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    return plotlyDiv






# Example to run 
#data = main("BTC-USD")