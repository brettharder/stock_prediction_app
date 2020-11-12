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
os.getcwd()
import model as modelPack
import json



"""
Given a ticker this main method will train an LSTM on the oldest 80% of stock data predicting
closing price, and then test the model on the latest 20% of stock data.

Returns:
    list of plotly fig data
"""
num_epochs=100
lookback=20
data = modelPack.get_stock_history('ETH-USD')
data

price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
vals=scaler.fit_transform(price['Close'].values.reshape(-1,1))
price2=pd.DataFrame()
price2['Close']=vals.reshape(-1)


x_train, y_train, x_test, y_test = modelPack.split_data(price, 20)

model = modelPack.LSTM()
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
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

d_out = {}
d_out['train'] = {'x': data['Date'].values,'y': result[0].values}
d_out['test'] = {'x': data['Date'].values,'y': result[1].values}
d_out['actual'] = {'x': data['Date'].values,'y': result[2].values}


import plotly.express as px
import plotly.graph_objects as go
import plotly

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=d_out['train']['x'], y=d_out['train']['y'],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=d_out['test']['x'], y=d_out['test']['y'],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=d_out['actual']['x'], y=d_out['actual']['y'],
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
                              text='Results (LSTM)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()
plotlyDiv=plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
plotlyDiv


def plotThis(trainX,trainY,testX,testY,actualX,actualY):
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
                                text='Results (LSTM)',
                                font=dict(family='Rockwell',
                                            size=26,
                                            color='white'),
                                showarrow=False))
    fig.update_layout(annotations=annotations)
    fig.show()
    plotlyDiv=plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    return plotlyDiv

