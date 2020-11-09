from flask import Flask, render_template,request
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

import plotly.io as pio

app = Flask(__name__)


@app.route('/')
def index():
    #plot = create_plot()
    return render_template('index.html')

def create_plot():
    # if feature == 'Bar':
    #     N = 40
    #     x = np.linspace(0, 1, N)
    #     y = np.random.randn(N)
    #     df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    #     data = [
    #         go.Bar(
    #             x=df['x'], # assign x as the dataframe column 'x'
    #             y=df['y']
    #         )
    #     ]
    # else:
    #     N = 1000
    #     random_x = np.random.randn(N)
    #     random_y = np.random.randn(N)

    #     # Create a trace
    #     data = [go.Scatter(
    #         x = random_x,
    #         y = random_y,
    #         mode = 'markers'
    #     )]

    # Hardcode Bitcoin for now
    data = chart_stock("BTC-USD")
    
    pio.write_html(data, file='figure.html', auto_open=True)

    #graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return None

import yfinance as yf

def get_stock_history(ticker,history='max'):
    data = yf.Ticker(ticker).history(period=history)
    
    data = data.reset_index()
    for i in ['Open', 'High', 'Close', 'Low']:
        data[i]  =  data[i].astype('float64')

    return data

#btc = get_stock_history("BTC-USD")

def chart_stock(ticker):
    data = get_stock_history(ticker)

    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                   open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'])])
    #fig.show()

    return fig

if __name__ == '__main__':
    app.run()