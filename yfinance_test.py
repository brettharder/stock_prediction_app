import yfinance as yf
import numpy as np

msft = yf.Ticker("MSFT")
print(msft)
msft.info
msft.info['shortName']

testThis=yf.Ticker('bobs burgers')
testThis.info

def confirmStock(stockName):
      stock=yf.Ticker(stockName)
      returnName=np.nan
      try:
            returnName=stock.info['shortName']
      except:
            pass
      return returnName
confirmStock('CL=F')


data = msft.history(period="max")
msft.sustainability

bitcoin = yf.Ticker("BTC-USD")
bitcoin.info['name']

old = bitcoin.history(period='max')
old

old = old.reset_index()
for i in ['Open', 'High', 'Close', 'Low']:
      old[i]  =  old[i].astype('float64')

import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=old['Date'],
                                   open=old['Open'],
high=old['High'],
low=old['Low'],
close=old['Close'])])
fig.show()