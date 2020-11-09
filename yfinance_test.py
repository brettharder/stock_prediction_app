import yfinance as yf

msft = yf.Ticker("MSFT")
print(msft)
msft.info

data = msft.history(period="max")
msft.sustainability

bitcoin = yf.Ticker("BTC-USD")
bitcoin.info

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