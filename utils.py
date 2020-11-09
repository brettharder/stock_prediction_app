import yfinance as yf

def get_stock_history(ticker,history='max'):
    data = yf.Ticker(ticker).history(period=history)
    
    data = data.reset_index()
    for i in ['Open', 'High', 'Close', 'Low']:
        data[i]  =  data[i].astype('float64')

    return data

btc = get_stock_history("BTC-USD")

def chart_stock(ticker):
    data = get_stock_history(ticker)

    fig = go.Figure(data=[go.Candlestick(x=old['Date'],
                                   open=old['Open'],
    high=old['High'],
    low=old['Low'],
    close=old['Close'])])
    fig.show()

    return None

#chart_stock("BTC-USD")