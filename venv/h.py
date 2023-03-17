import yfinance as yf

data_train = yf.download(tickers='TTM', period='1y', interval='1mo')
print(data_train)