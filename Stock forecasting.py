#!/usr/bin/env python
# coding: utf-8

# In[52]:


import yfinance as yf
import pandas as pd
# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
# Download historical data
data = yf.download(tickers, start="2015-01-01", end="2023-01-01", group_by='ticker')
# Save to CSV for future use
data.to_csv('stock_data.csv')


# In[53]:


# Load data from CSV
data = pd.read_csv('stock_data.csv', header=[0, 1], index_col=0, parse_dates=True)
# Handle missing values
data.ffill(inplace=True)  # Forward fill missing values
data.dropna(inplace=True)  # Drop any remaining rows with missing values
# Check for anomalies (e.g., negative prices)
print(data.describe())


# In[55]:


# Ensure the index is a datetime object
data.index = pd.to_datetime(data.index)
# Sort by date
data.sort_index(inplace=True)


# # EDA

# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[57]:


# Plot closing prices for each stock
for ticker in tickers:
    data[ticker]['Close'].plot(title=f'{ticker} Closing Prices')
    plt.show()

# Plot volume trends
for ticker in tickers:
    data[ticker]['Volume'].plot(title=f'{ticker} Trading Volume')
    plt.show()


# In[58]:


# Calculate and plot moving averages
for ticker in tickers:
    data[ticker]['Close'].rolling(window=30).mean().plot(label='30-Day MA')
    data[ticker]['Close'].rolling(window=90).mean().plot(label='90-Day MA')
    plt.title(f'{ticker} Moving Averages')
    plt.legend()
    plt.show()


# In[59]:


for ticker in tickers:
    # Lagged variables
    data[ticker, 'Lag1'] = data[ticker]['Close'].shift(1)
    # Rolling means
    data[ticker, 'RollingMean7'] = data[ticker]['Close'].rolling(window=7).mean()
    # Percentage changes
    data[ticker, 'PctChange'] = data[ticker]['Close'].pct_change()
# Drop rows with NaN values created by feature engineering
data.dropna(inplace=True)


# # Apple Stock

# In[60]:


# Select a stock for ARIMA modeling
stock = 'AAPL'
close_prices = data[stock]['Close']
# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]
# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Example parameters (p, d, q)
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))
# Evaluate
rmse = mean_squared_error(test, forecast, squared=False)
print(f'RMSE for {stock}: {rmse}')


# In[61]:


# ARIMA Forecast Visualization
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot testing data
plt.plot(test.index, test, label='Testing Data', color='green')
# Plot ARIMA forecast
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
# Add labels and title
plt.title(f'{stock} ARIMA Model: Training, Testing, and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# # Amazon stock

# In[64]:


# Select a stock for ARIMA modeling
stock = 'AMZN'
close_prices = data[stock]['Close']
# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]
# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Example parameters (p, d, q)
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))
# Evaluate
rmse = mean_squared_error(test, forecast, squared=False)
print(f'RMSE for {stock}: {rmse}')


# In[65]:


# ARIMA Forecast Visualization
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot testing data
plt.plot(test.index, test, label='Testing Data', color='green')
# Plot ARIMA forecast
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
# Add labels and title
plt.title(f'{stock} ARIMA Model: Training, Testing, and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# # Microsoft stock

# In[66]:


# Select a stock for ARIMA modeling
stock = 'MSFT'
close_prices = data[stock]['Close']
# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]
# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Example parameters (p, d, q)
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))
# Evaluate
rmse = mean_squared_error(test, forecast, squared=False)
print(f'RMSE for {stock}: {rmse}')


# In[46]:


# ARIMA Forecast Visualization
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot testing data
plt.plot(test.index, test, label='Testing Data', color='green')
# Plot ARIMA forecast
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
# Add labels and title
plt.title(f'{stock} ARIMA Model: Training, Testing, and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# # Google stock

# In[67]:


# Select a stock for ARIMA modeling
stock = 'GOOGL'
close_prices = data[stock]['Close']
# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]
# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Example parameters (p, d, q)
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))
# Evaluate
rmse = mean_squared_error(test, forecast, squared=False)
print(f'RMSE for {stock}: {rmse}')


# In[68]:


# ARIMA Forecast Visualization
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot testing data
plt.plot(test.index, test, label='Testing Data', color='green')
# Plot ARIMA forecast
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
# Add labels and title
plt.title(f'{stock} ARIMA Model: Training, Testing, and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# # Tesla stock

# In[69]:


# Select a stock for ARIMA modeling
stock = 'TSLA'
close_prices = data[stock]['Close']
# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]
# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Example parameters (p, d, q)
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))
# Evaluate
rmse = mean_squared_error(test, forecast, squared=False)
print(f'RMSE for {stock}: {rmse}')


# In[70]:


# ARIMA Forecast Visualization
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot testing data
plt.plot(test.index, test, label='Testing Data', color='green')
# Plot ARIMA forecast
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
# Add labels and title
plt.title(f'{stock} ARIMA Model: Training, Testing, and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# In[71]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Prepare features and target
features = ['Lag1', 'RollingMean7', 'PctChange']
X = data[stock][features]
y = data[stock]['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)

# Predict
y_pred = gb_model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE for Gradient Boosting: {rmse}')


# In[72]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ARIMA evaluation
arima_mae = mean_absolute_error(test, forecast)
arima_mape = mean_absolute_percentage_error(test, forecast)

# Gradient Boosting evaluation
gb_mae = mean_absolute_error(y_test, y_pred)
gb_mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'ARIMA MAE: {arima_mae}, MAPE: {arima_mape}')
print(f'Gradient Boosting MAE: {gb_mae}, MAPE: {gb_mape}')


# In[ ]:





# In[ ]:




