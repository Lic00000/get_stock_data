# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Get historical market data
ticker = yf.Ticker('TSLA')
tsla_df = ticker.history(period='1y')


# Print raw data format
print('\nDATA TYPES:')
print(tsla_df.dtypes)

print('\nMISSING VALUES PER COLUMN IN RAW DATA:')
raw_missing_values = tsla_df.isna().sum() # Missing values per column
print(raw_missing_values)

print('\nSHAPE BEFORE CLEANING:')
print(tsla_df.shape)

tsla_df = tsla_df.dropna(subset=['Close']) # Remove rows with missing close values only

print('\nSHAPE AFTER CLEANING:')
print(tsla_df.shape)


# Remove unnecessary columns
tsla_df = tsla_df.drop(columns=['Dividends', 'Stock Splits'])


# Calculate returns and rolling statistics
tsla_df['Simple return'] = tsla_df['Close'].pct_change() # (current close / previous close) - 1, which comes from (c - p) / p; p / p simplifies to 1
tsla_df['Log return'] = np.log(tsla_df['Close']/tsla_df['Close'].shift()) # ln(current close / previous close)
tsla_df['rm20'] = tsla_df['Close'].rolling(20).mean() # 20 day rolling mean of the closing price
tsla_df['rm50'] = tsla_df['Close'].rolling(50).mean() # 50 day rolling mean of the closing price
tsla_df['rm200'] = tsla_df['Close'].rolling(window=200).mean() # 200 day rolling mean of the closing price
tsla_df['rv20'] = tsla_df['Log return'].rolling(20).std() # 20 day rolling volatility of the log returns
tsla_df['rolling annualized volatility'] = tsla_df['rv20'] * np.sqrt(252) # Annualized volatility of the log returns, 252 is the number of trading days in a year


# Certain value calculations
first_close = tsla_df['Close'].iloc[0] # First closing price at index 0 in the 'Close' column
last_close = tsla_df['Close'].iloc[-1] # Last closing price
total_simple = last_close / first_close - 1 # Total simple return, (last price - first price) / first price which can be simplified to this
sum_of_log = tsla_df['Log return'].sum() # Total log return, it should be equal to ln(last/first) because of the product rule; ln(a) + ln(b) = ln(ab), where ln(p₁/p₀) + ln(p₂/p₁) + … + ln(pₙ/pₙ₋₁) = ln[(p₁/p₀) × (p₂/p₁) × … × (pₙ/pₙ₋₁)] = ln(pₙ/p₀)
back_to_simple = np.exp(sum_of_log) - 1 # Converts the log return to simple return so should be equal to total_simple
gap = (tsla_df['Simple return'] - tsla_df['Log return']).abs() # Defines the calculation for the absolute gap between simple and log returns
biggest_gap_date = gap.idxmax() # Date with the biggest gap using previous gap variable and applying it with index maximum to find the date
smallest_gap_date = gap.idxmin() # Date with the smallest gap
daily_volatility = tsla_df['Log return'].std() # Volatility of log returns
whole_period_annualized_volatility = daily_volatility * np.sqrt(252) # Annualized volatility
new_missing_values = tsla_df.isna().sum()  # Missing values per column in the new DataFrame


# Print missing data from new DataFrame
print('\nMISSING VALUES AFTER CALCULATIONS:')
print(new_missing_values)

# Print certain values
print('\nADDITIVITY VALUES:')
print(f'Total simple return:                       {total_simple:.6f}')
print(f'Total log returns:                         {sum_of_log:.6f}')
print(f'Total simple return (from log returns):    {back_to_simple:.6f}')

print('\nVOLATILITY VALUES:')
print(f'Whole period daily volatility:              {daily_volatility:.6f}')
print(f'Whole period annualized volatility:         {whole_period_annualized_volatility:.6f}')

print('\nGAP VALUES:')
print(f'Biggest gap: {gap.max():.6f} on {biggest_gap_date.date()}')
print(tsla_df.loc[biggest_gap_date, ['Simple return', 'Log return']])
print(f'\nSmallest gap: {gap.min():.6f} on {smallest_gap_date.date()}')
print(tsla_df.loc[smallest_gap_date, ['Simple return', 'Log return']])


# Print DataFrame
print(f'\nHEAD: \n {tsla_df.head()} \n {'_ '*50} \n\n TAIL: \n {tsla_df.tail()}')



# Plot charts
x = tsla_df.index
y1, y2, y3, y4 = tsla_df['Close'], tsla_df['rolling annualized volatility'], tsla_df['rm20'], tsla_df['rm50']

# Graph 1
plt.plot(x, y1, label='Close Price')
plt.plot(x, y3, label='20 Day Rolling Mean')
plt.plot(x, y4, label='50 Day Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Tesla\'s Close, rolling mean 20 day and rolling mean 50 day price')
plt.legend()

# Graph 2
plt.figure() # Start new empty graph to draw on
plt.plot(x, y2, label='annualized rolling volatility')
plt.axhline(y=whole_period_annualized_volatility, color='r', linestyle='--', label='Whole Period Annualized Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Tesla\'s Annualized Volatility')
plt.legend()
plt.show()

