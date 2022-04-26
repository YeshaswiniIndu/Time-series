import pandas as pd
import numpy as np
import seaborn as sns
import math
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt


train = pd.read_csv('/Users/yeshaswiniindukuri/Documents/kaggle/GOOGL.csv')
train['Date'] = pd.to_datetime(train['Date'])
forecast_out = int(math.ceil(0.05 * len(train))) # getting 5 percent of data that we will predict

train['HL_PCT'] = (train['High'] - train['Low']) / train['Low'] * 100.0 #the highs and lows every day. 
train['PCT_change'] = (train['Close'] - train['Open']) / train['Open'] * 100.0
df = train[['HL_PCT', 'PCT_change', 'Adj Close','Volume']] #adding only columns neccessary for prediction
forecast_out = int(math.ceil(0.05 * len(df))) 
print(forecast_out)
df['label'] = df['Adj Close'].shift(-forecast_out) #adding the extra dates that would be predicted
scaler = StandardScaler() #scaling data between 1 and -1
X = np.array(df.drop(['label'], 1))
scaler.fit(X)
X = scaler.transform(X)    
print(X)
X_Predictions = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #training the models
lr = LinearRegression() #using linear regression for predicting the model
lr.fit(X_train, y_train)
lr_confidence = lr.score(X_test, y_test) #the confidence score.

last_date = df.index[-1] 
last_date = pd.to_datetime(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day 
forecast_set = lr.predict(X_Predictions) 
df['Forecast'] = np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
print(df['Forecast'])
plt.figure(figsize=(18, 8))
df['Adj Close'].plot() #plotting the predicted data and the previous data
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

