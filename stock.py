import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import os

model_file = 'trained_model.joblib'
if not os.path.exists(model_file):
    # Download data for the ticker symbol 'QQQ'
    data = yf.download('QQQ')
    # print(data.head())

    # Create an empty DataFrame with specified columns
    num_days = len(data)
    dates = data.index
    prices = data['Adj Close'].values

    # Preallocate DataFrame with an estimated number of rows
    max_combinations = (num_days * (num_days - 1)) // 2  # Maximum possible combinations
    df = pd.DataFrame(index=np.arange(max_combinations), columns=['buyDate', 'sellDate', 'buyPrice', 'sellPrice'])

    # Use itertools.combinations to generate date pairs
    index = 0
    for i, j in combinations(range(num_days), 2):
        if j <= i + 130:  # hold 130 trading days maximum
            df.loc[index] = [dates[i], dates[j], prices[i], prices[j]]
            index += 1

    # Trim DataFrame to actual number of rows
    df = df.iloc[:index]

    # Calculate profit
    df['profit'] = (df['sellPrice'] - df['buyPrice']) / df['buyPrice']

    # Convert 'buyDate' and 'sellDate' to 'MMDD' format as integers
    df['buyDate'] = df['buyDate'].apply(lambda x: int(x.strftime('%m%d')))
    df['sellDate'] = df['sellDate'].apply(lambda x: int(x.strftime('%m%d')))

    print(df.head())

    X = df[['buyDate', 'sellDate']]
    y = df['profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model instance for regression
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror')

    # Fit model
    model.fit(X_train, y_train)

    joblib.dump(model, 'trained_model.joblib')

    # Make predictions
    preds = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, preds)
    print(f'Mean Squared Error: {mse:.2f}')
else:
    model = joblib.load(model_file)
    for i in list(range(805, 832)) + list(range(901, 931)) + list(range(1001, 1032)) + list(range(1101, 1131)) + list(range(1201, 1232)):
        single_record = np.array([[804, i]])
        prediction = model.predict(single_record)
        print(i, prediction)
