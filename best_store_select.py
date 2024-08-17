import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization


def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    df['PE Ratio'] = yf.Ticker(ticker).info.get('trailingPE', np.nan)  # 获取市盈率因子
    return df.dropna()


def preprocess_data(df, sma_short, sma_long, momentum_period, rsi_period):
    df = df.copy()
    df['SMA_Short'] = df['Adj Close'].rolling(window=int(sma_short)).mean()
    df['SMA_Long'] = df['Adj Close'].rolling(window=int(sma_long)).mean()
    df['Momentum'] = df['Adj Close'].pct_change(periods=int(momentum_period))

    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=int(rsi_period)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(rsi_period)).mean()
    df['RSI'] = 100 - 100 / (1 + gain / loss)

    return df.dropna()


def execute_strategy(df, rsi_overbought, rsi_oversold, stop_loss, take_profit):
    df = df.copy()
    df['Signal'] = 0

    df.loc[(df['Momentum'] > 0) & (df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] < rsi_oversold), 'Signal'] = 1
    df.loc[(df['SMA_Short'] < df['SMA_Long']) | (df['RSI'] > rsi_overbought), 'Signal'] = -1

    df['Position'] = df['Signal'].shift()
    df['Strategy Return'] = df['Position'] * df['Return']
    df['Strategy Return'] = df['Strategy Return'].apply(lambda x: min(max(x, stop_loss), take_profit))

    return df.dropna()


def backtest_strategy(df, weight_sharpe=0.7, weight_return=0.3):
    df = df.copy()
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    annual_return = df['Strategy Return'].mean() * 252
    annual_volatility = df['Strategy Return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    weighted_score = weight_sharpe * sharpe_ratio + weight_return * annual_return

    return df, weighted_score


def bayes_opt_function(sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss,
                       take_profit, df_train):
    df_train_preprocessed = preprocess_data(df_train.copy(), sma_short, sma_long, momentum_period, rsi_period)
    df_train_strategy = execute_strategy(df_train_preprocessed, rsi_overbought, rsi_oversold, stop_loss, take_profit)
    _, score = backtest_strategy(df_train_strategy)
    return score


def select_best_stock(tickers, start_date, end_date, n_splits=5):
    best_stocks = []

    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = get_stock_data(ticker, start_date, end_date)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = float('-inf')
        best_params = None

        for train_index, test_index in tscv.split(df):
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]

            pbounds = {
                'sma_short': (10, 50),
                'sma_long': (100, 200),
                'momentum_period': (5, 20),
                'rsi_period': (10, 20),
                'rsi_overbought': (70, 80),
                'rsi_oversold': (20, 30),
                'stop_loss': (-0.05, -0.01),
                'take_profit': (0.01, 0.05)
            }

            optimizer = BayesianOptimization(
                f=lambda sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss,
                         take_profit: bayes_opt_function(
                    sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss,
                    take_profit, df_train),
                pbounds=pbounds,
                random_state=42,
                verbose=2
            )

            optimizer.maximize(init_points=10, n_iter=30)

            if optimizer.max['target'] > best_score:
                best_score = optimizer.max['target']
                best_params = optimizer.max['params']
                best_params = {key: int(value) if key not in ['stop_loss', 'take_profit'] else value for key, value in
                               best_params.items()}

        best_stocks.append((ticker, best_score, best_params))

    best_stocks = sorted(best_stocks, key=lambda x: x[1], reverse=True)
    best_stock = best_stocks[0]

    print(f"Best stock: {best_stock[0]} with score {best_stock[1]} and parameters {best_stock[2]}")

    return best_stocks


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # 可扩展到前100只股票
    start_date = '2015-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    best_stocks = select_best_stock(tickers, start_date, end_date)
    df_best_stocks = pd.DataFrame(best_stocks, columns=['Ticker', 'Best Score', 'Best Parameters'])
    df_best_stocks.to_csv('best_stocks_selection.csv', index=False)
