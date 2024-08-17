import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization


# 获取股票数据和基本面数据
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # 添加基本面因子（例如市盈率）
    df['PE Ratio'] = yf.Ticker(ticker).info['trailingPE']  # 市盈率因子
    return df.dropna()


# 数据预处理：计算技术指标
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


# 执行策略，包含止损和止盈规则
def execute_strategy(df, rsi_overbought, rsi_oversold, stop_loss, take_profit):
    df = df.copy()
    df['Signal'] = 0

    # 生成多头信号
    df.loc[(df['Momentum'] > 0) & (df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] < rsi_oversold), 'Signal'] = 1

    # 生成空头信号
    df.loc[(df['SMA_Short'] < df['SMA_Long']) | (df['RSI'] > rsi_overbought), 'Signal'] = -1

    df['Position'] = df['Signal'].shift()

    # 计算策略收益率并加入止损和止盈规则
    df['Strategy Return'] = df['Position'] * df['Return']
    df['Strategy Return'] = df['Strategy Return'].apply(lambda x: min(max(x, stop_loss), take_profit))

    return df.dropna()


# 回测策略并计算性能指标
def backtest_strategy(df, weight_sharpe=0.7, weight_return=0.3):
    df = df.copy()
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    annual_return = df['Strategy Return'].mean() * 252
    annual_volatility = df['Strategy Return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    weighted_score = weight_sharpe * sharpe_ratio + weight_return * annual_return

    return df, weighted_score


# 贝叶斯优化目标函数
def bayes_opt_function(sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss,
                       take_profit, df_train):
    df_train_preprocessed = preprocess_data(df_train.copy(), sma_short, sma_long, momentum_period, rsi_period)
    df_train_strategy = execute_strategy(df_train_preprocessed, rsi_overbought, rsi_oversold, stop_loss, take_profit)
    _, score = backtest_strategy(df_train_strategy)
    return score


# 主函数
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    df = get_stock_data(ticker, start_date, end_date)

    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
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
            'stop_loss': (-0.05, -0.01),  # 止损范围
            'take_profit': (0.01, 0.05)  # 止盈范围
        }

        optimizer = BayesianOptimization(
            f=lambda sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss,
                     take_profit: bayes_opt_function(
                sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss, take_profit,
                df_train),
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

    print(f"Best Parameters: {best_params}")
    print(f"Best Weighted Score: {best_score:.2f}")

    # 使用最佳参数在最后的测试集上进行回测
    df_test_preprocessed = preprocess_data(df_test.copy(),
                                           best_params['sma_short'],
                                           best_params['sma_long'],
                                           best_params['momentum_period'],
                                           best_params['rsi_period'])
    df_test_strategy = execute_strategy(df_test_preprocessed,
                                        best_params['rsi_overbought'],
                                        best_params['rsi_oversold'],
                                        best_params['stop_loss'],
                                        best_params['take_profit'])
    df_test_result, test_score = backtest_strategy(df_test_strategy, weight_sharpe=0.7, weight_return=0.3)
    print(f"Weighted Score on Final Test Set: {test_score:.2f}")

    # 绘制测试集的回测结果
    plt.figure(figsize=(14, 7))
    plt.plot(df_test_result.index, df_test_result['Cumulative Strategy Return'], label='Strategy Return')
    plt.legend()
    plt.show()