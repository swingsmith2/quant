import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization

# 获取股票数据及基本面数据
def get_stock_data(ticker, start_date, end_date):
    # 使用yfinance下载股票数据
    df = yf.download(ticker, start=start_date, end=end_date)
    # 计算股票收益率
    df['Return'] = df['Adj Close'].pct_change()
    # 计算对数收益率
    df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # 获取股票的基本面数据（如市盈率）
    ticker_info = yf.Ticker(ticker)
    pe_ratio = ticker_info.info.get('trailingPE', np.nan)  # 市盈率
    df['PE Ratio'] = pe_ratio

    return df.dropna()  # 删除任何包含NaN值的行

# 数据预处理，计算技术指标
def preprocess_data(df, sma_short, sma_long, momentum_period, rsi_period):
    df = df.copy()
    # 计算短期简单移动平均
    df['SMA_Short'] = df['Adj Close'].rolling(window=int(sma_short)).mean()
    # 计算长期简单移动平均
    df['SMA_Long'] = df['Adj Close'].rolling(window=int(sma_long)).mean()
    # 计算动量指标
    df['Momentum'] = df['Adj Close'].pct_change(periods=int(momentum_period))

    # 计算相对强弱指数（RSI）
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=int(rsi_period)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(rsi_period)).mean()
    df['RSI'] = 100 - 100 / (1 + gain / loss)

    return df.dropna()  # 删除任何包含NaN值的行

# 执行交易策略，包括止损和止盈规则
def execute_strategy(df, rsi_overbought, rsi_oversold, stop_loss, take_profit):
    df = df.copy()
    df['Signal'] = 0  # 初始化信号列

    # 生成买入信号：动量指标大于0，短期均线高于长期均线，且RSI低于超卖阈值
    df.loc[(df['Momentum'] > 0) & (df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] < rsi_oversold), 'Signal'] = 1
    # 生成卖出信号：短期均线低于长期均线，或RSI高于超买阈值
    df.loc[(df['SMA_Short'] < df['SMA_Long']) | (df['RSI'] > rsi_overbought), 'Signal'] = -1

    df['Position'] = df['Signal'].shift()  # 根据信号生成持仓情况
    df['Strategy Return'] = df['Position'] * df['Return']  # 计算策略收益率
    df['Strategy Return'] = np.clip(df['Strategy Return'], stop_loss, take_profit)  # 应用止损和止盈规则

    return df.dropna()  # 删除任何包含NaN值的行

# 回测策略并计算性能指标
def backtest_strategy(df, weight_sharpe=0.7, weight_return=0.3):
    df = df.copy()
    # 计算策略的累计收益
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    # 计算年化收益率
    annual_return = df['Strategy Return'].mean() * 252
    # 计算年化波动率
    annual_volatility = df['Strategy Return'].std() * np.sqrt(252)
    # 计算夏普比率
    sharpe_ratio = annual_return / annual_volatility
    # 加权得分
    weighted_score = weight_sharpe * sharpe_ratio + weight_return * annual_return

    return df, weighted_score

# 贝叶斯优化的目标函数
def bayes_opt_function(sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss, take_profit, df_train):
    df_train_preprocessed = preprocess_data(df_train.copy(), sma_short, sma_long, momentum_period, rsi_period)
    df_train_strategy = execute_strategy(df_train_preprocessed, rsi_overbought, rsi_oversold, stop_loss, take_profit)
    _, score = backtest_strategy(df_train_strategy)
    return score

if __name__ == "__main__":
    ticker = 'AAPL'  # 设定要分析的股票代码
    start_date = '2015-01-01'  # 数据开始日期
    end_date = datetime.today().strftime('%Y-%m-%d')  # 数据结束日期（当前日期）

    df = get_stock_data(ticker, start_date, end_date)  # 获取股票数据

    # 将数据集分割为训练集和测试集
    train_size = int(len(df) * 0.8)  # 设定80%作为训练集
    df_train = df[:train_size]  # 训练集数据
    df_test = df[train_size:]  # 测试集数据

    # 定义贝叶斯优化的参数范围
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

    # 创建贝叶斯优化器
    optimizer = BayesianOptimization(
        f=lambda sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss, take_profit:
        bayes_opt_function(sma_short, sma_long, momentum_period, rsi_period, rsi_overbought, rsi_oversold, stop_loss, take_profit, df_train),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # 执行贝叶斯优化
    optimizer.maximize(init_points=10, n_iter=30)

    # 输出最佳参数和得分
    best_score = optimizer.max['target']
    best_params = optimizer.max['params']
    best_params = {key: int(value) if key not in ['stop_loss', 'take_profit'] else value for key, value in best_params.items()}

    print(f"Best Parameters: {best_params}")
    print(f"Best Weighted Score on Training Set: {best_score:.2f}")

    # 在测试集上应用最佳参数进行回测
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
    print(f"Weighted Score on Test Set: {test_score:.2f}")

    # 绘制测试集的回测结果
    plt.figure(figsize=(14, 7))
    plt.plot(df_test_result.index, df_test_result['Cumulative Strategy Return'], label='Strategy Return')
    plt.legend()
    plt.show()

    # 计算年化收益率并添加到数据框
    df_test_result['Annualized Return'] = (df_test_result['Cumulative Strategy Return']) ** (
                252 / np.arange(1, len(df_test_result) + 1)) - 1

    # 绘制年化收益率图表
    plt.figure(figsize=(14, 7))
    plt.plot(df_test_result.index, df_test_result['Annualized Return'], label='Annualized Return')
    plt.title('Annualized Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Annualized Return')
    plt.legend()
    plt.show()
