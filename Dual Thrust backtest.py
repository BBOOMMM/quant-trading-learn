# -*- coding: utf-8 -*-
"""
Dual Thrust 开盘区间突破策略示例。

策略思想：
1. 先用过去若干个交易日的日内 open/close/high/low 计算波动区间 range；
2. 当天市场开盘时，根据开盘价和 range 生成上、下突破阈值；
3. 价格突破上阈值时做多，跌破下阈值时做空；
4. 如果盘中从多头反转为空头，或从空头反转为多头，则直接反手；
5. 到日内交易结束时间，清空所有仓位。

这份脚本使用项目内 data/gbpusd.csv 的 1 分钟 GBP/USD 数据。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_price_data(data_path='data/gbpusd.csv'):
    """
    读取本地 1 分钟价格数据。

    数据需要至少包含两列：
    date  ：时间
    price ：价格
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f'找不到数据文件：{path}')

    df = pd.read_csv(path, encoding='utf-8-sig')

    required_cols = {'date', 'price'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'数据缺少必要列：{sorted(missing_cols)}')

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date', drop=False).sort_index()

    return df


def min2day(df, column='price', rg=5, start_hour=3, end_hour=12):
    """
    将 1 分钟数据聚合成日内 OHLC，并计算 Dual Thrust 的历史波动区间。

    注意：
    range 必须用过去 rg 个交易日计算，不能使用当天完整 high/low，
    否则会产生前视偏差。
    """

    rows = []

    for day, group in df.groupby(df.index.normalize()):
        intraday = group.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')

        if intraday.empty:
            continue

        rows.append({
            'date': day,
            'open': intraday[column].iloc[0],
            'close': intraday[column].iloc[-1],
            'high': intraday[column].max(),
            'low': intraday[column].min(),
        })

    intraday = pd.DataFrame(rows)

    if intraday.empty:
        raise ValueError('无法从分钟数据中生成日内 OHLC。')

    intraday = intraday.set_index('date')

    previous_high = intraday['high'].shift(1)
    previous_low = intraday['low'].shift(1)
    previous_close = intraday['close'].shift(1)

    intraday['range1'] = (
        previous_high.rolling(rg).max() -
        previous_close.rolling(rg).min()
    )
    intraday['range2'] = (
        previous_close.rolling(rg).max() -
        previous_low.rolling(rg).min()
    )
    intraday['range'] = np.maximum(intraday['range1'], intraday['range2'])

    return intraday


def signal_generation(
    df,
    intraday,
    param=0.5,
    column='price',
    start_hour=3,
    end_hour=12,
):
    """
    根据 Dual Thrust 阈值生成交易信号。

    param:
        突破参数。常见取值为 0.5，表示上下方向各使用一半 range。

    signals:
        1  表示买入方向的交易；
       -1  表示卖出方向的交易；
        2  表示从空头直接反手为多头；
       -2  表示从多头直接反手为空头。

    cumsum:
        当前仓位。1 是多头，-1 是空头，0 是空仓。
    """

    if not 0 < param < 1:
        raise ValueError('param 应该在 0 和 1 之间。')

    signals = df.copy()
    signals['signals'] = 0
    signals['cumsum'] = 0
    signals['upper'] = 0.0
    signals['lower'] = 0.0

    sigup = None
    siglo = None
    position = 0

    for current_time in signals.index:
        current_day = current_time.normalize()
        price = signals.at[current_time, column]
        signal = 0

        # 开盘时根据当日开盘价和过去 rg 天 range 设置上下阈值。
        if current_time.hour == start_hour and current_time.minute == 0:
            day_range = (
                intraday.at[current_day, 'range']
                if current_day in intraday.index
                else np.nan
            )

            if not pd.isna(day_range):
                sigup = price + param * day_range
                siglo = price - (1 - param) * day_range
            else:
                sigup = None
                siglo = None

        # 日内交易结束，清空所有仓位并重置阈值。
        if current_time.hour == end_hour and current_time.minute == 0:
            if position != 0:
                signal = -position
                position = 0

            sigup = None
            siglo = None

        # 交易时段内，价格突破阈值时开仓或反手。
        elif sigup is not None and siglo is not None:
            if price > sigup and position != 1:
                desired_position = 1
                signal = desired_position - position
                position = desired_position

            elif price < siglo and position != -1:
                desired_position = -1
                signal = desired_position - position
                position = desired_position

        signals.at[current_time, 'signals'] = signal
        signals.at[current_time, 'cumsum'] = position
        signals.at[current_time, 'upper'] = sigup if sigup is not None else 0.0
        signals.at[current_time, 'lower'] = siglo if siglo is not None else 0.0

    return signals


def plot(signals, intraday, column='price'):
    """
    绘制某个发生交易的交易日，展示价格、上下阈值和交易信号。
    """

    output_dir = Path(__file__).resolve().parent / 'Dual Thrust'
    output_dir.mkdir(exist_ok=True)

    trade_days = signals.index[signals['signals'] != 0].normalize().unique()

    if len(trade_days) > 0:
        date = trade_days[-1]
    else:
        valid_days = intraday.index[~intraday['range'].isna()]
        if len(valid_days) == 0:
            print('没有足够数据绘图。')
            return
        date = valid_days[-1]

    start = pd.Timestamp(date) + pd.Timedelta(hours=2)
    end = pd.Timestamp(date) + pd.Timedelta(hours=13)
    signew = signals.loc[start:end]

    if signew.empty:
        print('选定日期没有可绘制的数据。')
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.plot(signew.index, signew[column], label=column)

    threshold_mask = signew['upper'] != 0
    ax.fill_between(
        signew.index[threshold_mask],
        signew.loc[threshold_mask, 'upper'],
        signew.loc[threshold_mask, 'lower'],
        alpha=0.2,
        color='#355c7d',
    )

    long_mask = signew['signals'] > 0
    short_mask = signew['signals'] < 0

    ax.plot(
        signew.index[long_mask],
        signew.loc[long_mask, column],
        lw=0,
        marker='^',
        markersize=10,
        c='g',
        label='LONG',
    )
    ax.plot(
        signew.index[short_mask],
        signew.loc[short_mask, column],
        lw=0,
        marker='v',
        markersize=10,
        c='r',
        label='SHORT',
    )

    legend_texts = ax.legend(loc='best').get_texts()
    for text in legend_texts:
        text.set_color('#6C5B7B')

    opening_time = pd.Timestamp(date) + pd.Timedelta(hours=3)
    if opening_time in signew.index:
        ax.text(
            opening_time,
            signew.at[opening_time, 'upper'],
            'Upper Bound',
            color='#C06C84',
        )
        ax.text(
            opening_time,
            signew.at[opening_time, 'lower'],
            'Lower Bound',
            color='#C06C84',
        )

    ax.set_ylabel(column)
    ax.set_xlabel('Date')
    ax.set_title('Dual Thrust')
    ax.grid(True)
    fig.savefig(
        output_dir / f'{pd.Timestamp(date).strftime("%Y-%m-%d")}_dual_thrust.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


def main():
    """
    读取数据、计算 Dual Thrust 阈值、生成信号并绘图。
    """

    rg = 5
    param = 0.5
    column = 'price'

    df = load_price_data()
    intraday = min2day(df, column=column, rg=rg)
    signals = signal_generation(
        df,
        intraday,
        param=param,
        column=column,
    )
    plot(signals, intraday, column)


# 绩效统计可以参考项目中的 Heikin-Ashi backtest.py。


if __name__ == '__main__':
    main()
