# coding: utf-8

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# London Breakout 是一个外汇日内突破策略。
#
# 核心想法：
# 1. 外汇市场工作日 24 小时连续交易；
# 2. 伦敦时段通常流动性更强；
# 3. 在伦敦开盘前，用东京时段最后一小时的价格区间作为上下边界；
# 4. 伦敦开盘后的前 30 分钟，如果价格突破上边界就做多，跌破下边界就做空；
# 5. 开仓后用固定止盈/止损或收盘时间退出。
#
# 原始数据来自 histdata.com 的 1 分钟外汇报价。
# 本项目中的 data/gbpusd.csv 已经整理成两列：
# date  ：时间
# price ：bid/ask 平均价
#
# 注意：
# 原始网站时间通常使用纽约时间。这里沿用原脚本假设：
# 02:00-02:59 是伦敦开盘前一小时；
# 03:00 是伦敦开盘；
# 12:00 是伦敦时段结束。


def london_breakout(df):
    """
    初始化策略需要的列。

    signals:
        1  表示开多或平空
       -1  表示开空或平多
        0  表示不交易

    cumsum:
        累计仓位。1 表示持有多头，-1 表示持有空头，0 表示空仓。
    """

    data = df.copy()
    data['signals'] = 0
    data['cumsum'] = 0
    data['upper'] = 0.0
    data['lower'] = 0.0

    return data


def signal_generation(
    df,
    method,
    risky_stop=0.01,
    open_minutes=30,
):
    """
    生成 London Breakout 交易信号。

    risky_stop:
        最大可接受突破距离。比如 0.01 表示价格如果已经离阈值太远，
        就认为追入风险过高，不再开仓。

    open_minutes:
        伦敦开盘后允许触发突破交易的分钟数。
    """

    signals = method(df)
    signals['date'] = pd.to_datetime(signals['date'])

    tokyo_price = []
    upper = None
    lower = None
    executed_price = None
    position = 0

    for i in range(len(signals)):
        current_time = signals.at[i, 'date']
        price = signals.at[i, 'price']
        signal = 0

        # 02:00-02:59：记录伦敦开盘前一小时的价格，用于计算突破上下边界。
        if current_time.hour == 2:
            tokyo_price.append(price)

        # 03:00：用上一小时价格的最高/最低值确定当日突破边界。
        elif current_time.hour == 3 and current_time.minute == 0:
            if tokyo_price:
                upper = max(tokyo_price)
                lower = min(tokyo_price)
                tokyo_price = []

                signals.at[i, 'upper'] = upper
                signals.at[i, 'lower'] = lower

        # 03:00-03:29：伦敦开盘后前 30 分钟，检查是否突破。
        elif current_time.hour == 3 and current_time.minute < open_minutes:
            if upper is not None and lower is not None:
                signals.at[i, 'upper'] = upper
                signals.at[i, 'lower'] = lower

                # 只在空仓时开新仓，避免同一天重复追单。
                if position == 0:
                    if 0 < price - upper <= risky_stop:
                        signal = 1
                        position = 1
                        executed_price = price

                    elif 0 < lower - price <= risky_stop:
                        signal = -1
                        position = -1
                        executed_price = price

        # 12:00：伦敦时段结束，若还有仓位则强制平仓。
        elif current_time.hour == 12:
            if position != 0:
                signal = -position
                position = 0
                executed_price = None

        # 其他时间：如果已有仓位，检查是否达到止盈或止损。
        else:
            if position != 0 and executed_price is not None:
                upper_exit = executed_price + risky_stop / 2
                lower_exit = executed_price - risky_stop / 2

                if price > upper_exit or price < lower_exit:
                    signal = -position
                    position = 0
                    executed_price = None

        signals.at[i, 'signals'] = signal
        signals.at[i, 'cumsum'] = position

    return signals


def plot(new):
    """
    绘制两张图：
    1. 当日完整价格走势和交易信号；
    2. 伦敦开盘附近的突破细节。
    """

    if new.empty:
        print('没有可绘制的数据。')
        return

    output_dir = Path(__file__).resolve().parent / 'London Breakout'
    output_dir.mkdir(exist_ok=True)

    # 第一张图：全天价格与交易信号。
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['price'].plot(ax=ax, label='price')

    long_mask = new['signals'] == 1
    short_mask = new['signals'] == -1

    ax.plot(
        new.index[long_mask],
        new.loc[long_mask, 'price'],
        lw=0,
        marker='^',
        c='g',
        label='LONG',
    )
    ax.plot(
        new.index[short_mask],
        new.loc[short_mask, 'price'],
        lw=0,
        marker='v',
        c='r',
        label='SHORT',
    )

    date = new.index[0].strftime('%Y-%m-%d')
    ax.axvline(pd.Timestamp(f'{date} 03:00:00'), linestyle=':', c='k')
    ax.axvline(pd.Timestamp(f'{date} 12:00:00'), linestyle=':', c='k')

    ax.legend(loc='best')
    ax.set_title('London Breakout')
    ax.set_ylabel('price')
    ax.set_xlabel('Date')
    ax.grid(True)
    fig.savefig(
        output_dir / f'{date}_london_breakout.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)

    # 第二张图：伦敦开盘附近的突破和阈值。
    start = f'{date} 02:50:00'
    end = f'{date} 03:30:00'
    opening = new.loc[start:end]

    if opening.empty:
        print('伦敦开盘附近没有可绘制的数据。')
        return

    fig = plt.figure()
    bx = fig.add_subplot(111)

    opening_long = opening['signals'] == 1
    opening_short = opening['signals'] == -1

    bx.plot(
        opening.index[opening_long],
        opening.loc[opening_long, 'price'],
        lw=0,
        marker='^',
        markersize=10,
        c='g',
        label='LONG',
    )
    bx.plot(
        opening.index[opening_short],
        opening.loc[opening_short, 'price'],
        lw=0,
        marker='v',
        markersize=10,
        c='r',
        label='SHORT',
    )

    # 只绘制非零阈值；0 表示当前行不处于开盘突破观察区间。
    upper_mask = opening['upper'] != 0
    lower_mask = opening['lower'] != 0

    bx.plot(
        opening.index[upper_mask],
        opening.loc[upper_mask, 'upper'],
        lw=0,
        marker='.',
        markersize=7,
        c='#BC8F8F',
        label='upper threshold',
    )
    bx.plot(
        opening.index[lower_mask],
        opening.loc[lower_mask, 'lower'],
        lw=0,
        marker='.',
        markersize=5,
        c='#FF4500',
        label='lower threshold',
    )
    bx.plot(opening.index, opening['price'], label='price')

    bx.grid(True)
    bx.set_ylabel('price')
    bx.set_xlabel('time interval')
    bx.set_xticks([])
    bx.set_title(f'{date} Market Opening')
    bx.legend(loc='best')
    fig.savefig(
        output_dir / f'{date}_market_opening.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)


def main(data_path='data/gbpusd.csv'):
    """
    读取本地 GBP/USD 1 分钟数据，生成信号，并绘制第一天的回测示例。
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f'找不到数据文件：{path}')

    df = pd.read_csv(path, encoding='utf-8-sig')

    required_cols = {'date', 'price'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'数据缺少必要列：{sorted(missing_cols)}')

    signals = signal_generation(df, london_breakout)
    signals = signals.set_index(pd.to_datetime(signals['date']))

    first_date = signals.index[0].strftime('%Y-%m-%d')
    one_day = signals.loc[first_date].copy()

    plot(one_day)


# 绩效统计可以参考项目中的 Heikin-Ashi backtest.py。


if __name__ == '__main__':
    main()
