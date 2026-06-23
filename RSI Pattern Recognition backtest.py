# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf


# RSI, Relative Strength Index，相对强弱指数。
# 这份脚本包含两类 RSI 用法：
# 1. 普通的 overbought / oversold 超买超卖策略；
# 2. 在 RSI 序列上识别 Head-and-Shoulders 头肩顶形态。
#
# 注意：
# 技术指标规则只是经验规则，不代表天然有效。
# 真正使用前需要做样本外回测，并考虑手续费、滑点和数据偏差。


def _normalize_yfinance_columns(df, ticker=None):
    """
    yfinance 在某些版本或某些参数下会返回 MultiIndex columns。
    这里把它整理成普通的一层列名，方便后续统一使用 df['Close']。
    """

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        # 常见情况一：列结构类似 ('Close', 'AAPL')
        if 'Close' in data.columns.get_level_values(0):
            if ticker is not None and ticker in data.columns.get_level_values(-1):
                data = data.xs(ticker, level=-1, axis=1)
            else:
                data.columns = data.columns.get_level_values(0)

        # 常见情况二：列结构类似 ('AAPL', 'Close')
        elif 'Close' in data.columns.get_level_values(-1):
            if ticker is not None and ticker in data.columns.get_level_values(0):
                data = data.xs(ticker, level=0, axis=1)
            else:
                data.columns = data.columns.get_level_values(-1)

    return data


def _output_dir():
    output_dir = Path(__file__).resolve().parent / 'RSI'
    output_dir.mkdir(exist_ok=True)

    return output_dir


# Smoothed Moving Average, SMMA。
# Wilder 版 RSI 使用的平滑方法本质上就是：
# 第一个平滑值 = 前 n 个数的简单平均；
# 之后的平滑值 = (上一期平滑值 * (n - 1) + 当前值) / n。
def smma(series, n):
    values = pd.Series(series, dtype='float64').copy()
    output = pd.Series(np.nan, index=values.index, dtype='float64')

    if n <= 0:
        raise ValueError('n 必须是正整数')

    if len(values) < n:
        return output

    # 第一个 SMMA 值用前 n 个值的简单平均初始化
    output.iloc[n - 1] = values.iloc[:n].mean()

    # 后续用 Wilder smoothing 递推
    for i in range(n, len(values)):
        output.iloc[i] = (output.iloc[i - 1] * (n - 1) + values.iloc[i]) / n

    return output


# 计算 RSI。
# RSI 的核心步骤：
# 1. 计算价格变化 delta；
# 2. 把上涨部分记为 gain，把下跌部分的绝对值记为 loss；
# 3. 分别对 gain 和 loss 做 Wilder smoothing；
# 4. RS = 平均上涨 / 平均下跌；
# 5. RSI = 100 - 100 / (1 + RS)。
def rsi(data, n=14):
    close = pd.Series(data, dtype='float64').copy()

    if n <= 0:
        raise ValueError('n 必须是正整数')

    delta = close.diff()

    gain = delta.clip(lower=0).dropna()
    loss = (-delta.clip(upper=0)).dropna()

    avg_gain = smma(gain, n).reindex(close.index)  # 每一个值代表过去 n 天的平均上涨和下跌
    avg_loss = smma(loss, n).reindex(close.index)

    rs = avg_gain / avg_loss   # 相对强度
    rsi = 100 - 100 / (1 + rs)  # 把 RS 这个从 0 到无穷大的数，压缩到 0 到 100 之间, 随 rs 单增
    # RS = 1， RSI = 50， 上涨和下跌力量一样
    # RSI 接近 100：近期上涨力量远强于下跌力量
    # RSI 接近 0：近期下跌力量远强于上涨力量
    # RSI 接近 50：上涨和下跌相对均衡

    # 特殊情况处理：
    # 平均下跌为 0 且平均上涨大于 0，RSI 记为 100；
    # 平均上涨和平均下跌都为 0，说明价格没有变化，RSI 记为 50。
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return rsi


# 普通 RSI 超买/超卖信号生成。
#
# positions 是目标仓位：
# 1  表示做多；
# -1 表示做空；
# 0  表示空仓。
#
# signals 是交易动作：
# 正数表示买入方向的交易；
# 负数表示卖出方向的交易；
# 0 表示不交易。
#
# 注意：
# 如果从 -1 直接变成 1，signals 会等于 2，
# 含义是先平空，再开多；
# 如果从 1 直接变成 -1，signals 会等于 -2，
# 含义是先平多，再开空。
def signal_generation(df, method=rsi, n=14, lower=30, upper=70):
    data = df.copy()
    data = _normalize_yfinance_columns(data)

    if 'Close' not in data.columns:
        raise ValueError("输入数据中必须包含 'Close' 列")

    data['rsi'] = method(data['Close'], n=n)

    # RSI 低于 lower 视为超卖，给出做多目标仓位；
    # RSI 高于 upper 视为超买，给出做空目标仓位；
    # 其他区域保持空仓。
    data['positions'] = np.select(
        [data['rsi'] < lower, data['rsi'] > upper],
        [1, -1],
        default=0
    )

    data['positions'] = pd.Series(data['positions'], index=data.index, dtype='int64')

    # diff 表示仓位变化，也就是实际交易动作。
    data['signals'] = data['positions'].diff().fillna(data['positions'])

    return data.iloc[n:].copy()


# 绘制普通 RSI 超买/超卖策略。
# 上图是 Close 价格和交易点；
# 下图是 RSI，以及 30/70 超卖/超买区间。
def plot(new, ticker='ticker', lower=30, upper=70):
    data = new.copy()

    if data.empty:
        print('数据为空，无法绘图。')
        return

    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True
    )

    # 价格图
    data['Close'].plot(ax=ax, label=ticker)

    buy_mask = data['signals'] > 0
    sell_mask = data['signals'] < 0

    ax.plot(
        data.index[buy_mask],
        data.loc[buy_mask, 'Close'],
        label='BUY / LONG / COVER',
        linewidth=0,
        marker='^',
        color='g'
    )

    ax.plot(
        data.index[sell_mask],
        data.loc[sell_mask, 'Close'],
        label='SELL / SHORT',
        linewidth=0,
        marker='v',
        color='r'
    )

    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title('RSI Positions')
    ax.set_xlabel('Date')
    ax.set_ylabel('price')

    # RSI 图
    data['rsi'].plot(ax=bx, label='RSI', color='#522e75')
    bx.fill_between(
        data.index,
        lower,
        upper,
        alpha=0.5,
        color='#f22f08',
        label='normal range'
    )

    # 避免样本长度小于 45 时 index[-45] 报错
    text_idx = data.index[-45] if len(data) >= 45 else data.index[0]

    bx.text(text_idx, upper + 5, 'overbought', color='#594346', size=12.5)
    bx.text(text_idx, lower - 5, 'oversold', color='#594346', size=12.5)

    bx.set_xlabel('Date')
    bx.set_ylabel('value')
    bx.set_title('RSI')
    bx.legend(loc='best')
    bx.grid(True)

    plt.tight_layout()
    fig.savefig(
        _output_dir() / f'{ticker}_rsi_positions.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


# 在 RSI 序列上识别 Head-and-Shoulders 头肩顶形态。
#
# 这个函数识别的是 RSI 上的头肩顶，而不是价格上的头肩顶。
# 原代码注释说是在 RSI 上识别形态，但实际用 Close 做判断；
# 这里修正为使用 RSI。
#
# 策略含义：
# 出现 RSI 头肩顶时，认为动能可能转弱，给出做空信号 -1；
# 之后如果 RSI 从入场点下降超过 exit_rsi，或者持仓超过 exit_days，
# 则用信号 1 平空。
def pattern_recognition(
    df,
    method=rsi,
    lag=14,
    period=25,
    delta=0.2,
    head=1.1,
    shoulder=1.1,
    exit_rsi=4,
    exit_days=5
):
    data = df.copy()
    data = _normalize_yfinance_columns(data)

    if 'Close' not in data.columns:
        raise ValueError("输入数据中必须包含 'Close' 列")

    data['rsi'] = method(data['Close'], n=lag)
    data['signals'] = 0
    data['cumsum'] = 0
    data['coordinates'] = ''

    rsi_values = data['rsi'].to_numpy(dtype='float64')

    # position = 0 表示空仓；
    # position = -1 表示持有空头仓位。
    position = 0
    entry_rsi = np.nan
    holding_days = 0

    start_i = period + lag

    for i in range(start_i, len(data)):
        signal = 0
        coordinate_text = ''

        current_rsi = rsi_values[i]

        if np.isnan(current_rsi):
            data.at[data.index[i], 'cumsum'] = position
            continue

        pattern_found = False

        # ------------------------------------------------------------
        # 识别 RSI 头肩顶形态
        # ------------------------------------------------------------
        #
        # 节点顺序为：
        # m -> n -> l -> j -> k -> o -> i
        #
        # j：Head，头部，是回看窗口内的最高点；
        # n：左肩，应该低于头部，但高于 neckline 附近低点；
        # o：右肩，应该和左肩高度接近；
        # m、l、k、i：neckline 附近的低点或确认点。
        #
        # 这里用 delta 判断“两个 RSI 数值是否足够接近”。
        # ------------------------------------------------------------
        if position == 0:
            window_start = i - period
            window_end = i

            window = rsi_values[window_start:window_end]

            # 如果窗口内还有 NaN，则跳过，避免形态识别不稳定
            if not np.isnan(window).any():
                # 当前点不能是窗口内最高点；
                # 否则它更像新的高点，而不是头肩顶右侧确认点。
                if current_rsi < np.max(window):
                    j = window_start + int(np.argmax(window))  # Head
                    bottom = current_rsi

                    # 头部必须显著高于当前 neckline 附近点
                    if rsi_values[j] - bottom > head * delta:

                        k = None
                        l = None
                        m = None
                        n_idx = None
                        o = None
                        top = None

                        # 在 Head 和当前点之间寻找一个 neckline 附近点 k
                        for idx in range(j + 1, i):
                            if abs(rsi_values[idx] - bottom) < delta:
                                k = idx
                                break

                        # 在 Head 左侧寻找另一个 neckline 附近点 l
                        if k is not None:
                            for idx in range(j - 1, window_start - 1, -1):
                                if abs(rsi_values[idx] - bottom) < delta:
                                    l = idx
                                    break

                        # 在窗口左端到 l 之间寻找 neckline 附近点 m
                        if l is not None:
                            for idx in range(window_start, l):
                                if abs(rsi_values[idx] - bottom) < delta:
                                    m = idx
                                    break

                        # 在 m 和 l 之间寻找左肩 n
                        if m is not None and l is not None and m + 1 < l:
                            segment = rsi_values[m:l]
                            if not np.isnan(segment).any():
                                n_idx = m + int(np.argmax(segment))

                                # 左肩需要显著高于 neckline，
                                # 同时显著低于 Head。
                                if (
                                    rsi_values[n_idx] - bottom > shoulder * delta and
                                    rsi_values[j] - rsi_values[n_idx] > shoulder * delta
                                ):
                                    top = rsi_values[n_idx]
                                else:
                                    n_idx = None

                        # 在 k 和当前点之间寻找右肩 o
                        # 右肩高度应该接近左肩高度。
                        if n_idx is not None and k is not None:
                            for idx in range(k + 1, i):
                                if abs(rsi_values[idx] - top) < delta:
                                    o = idx
                                    break

                        if (
                            m is not None and
                            n_idx is not None and
                            l is not None and
                            k is not None and
                            o is not None
                        ):
                            signal = -1
                            position = -1
                            entry_rsi = current_rsi
                            holding_days = 0
                            coordinate_text = f'{m},{n_idx},{l},{j},{k},{o},{i}'
                            pattern_found = True

        # ------------------------------------------------------------
        # 平空逻辑
        # ------------------------------------------------------------
        #
        # 如果已经持有空头：
        # 1. RSI 从入场点下降超过 exit_rsi，视为达到动能回落目标，平空；
        # 2. 或者持仓超过 exit_days，也平空。
        #
        # pattern_found 用来避免同一根 K 线刚开空又立刻平空。
        # ------------------------------------------------------------
        if position == -1 and not pattern_found:
            holding_days += 1

            if (entry_rsi - current_rsi >= exit_rsi) or (holding_days >= exit_days):
                signal = 1
                position = 0
                entry_rsi = np.nan
                holding_days = 0

        data.at[data.index[i], 'signals'] = signal
        data.at[data.index[i], 'coordinates'] = coordinate_text
        data.at[data.index[i], 'cumsum'] = position

    return data


# 可视化 RSI Head-and-Shoulders 形态。
# 上图展示价格和交易点；
# 下图展示 RSI、超买/超卖区间，以及识别出来的头肩顶节点。
def pattern_plot(new, ticker='ticker', lower=30, upper=70):
    data = new.copy()

    if data.empty:
        print('数据为空，无法绘图。')
        return

    # 找到第一个带 coordinates 的入场信号
    entry_rows = data[(data['signals'] == -1) & (data['coordinates'] != '')]

    if entry_rows.empty:
        print('没有识别到 RSI Head-and-Shoulders 形态，无法绘图。')
        return

    entry_label = entry_rows.index[0]
    entry_pos = data.index.get_loc(entry_label)

    # 找到入场后的第一个平空信号
    exit_candidates = data.index[(np.arange(len(data)) > entry_pos) & (data['signals'] == 1)]

    if len(exit_candidates) > 0:
        exit_label = exit_candidates[0]
        exit_pos = data.index.get_loc(exit_label)
    else:
        exit_pos = min(len(data) - 1, entry_pos + 20)

    # 解析形态节点坐标
    coordinate_text = data.at[entry_label, 'coordinates']
    pattern_positions = [int(x) for x in coordinate_text.split(',')]
    pattern_index = data.index[pattern_positions]

    # 截取一小段数据，让图更清楚
    start_pos = max(0, pattern_positions[0] - 30)
    end_pos = min(len(data), exit_pos + 20)
    view = data.iloc[start_pos:end_pos].copy()

    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True
    )

    # 价格图
    view['Close'].plot(ax=ax, label=ticker)

    cover_mask = view['signals'] > 0
    short_mask = view['signals'] < 0

    ax.plot(
        view.index[cover_mask],
        view.loc[cover_mask, 'Close'],
        marker='^',
        markersize=12,
        linewidth=0,
        color='g',
        label='COVER'
    )

    ax.plot(
        view.index[short_mask],
        view.loc[short_mask, 'Close'],
        marker='v',
        markersize=12,
        linewidth=0,
        color='r',
        label='SHORT'
    )

    ax.legend(loc='best')
    ax.set_title('Positions')
    ax.set_xlabel('Date')
    ax.set_ylabel('price')
    ax.grid(True)

    # RSI 图
    view['rsi'].plot(ax=bx, label='RSI', color='#f4ed71')

    bx.fill_between(
        view.index,
        lower,
        upper,
        alpha=0.6,
        label='overbought / oversold range',
        color='#000d29'
    )

    bx.plot(
        pattern_index,
        data.loc[pattern_index, 'rsi'],
        linewidth=3,
        alpha=0.7,
        marker='o',
        markersize=6,
        color='#8d2f23',
        label='head-and-shoulders pattern'
    )

    bx.plot(
        view.index[cover_mask],
        view.loc[cover_mask, 'rsi'],
        marker='^',
        markersize=12,
        linewidth=0,
        color='g',
        label='COVER'
    )

    bx.plot(
        view.index[short_mask],
        view.loc[short_mask, 'rsi'],
        marker='v',
        markersize=12,
        linewidth=0,
        color='r',
        label='SHORT'
    )

    # 节点顺序为 m, n, l, j, k, o, i
    # 其中 n 是左肩，j 是头部，o 是右肩。
    for pos, text in [(1, 'Shoulder'), (3, 'Head'), (5, 'Shoulder')]:
        idx = pattern_index[pos]
        bx.text(
            idx,
            data.loc[idx, 'rsi'] + 2,
            text,
            fontsize=10,
            color='#e4ebf2',
            horizontalalignment='center',
            verticalalignment='center'
        )

    bx.set_title('RSI Head-and-Shoulders')
    bx.legend(loc='best')
    bx.set_xlabel('Date')
    bx.set_ylabel('value')
    bx.grid(True)

    plt.tight_layout()
    fig.savefig(
        _output_dir() / f'{ticker}_rsi_head_and_shoulders.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


def main():
    ticker = 'AAPL'
    startdate = '2016-01-01'
    enddate = '2018-01-01'

    df = yf.download(
        ticker,
        start=startdate,
        end=enddate,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(
            'yfinance 没有下载到数据。可以换一个 ticker，或者改成读取本地 CSV。'
        )

    df = _normalize_yfinance_columns(df, ticker=ticker)

    # 普通 RSI 超买/超卖信号
    new = signal_generation(df, rsi, n=14)
    plot(new, ticker)

    # 如果想测试 RSI Head-and-Shoulders 形态，取消下面两行注释：
    pattern_new = pattern_recognition(df, rsi, lag=14)
    pattern_plot(pattern_new, ticker)


if __name__ == '__main__':
    main()
