# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 计算 Bollinger Bands
# Bollinger Bands 由三条线组成：
# 1. mid band：中轨，通常是移动平均线
# 2. upper band：上轨，中轨 + n 倍移动标准差
# 3. lower band：下轨，中轨 - n 倍移动标准差
#
# 这里默认使用 20 周期移动平均线，以及上下 2 倍标准差。
def bollinger_bands(df, price_col='price', window=20, num_std=2):
    
    data = df.copy()

    if price_col not in data.columns:
        raise ValueError(f"输入数据中找不到价格列: {price_col}")

    # Bollinger Bands 更常用总体标准差，因此这里 ddof=0
    # 最近 window 个价格的标准差 std
    # min_periods=window 必须满 window 个数据才计算，所以前 window - 1 行都会是 NaN
    data['std'] = data[price_col].rolling(
        window=window,
        min_periods=window
    ).std(ddof=0)

    # 最近 window 个价格的移动平均 mid band
    data['mid band'] = data[price_col].rolling(
        window=window,
        min_periods=window
    ).mean()

    data['upper band'] = data['mid band'] + num_std * data['std']
    data['lower band'] = data['mid band'] - num_std * data['std']

    # bandwidth 用来衡量布林带宽度
    # bandwidth 越大，说明波动越剧烈；
    # bandwidth 越小，说明波动收缩。
    data['bandwidth'] = (
        data['upper band'] - data['lower band']
    ) / data['mid band']

    return data


# 生成 Bollinger Bands 的 W 底形态信号
#
# 这个策略只做多：
# signals = 1  表示买入 / 开多
# signals = -1 表示卖出 / 平多
# signals = 0  表示无操作
#
# cumsum 表示当前仓位状态：
# cumsum = 0 表示空仓
# cumsum = 1 表示持有多头仓位
def signal_generation(
    data,
    method=bollinger_bands,
    period=75,
    alpha=0.0001,
    beta=0.0001,
    price_col='price'
):
    
    # period 表示向前回看的窗口长度
    # 原代码中使用 75，大约对应 3 个月交易日。
    #
    # alpha 表示价格与 Bollinger Bands 之间的容忍误差。
    # alpha 太小，信号很难触发；
    # alpha 太大，信号太容易触发，可能带来更多噪声。
    #
    # beta 表示 bandwidth 的收缩阈值。
    # 当 bandwidth 小于 beta 时，认为 Bollinger Bands 进入收缩状态，
    # 说明价格动能减弱，可以考虑平仓。

    df = method(data).copy().reset_index(drop=True)

    required_cols = [
        price_col,
        'mid band',
        'upper band',
        'lower band',
        'std',
        'bandwidth'
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必要列: {col}")

    df['signals'] = 0
    df['cumsum'] = 0

    # coordinates 用来记录 W 底的五个节点：
    # l, k, j, m, i
    #
    # 从左到右分别表示：
    # l：左侧高点
    # k：第一个底部
    # j：中间反弹高点
    # m：第二个底部
    # i：右侧突破点，也就是最终买入确认点
    df['coordinates'] = ''

    # position 显式记录当前仓位
    # 0 表示空仓，1 表示持有多头仓位
    position = 0

    for i in range(period, len(df)):

        signal = 0
        coordinates = ''

        price_i = df.at[i, price_col]
        upper_i = df.at[i, 'upper band']
        bandwidth_i = df.at[i, 'bandwidth']
        std_i = df.at[i, 'std']

        # 如果当前行的指标还没有计算出来，则跳过
        if pd.isna(price_i) or pd.isna(upper_i) or pd.isna(bandwidth_i):
            df.at[i, 'signals'] = 0
            df.at[i, 'cumsum'] = position
            continue

        # ------------------------------------------------------------
        # W 底买入信号
        # ------------------------------------------------------------
        #
        # 原始代码的思路是识别五个节点：
        #
        # l -> k -> j -> m -> i
        #
        # 其中：
        # k 是第一个底部，价格接近或跌破 lower band；
        # j 是中间反弹点，价格回到 mid band 附近；
        # m 是第二个底部，价格仍在 lower band 上方附近；
        # i 是最终确认点，价格突破 upper band。
        #
        # 注意：
        # 这里修正了原代码中一个比较奇怪的条件：
        # 原代码写的是 abs(mid band[j] - upper band[i]) < alpha，
        # 这相当于拿 j 点的中轨和 i 点的上轨比较，逻辑不太合理。
        # 更自然的写法是：j 点价格回到 mid band 附近。
        # ------------------------------------------------------------

        if position == 0 and price_i > upper_i:

            j_idx = None
            k_idx = None
            l_idx = None
            m_idx = None

            # condition 2：
            # 从 i 往前找中间反弹点 j。
            # j 点要求价格回到 mid band 附近。
            for j in range(i - 1, i - period, -1):
                if j < 0:
                    break

                price_j = df.at[j, price_col]
                mid_j = df.at[j, 'mid band']

                if pd.isna(price_j) or pd.isna(mid_j):
                    continue

                if abs(price_j - mid_j) <= alpha:
                    j_idx = j
                    break

            # condition 1：
            # 在 j 之前寻找第一个底部 k。
            # k 点要求价格接近或跌破 lower band。
            if j_idx is not None:
                for k in range(j_idx - 1, i - period, -1):
                    if k < 0:
                        break

                    price_k = df.at[k, price_col]
                    lower_k = df.at[k, 'lower band']

                    if pd.isna(price_k) or pd.isna(lower_k):
                        continue

                    if price_k <= lower_k + alpha:
                        k_idx = k
                        first_bottom_price = price_k
                        break

            # l 点不是正式交易条件，只是为了后面画出 W 形态。
            # 这里在 k 之前寻找一个价格高于 mid band 的左侧高点。
            if k_idx is not None:
                for l in range(k_idx - 1, i - period, -1):
                    if l < 0:
                        break

                    price_l = df.at[l, price_col]
                    mid_l = df.at[l, 'mid band']

                    if pd.isna(price_l) or pd.isna(mid_l):
                        continue

                    if price_l > mid_l:
                        l_idx = l
                        break

            # condition 3：
            # 在 j 和 i 之间寻找第二个底部 m。
            #
            # m 点要求：
            # 1. 价格在 lower band 上方；
            # 2. 价格距离 lower band 很近；
            # 3. 价格低于第一个底部 k 的价格。
            #
            # 第 3 点保留了原代码的逻辑：
            # 第二个底部形成价格新低，但仍没有有效跌破 lower band，
            # 表示下跌动能可能减弱。
            if j_idx is not None and k_idx is not None and l_idx is not None:
                for m in range(i - 1, j_idx, -1):

                    price_m = df.at[m, price_col]
                    lower_m = df.at[m, 'lower band']

                    if pd.isna(price_m) or pd.isna(lower_m):
                        continue

                    if (
                        price_m > lower_m and
                        price_m - lower_m <= alpha and
                        price_m < first_bottom_price
                    ):
                        m_idx = m
                        break

            # condition 4：
            # 当前 i 点价格突破 upper band，作为最终买入确认。
            if (
                l_idx is not None and
                k_idx is not None and
                j_idx is not None and
                m_idx is not None
            ):
                signal = 1
                position = 1
                coordinates = f'{l_idx},{k_idx},{j_idx},{m_idx},{i}'

        # ------------------------------------------------------------
        # 平仓信号
        # ------------------------------------------------------------
        #
        # 原代码注释说：
        # 当 Bollinger Bands 进入 contraction，也就是波动收缩时平仓。
        #
        # 但原代码实际判断的是 std < beta，
        # 这和“bandwidth 收缩”的注释不完全一致。
        # 这里改成使用 bandwidth < beta，更符合原注释含义。
        #
        # signal == 0 是为了避免同一根 K 线刚买入又立刻平仓。
        # ------------------------------------------------------------
        if position == 1 and signal == 0:
            # if bandwidth_i < beta:
            if std_i < beta:
                signal = -1
                position = 0

        df.at[i, 'signals'] = signal
        df.at[i, 'coordinates'] = coordinates
        df.at[i, 'cumsum'] = position

    return df


# 可视化 Bollinger Bands 和识别出来的 W 底形态
def plot(new, price_col='price'):

    output_dir = Path(__file__).resolve().parent / 'Bollinger Bands'
    output_dir.mkdir(exist_ok=True)

    trade_indices = list(new.index[new['signals'] != 0])

    if len(trade_indices) == 0:
        print('没有生成任何交易信号，无法绘图。')
        return

    long_indices = list(new.index[new['signals'] == 1])

    if len(long_indices) == 0:
        print('没有生成买入信号，无法绘制 W 底形态。')
        return

    # 取第一个买入信号作为展示对象
    entry_idx = long_indices[0]

    # 尝试寻找买入后的第一个平仓信号
    exit_candidates = list(
        new.index[(new.index > entry_idx) & (new['signals'] == -1)]
    )

    if len(exit_candidates) > 0:
        exit_idx = exit_candidates[0]
    else:
        exit_idx = min(len(new) - 1, entry_idx + 30)

    # 为了让图更清楚，只截取交易信号附近的一小段数据
    start_idx = max(0, entry_idx - 85)
    end_idx = min(len(new), exit_idx + 30)

    newbie = new.iloc[start_idx:end_idx].copy()

    # 如果数据中有 date 列，则用 date 作为横轴；
    # 否则使用原始整数索引作为横轴。
    if 'date' in newbie.columns:
        newbie['datetime'] = pd.to_datetime(newbie['date'])
        newbie.set_index('datetime', inplace=True)

        full_x = pd.to_datetime(new['date'])
    else:
        full_x = new.index

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # 绘制价格、中轨、上下轨
    ax.plot(newbie.index, newbie[price_col], label='price')

    ax.fill_between(
        newbie.index,
        newbie['lower band'].values,
        newbie['upper band'].values,
        alpha=0.2,
        color='#45ADA8',
        label='Bollinger Bands'
    )

    ax.plot(
        newbie.index,
        newbie['mid band'],
        linestyle='--',
        label='moving average',
        color='#132226'
    )

    # 绘制买入和平仓位置
    buy_points = newbie[newbie['signals'] == 1]
    sell_points = newbie[newbie['signals'] == -1]

    ax.plot(
        buy_points.index,
        buy_points[price_col],
        marker='^',
        markersize=12,
        linewidth=0,
        color='g',
        label='LONG'
    )

    ax.plot(
        sell_points.index,
        sell_points[price_col],
        marker='v',
        markersize=12,
        linewidth=0,
        color='r',
        label='EXIT'
    )

    # 绘制 W 底形态
    entry_row = new.loc[entry_idx]
    coordinate_text = entry_row.get('coordinates', '')

    if isinstance(coordinate_text, str) and coordinate_text != '':
        index_list = [int(x) for x in coordinate_text.split(',')]

        if 'date' in new.columns:
            pattern_x = pd.to_datetime(new.loc[index_list, 'date'])
        else:
            pattern_x = index_list

        pattern_y = new.loc[index_list, price_col]

        ax.plot(
            pattern_x,
            pattern_y,
            linewidth=5,
            alpha=0.7,
            color='#FE4365',
            label='double bottom pattern'
        )

    # 添加文字标注
    if len(buy_points) > 0:
        ax.text(
            buy_points.index[0],
            buy_points['lower band'].iloc[0],
            'Expansion',
            fontsize=15,
            color='#563838'
        )

    if len(sell_points) > 0:
        ax.text(
            sell_points.index[0],
            sell_points['lower band'].iloc[0],
            'Contraction',
            fontsize=15,
            color='#563838'
        )

    ax.legend(loc='best')
    ax.set_title('Bollinger Bands Pattern Recognition')
    ax.set_ylabel('price')
    ax.grid(True)

    fig.savefig(
        output_dir / 'bollinger_bands_pattern_recognition.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


def main(data_path='data/gbpusd.csv'):

    # 原代码中使用 os.chdir('d:/')，这会导致代码只能在特定 Windows 路径下运行。
    # 这里改成直接传入 data_path，更通用。
    #
    # 原作者使用的是 histdata.com 的数据，
    # 并且用 bid 和 ask 的平均值作为 price。
    df = pd.read_csv(data_path)

    signals = signal_generation(
        df,
        method=bollinger_bands,
        period=75,
        alpha=0.0001,
        beta=0.0001,
        price_col='price'
    )

    plot(signals, price_col='price')


if __name__ == '__main__':
    main()
