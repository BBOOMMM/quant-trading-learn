# -*- coding: utf-8 -*-
"""
Heikin-Ashi 策略示例：指标计算、信号生成、简单回测和绩效统计。

说明：
1. 这份代码是在原脚本基础上修正的版本。
2. 主要修正了 pandas 链式赋值、索引访问、信号累计、蜡烛图颜色判断、回测统计口径不一致等问题。
3. 原策略的核心交易规则基本保留：signals=1 表示加一份多头，signals<0 表示清空已有多头。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yfinance as yf


def _output_dir() -> Path:
    output_dir = Path(__file__).resolve().parent / "Heikin-Ashi candlestick"
    output_dir.mkdir(exist_ok=True)

    return output_dir


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    兼容新版 yfinance 的返回格式。

    yfinance 在某些版本中，即使只下载一个 ticker，也可能返回 MultiIndex 列，
    例如 ('Close', 'NVDA')。这里将其转换为普通列名：Open, High, Low, Close 等。
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    out = df.copy()

    # 常见格式：第一层是价格字段，第二层是 ticker。
    if ticker is not None and ticker in out.columns.get_level_values(-1):
        out = out.xs(ticker, axis=1, level=-1)
    else:
        # 如果无法按 ticker 切分，就保留第一层字段名。
        out.columns = out.columns.get_level_values(0)

    return out


def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    下载 OHLCV 数据。

    auto_adjust=False 是为了明确保留 Open/High/Low/Close/Adj Close/Volume。
    如果下载失败，直接抛出异常，避免后面在空 DataFrame 上报更隐蔽的错误。
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    df = _flatten_yfinance_columns(df, ticker=ticker)

    if df.empty:
        raise ValueError(f"{ticker} 在 {start} 到 {end} 之间没有下载到数据。")

    required_cols = {"Open", "High", "Low", "Close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"下载数据缺少必要列：{sorted(missing_cols)}")

    return df


def heikin_ashi(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Heikin-Ashi，即“平均 K 线”。

    普通 K 线直接使用真实的 Open/High/Low/Close；
    Heikin-Ashi 会对 OHLC 做平滑变换，从而减少短期噪声。

    计算公式：
    HA close = (Open + High + Low + Close) / 4
    HA open  = 前一根 K 线的 (HA open + HA close) / 2
    HA high  = max(High, HA open, HA close)
    HA low   = min(Low, HA open, HA close)
    """
    df = data.copy()
    df = _flatten_yfinance_columns(df)

    # reset_index 后，原来的日期索引会变成 Date 或 Datetime 列。
    df = df.reset_index()
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            df = df.rename(columns={df.columns[0]: "Date"})

    required_cols = ["Open", "High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少必要列：{missing_cols}")

    # Heikin-Ashi 收盘价。
    df["HA close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

    # Heikin-Ashi 开盘价需要递推计算。
    df["HA open"] = np.nan
    df.at[0, "HA open"] = df.at[0, "Open"]
    for n in range(1, len(df)):
        df.at[n, "HA open"] = (df.at[n - 1, "HA open"] + df.at[n - 1, "HA close"]) / 2.0

    # Heikin-Ashi 最高价和最低价。
    ha_cols = ["HA open", "HA close", "High", "Low"]
    df["HA high"] = df[ha_cols].max(axis=1)
    df["HA low"] = df[ha_cols].min(axis=1)

    # 这两个字段在后续策略里没有用到；用 errors="ignore" 避免新版 yfinance 没有 Adj Close 时报错。
    df = df.drop(columns=["Adj Close", "Volume"], errors="ignore")

    return df


def signal_generation(df: pd.DataFrame, method=heikin_ashi, stls: int = 3) -> pd.DataFrame:
    """
    根据 Heikin-Ashi 形态生成交易信号。

    signals 的含义：
    1   ：加一份多头仓位
    0   ：不操作
    < 0 ：清空已有多头仓位，数值大小等于要卖出的仓位份数

    stls 表示最多允许持有多少份多头仓位。
    原代码在循环中反复整体计算 cumsum，且在修改 signals 后没有同步更新当前行 cumsum；
    这里改成用 position 变量逐行维护仓位，逻辑更清楚，也更不容易出错。
    """
    if stls <= 0:
        raise ValueError("stls 必须是正整数。")

    data = method(df).copy()
    data["signals"] = 0
    data["cumsum"] = 0

    position = 0

    for n in range(1, len(data)):
        ha_open = data.at[n, "HA open"]
        ha_close = data.at[n, "HA close"]
        ha_high = data.at[n, "HA high"]
        ha_low = data.at[n, "HA low"]

        prev_ha_open = data.at[n - 1, "HA open"]
        prev_ha_close = data.at[n - 1, "HA close"]

        # 原代码的“开多/加仓”条件：当前和上一根 HA K 线都偏弱，并且当前实体更大。
        # 这更像是逆势抄底规则，而不是典型顺势做多规则；这里保留原策略含义。
        long_condition = (
            ha_open > ha_close
            and ha_open == ha_high
            and abs(ha_open - ha_close) > abs(prev_ha_open - prev_ha_close)
            and prev_ha_open > prev_ha_close
        )

        # 原代码的“退出”条件：出现强势 HA 阳线时，清空已有多头。
        exit_condition = (
            ha_open < ha_close
            and ha_open == ha_low
            and prev_ha_open < prev_ha_close
        )

        signal = 0

        if long_condition and position < stls:
            signal = 1
        elif exit_condition and position > 0:
            signal = -position

        position += signal
        data.at[n, "signals"] = signal
        data.at[n, "cumsum"] = position

    return data


def candlestick(
    df: pd.DataFrame,
    ax=None,
    titlename: str = "",
    highcol: str = "High",
    lowcol: str = "Low",
    opencol: str = "Open",
    closecol: str = "Close",
    xcol: str = "Date",
    colorup: str = "r",
    colordown: str = "g",
    width: float = 0.6,
):
    """
    手动画 K 线图。

    matplotlib 早期版本移除了原来的 candlestick 函数；
    这里用矩形画实体，用竖线画上下影线。

    默认颜色沿用原代码/国内习惯：
    红色表示上涨，绿色表示下跌。
    如果你想使用美股常见配色，可以把 colorup 改成 "g"，colordown 改成 "r"。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if len(df) == 0:
        ax.set_title(titlename)
        return ax

    for i in range(len(df)):
        open_price = float(df[opencol].iloc[i])
        close_price = float(df[closecol].iloc[i])
        high_price = float(df[highcol].iloc[i])
        low_price = float(df[lowcol].iloc[i])

        # 原代码这里写反了：Open > Close 是下跌，Close >= Open 才是上涨。
        bar_color = colorup if close_price >= open_price else colordown

        # 画上下影线。
        ax.plot([i, i], [low_price, high_price], color="k", linewidth=0.8)

        # 画实体。若开盘价等于收盘价，则画一条横线。
        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        if body_height == 0:
            ax.plot(
                [i - width / 2, i + width / 2],
                [close_price, close_price],
                color="k",
                linewidth=0.8,
            )
        else:
            rect = Rectangle(
                (i - width / 2, body_bottom),
                width,
                body_height,
                facecolor=bar_color,
                edgecolor="k",
                linewidth=0.8,
            )
            ax.add_patch(rect)

    # 只显示约 5 个横轴刻度，避免日期太密。
    step = max(1, len(df) // 5)
    tick_locs = list(range(0, len(df), step))
    tick_labels = pd.to_datetime(df[xcol].iloc[tick_locs]).dt.strftime("%Y-%m-%d")

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_title(titlename)
    ax.autoscale_view()

    return ax


def plot(trading_signals: pd.DataFrame, ticker: str):
    """
    画两张图：
    1. Heikin-Ashi K 线图；
    2. 真实收盘价曲线，并标出加仓和清仓位置。
    """
    df = trading_signals.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date", drop=False)

    fig = plt.figure(figsize=(12, 8))

    # 上半部分：Heikin-Ashi K 线。
    ax1 = plt.subplot2grid((200, 1), (0, 0), rowspan=120, ylabel="HA price")
    candlestick(
        df,
        ax=ax1,
        titlename="Heikin-Ashi",
        highcol="HA high",
        lowcol="HA low",
        opencol="HA open",
        closecol="HA close",
        xcol="Date",
        colorup="r",
        colordown="g",
    )
    ax1.grid(True)
    ax1.set_xticklabels([])

    # 下半部分：真实价格和交易信号。
    ax2 = plt.subplot2grid((200, 1), (120, 0), rowspan=80, ylabel="price")
    df["Close"].plot(ax=ax2, label=ticker)

    # signals=1 是加多；signals<0 是清仓，不是真正意义上的做空。
    long_mask = df["signals"] == 1
    exit_mask = df["signals"] < 0

    ax2.plot(
        df.index[long_mask],
        df.loc[long_mask, "Close"],
        marker="^",
        linestyle="None",
        color="g",
        label="long",
    )
    ax2.plot(
        df.index[exit_mask],
        df.loc[exit_mask, "Close"],
        marker="v",
        linestyle="None",
        color="r",
        label="exit",
    )

    ax2.grid(True)
    ax2.legend(loc="best")

    plt.tight_layout()
    fig.savefig(
        _output_dir() / f"{ticker}_heikin_ashi_candlestick.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def portfolio(data: pd.DataFrame, capital0: float = 10000, positions: int = 100) -> pd.DataFrame:
    """
    根据交易信号计算简单组合净值。

    capital0 ：初始资金
    positions：每一份信号对应的股票数量

    这里假设：
    1. 按当天 Close 价格成交；
    2. 不考虑手续费、滑点、印花税；
    3. signals=1 买入 positions 股；
    4. signals<0 卖出对应倍数的 positions 股。
    """
    data = data.copy()

    if "cumsum" not in data.columns:
        data["cumsum"] = data["signals"].cumsum()

    result = pd.DataFrame(index=pd.to_datetime(data["Date"]) if "Date" in data.columns else data.index)

    result["holdings"] = data["cumsum"].to_numpy() * data["Close"].to_numpy() * positions
    result["cash"] = capital0 - (data["signals"].to_numpy() * data["Close"].to_numpy() * positions).cumsum()
    result["total asset"] = result["holdings"] + result["cash"]
    result["return"] = result["total asset"].pct_change()
    result["signals"] = data["signals"].to_numpy()

    return result


def profit(portfolio_details: pd.DataFrame):
    """
    画组合总资产曲线，并标出加仓和清仓位置。
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    portfolio_details["total asset"].plot(ax=ax, label="Total Asset")

    long_mask = portfolio_details["signals"] == 1
    exit_mask = portfolio_details["signals"] < 0

    ax.plot(
        portfolio_details.index[long_mask],
        portfolio_details.loc[long_mask, "total asset"],
        linestyle="None",
        marker="^",
        color="g",
        label="long",
    )
    ax.plot(
        portfolio_details.index[exit_mask],
        portfolio_details.loc[exit_mask, "total asset"],
        linestyle="None",
        marker="v",
        color="r",
        label="exit",
    )

    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Asset Value")
    ax.set_title("Total Asset")

    plt.tight_layout()
    fig.savefig(
        _output_dir() / "total_asset.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    经验 Omega Ratio。

    原代码用 t 分布积分近似，但把真实收益率直接放进 t 分布变量中，解释性较弱。
    这里改为常见的经验计算方式：

    Omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan

    excess = r - threshold
    gain = excess[excess > 0].sum()
    loss = -excess[excess < 0].sum()

    if loss == 0:
        return np.inf

    return float(gain / loss)


def sortino_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    经验 Sortino Ratio。

    Sortino Ratio 只惩罚低于阈值的下行波动：
    Sortino = (mean(return) - threshold) / downside_deviation
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan

    downside = np.minimum(r - threshold, 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))

    if downside_deviation == 0:
        return np.inf

    return float((r.mean() - threshold) / downside_deviation)


def mdd(series: pd.Series) -> float:
    """
    计算最大回撤。

    对每一天，用当前资产值除以历史最高资产值再减 1；
    所有回撤中的最小值就是最大回撤，通常是一个非正数。
    """
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return np.nan

    running_max = s.cummax()
    drawdown = s / running_max - 1.0

    return float(drawdown.min())


def stats(
    portfolio_details: pd.DataFrame,
    trading_signals: pd.DataFrame,
    stdate: Optional[str] = None,
    eddate: Optional[str] = None,
    capital0: float = 10000,
) -> pd.DataFrame:
    """
    计算回测统计指标。

    注意：
    1. portfolio_details 和 trading_signals 应该来自同一段时间；
    2. 原代码在 main 中用截取后的 viz 做 portfolio，却用全量 trading_signals 做 stats，
       这里建议统一使用全量，或者统一使用截取后的数据；
    3. 这里默认用 S&P 500 作为基准，下载失败时相关指标记为 NaN。
    """
    if len(portfolio_details) == 0:
        raise ValueError("portfolio_details 为空，无法统计。")

    total_asset = portfolio_details["total asset"].dropna()
    returns = portfolio_details["return"].dropna()

    if len(total_asset) == 0:
        raise ValueError("total asset 为空，无法统计。")

    total_return = float(total_asset.iloc[-1] / capital0 - 1.0)
    n_periods = max(len(total_asset) - 1, 1)
    growth_rate = float((1.0 + total_return) ** (1.0 / n_periods) - 1.0)

    std = float(returns.std(ddof=0)) if len(returns) > 0 else np.nan
    maximum = float(returns.max()) if len(returns) > 0 else np.nan
    minimum = float(returns.min()) if len(returns) > 0 else np.nan

    benchmark_return = np.nan
    benchmark_growth_rate = 0.0

    if stdate is not None and eddate is not None:
        try:
            benchmark = download_price_data("^GSPC", stdate, eddate)
            benchmark_return = float(benchmark["Close"].iloc[-1] / benchmark["Open"].iloc[0] - 1.0)
            benchmark_growth_rate = float((1.0 + benchmark_return) ** (1.0 / n_periods) - 1.0)
        except Exception as exc:
            warnings.warn(f"基准数据下载或计算失败，benchmark 指标将记为 NaN。原因：{exc}")
            benchmark_return = np.nan
            benchmark_growth_rate = 0.0

    sharpe = (
        (growth_rate - benchmark_growth_rate) / std
        if std is not None and not np.isnan(std) and std != 0
        else np.nan
    )

    max_drawdown = mdd(total_asset)

    # 最大回撤是负数，Calmar Ratio 通常用 abs(max_drawdown) 做分母。
    calmar = growth_rate / abs(max_drawdown) if max_drawdown not in [0, np.nan] and not np.isnan(max_drawdown) else np.nan

    signals = trading_signals["signals"]
    cumsum = trading_signals["cumsum"] if "cumsum" in trading_signals.columns else signals.cumsum()

    number_of_longs = int((signals == 1).sum())
    number_of_exits = int((signals < 0).sum())
    number_of_trades = number_of_longs + number_of_exits
    total_length_of_trades = int((cumsum != 0).sum())

    average_length_of_trades = (
        total_length_of_trades / number_of_trades if number_of_trades > 0 else np.nan
    )
    profit_per_trade = (
        (float(total_asset.iloc[-1]) - capital0) / number_of_trades
        if number_of_trades > 0
        else np.nan
    )

    result = pd.DataFrame(
        {
            "CAGR": [growth_rate],
            "portfolio return": [total_return],
            "benchmark return": [benchmark_return],
            "sharpe ratio": [sharpe],
            "maximum drawdown": [max_drawdown],
            "calmar ratio": [calmar],
            "omega ratio": [omega_ratio(returns, benchmark_growth_rate)],
            "sortino ratio": [sortino_ratio(returns, benchmark_growth_rate)],
            "numbers of longs": [number_of_longs],
            # 原代码叫 shorts，但这里 signals<0 实际是清仓，不是做空。
            "numbers of exits": [number_of_exits],
            "numbers of trades": [number_of_trades],
            "total length of trades": [total_length_of_trades],
            "average length of trades": [average_length_of_trades],
            "profit per trade": [profit_per_trade],
            "max period return": [maximum],
            "min period return": [minimum],
        }
    )

    print(result)
    return result


def main():
    """
    主函数：下载数据、生成信号、回测、画图、输出统计。

    如果你本地 yfinance 下载失败，通常不是代码逻辑错误，
    而是网络环境、Yahoo Finance 限流、代理或 yfinance 版本问题。
    """
    # 最多持有 3 份多头仓位。
    stls = 3

    ticker = "NVDA"
    stdate = "2015-04-01"
    eddate = "2018-02-15"

    # 只截取最后一部分数据画图；回测统计默认仍使用全量数据。
    slicer = 700

    df = download_price_data(ticker, stdate, eddate)

    trading_signals = signal_generation(df, heikin_ashi, stls)

    viz = trading_signals.iloc[slicer:].copy()
    if len(viz) > 0:
        plot(viz, ticker)

    portfolio_details = portfolio(trading_signals)
    profit(portfolio_details.iloc[slicer:].copy())

    stats(portfolio_details, trading_signals, stdate, eddate)


if __name__ == "__main__":
    main()
