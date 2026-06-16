# coding: utf-8

#details of awesome oscillator can be found here
# https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
#basically i use awesome oscillator to compare with macd oscillator
#lets see which one makes more money
#there is not much difference between two of em
#this time i use exponential smoothing on macd
#for awesome oscillator, i use simple moving average instead
#the rules are quite simple
#these two are momentum trading strategy
#they compare the short moving average with long moving average
#if the difference is positive
#we long the asset, vice versa
#awesome oscillator has slightly more conditions for signals
#we will see about it later
#for more details about macd
# https://github.com/je-suis-tm/quant-trading/blob/master/MACD%20oscillator%20backtest.py


# In[1]:
#need to get fix yahoo finance package first
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# In[2]:

def normalize_price_columns(df,ticker):

    if isinstance(df.columns,pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df=df.xs(ticker,axis=1,level=-1)
        else:
            df.columns=df.columns.get_level_values(0)

    return df


#this part is macd
#i will not go into details as i have another session called macd
#the only difference is that i use ewma function to apply exponential smoothing technique
def ewmacd(signals,ma1,ma2):
    
    signals['macd ma1']=signals['Close'].ewm(span=ma1).mean()    
    signals['macd ma2']=signals['Close'].ewm(span=ma2).mean()   
    
    return signals
    
def signal_generation(df,method,ma1,ma2):
    
    signals=method(df,ma1,ma2)
    signals['macd positions']=0
    # signals['macd positions'][ma1:]=np.where(signals['macd ma1'][ma1:]>=signals['macd ma2'][ma1:],1,0)
    signals.loc[signals.index[ma1:], 'macd positions'] = np.where(
        signals['macd ma1'].iloc[ma1:] >= signals['macd ma2'].iloc[ma1:],
        1,
        0
    )
    signals['macd signals']=signals['macd positions'].diff().fillna(0)
    signals['macd oscillator']=signals['macd ma1']-signals['macd ma2']
    return signals


# In[3]:
    
#for awesome oscillator
#moving average is based on the mean of high and low instead of close price
def awesome_ma(signals):
    
    signals['awesome ma1'],signals['awesome ma2']=0,0
    signals['awesome ma1']=((signals['High']+signals['Low'])/2).rolling(window=5).mean()
    signals['awesome ma2']=((signals['High']+signals['Low'])/2).rolling(window=34).mean()
    
    return signals


# Awesome Oscillator 信号生成函数
def awesome_signal_generation(df, method):
    
    signals = method(df).copy()
    signals.reset_index(inplace=True)

    # 初始化信号列
    # 1  表示买入 / 开多
    # -1 表示卖出 / 平多
    # 0  表示无操作
    signals['awesome signals'] = 0

    # Awesome Oscillator，简称 AO
    # AO = 短周期均线 - 长周期均线
    signals['awesome oscillator'] = signals['awesome ma1'] - signals['awesome ma2']

    # cumsum 用来表示当前持仓状态
    # 0 表示空仓，1 表示持有多头仓位
    signals['cumsum'] = 0

    # position 显式记录当前仓位，避免每轮循环都重新计算 cumsum
    position = 0

    # Saucer 形态需要判断 AO 柱子的颜色
    # AO 柱子的颜色不是由 Open/Close 决定，而是由 AO 自身相邻两根柱子的变化决定：
    # AO[i] > AO[i-1] 表示当前 AO 柱变高，通常记为绿色
    # AO[i] < AO[i-1] 表示当前 AO 柱变低，通常记为红色
    #
    # 严格判断“两根红色柱接一根绿色柱”时，需要用到 i-3, i-2, i-1, i 四个 AO 数值，
    # 所以循环从 i = 3 开始。
    for i in range(3, len(signals)):

        ao_i3 = signals['awesome oscillator'].iloc[i-3]
        ao_i2 = signals['awesome oscillator'].iloc[i-2]
        ao_i1 = signals['awesome oscillator'].iloc[i-1]
        ao_i  = signals['awesome oscillator'].iloc[i]

        ma1_prev = signals['awesome ma1'].iloc[i-1]
        ma2_prev = signals['awesome ma2'].iloc[i-1]
        ma1_now  = signals['awesome ma1'].iloc[i]
        ma2_now  = signals['awesome ma2'].iloc[i]

        # 如果均线或 AO 存在 NaN，则这一根不产生交易信号
        if (
            pd.isna(ao_i3) or pd.isna(ao_i2) or pd.isna(ao_i1) or pd.isna(ao_i) or
            pd.isna(ma1_prev) or pd.isna(ma2_prev) or
            pd.isna(ma1_now) or pd.isna(ma2_now)
        ):
            signals.at[i, 'awesome signals'] = 0
            signals.at[i, 'cumsum'] = position
            continue

        # Bullish Saucer，看涨碟形信号
        #
        # 条件含义：
        # 1. AO 全部在 0 轴上方，说明整体处于多头动能环境
        # 2. AO 连续两根下降，表示多头动能短暂回调
        # 3. 当前 AO 重新上升，表示回调可能结束，多头重新增强
        #
        # 因此它是买入 / 开多信号
        bullish_saucer = (
            ao_i3 > 0 and
            ao_i2 > 0 and
            ao_i1 > 0 and
            ao_i > 0 and
            ao_i2 < ao_i3 and      # 第一根红色 AO 柱
            ao_i1 < ao_i2 and      # 第二根红色 AO 柱
            ao_i > ao_i1           # 当前绿色 AO 柱
        )

        # Bearish Saucer，看跌碟形信号
        #
        # 条件含义：
        # 1. AO 全部在 0 轴下方，说明整体处于空头动能环境
        # 2. AO 连续两根上升，表示空头动能短暂减弱，价格可能反弹
        # 3. 当前 AO 重新下降，表示反弹可能失败，空头重新增强
        #
        # 在只做多策略里，它用于卖出 / 平多
        bearish_saucer = (
            ao_i3 < 0 and
            ao_i2 < 0 and
            ao_i1 < 0 and
            ao_i < 0 and
            ao_i2 > ao_i3 and      # 第一根绿色 AO 柱
            ao_i1 > ao_i2 and      # 第二根绿色 AO 柱
            ao_i < ao_i1           # 当前红色 AO 柱
        )

        # 均线金叉信号
        # 这里不能只判断 ma1 > ma2，否则会在 ma1 持续大于 ma2 的每一根都尝试买入。
        # 正确做法是判断“从 ma1 <= ma2 变成 ma1 > ma2”。
        golden_cross = (
            ma1_prev <= ma2_prev and
            ma1_now > ma2_now
        )

        # 均线死叉信号
        # 同理，正确做法是判断“从 ma1 >= ma2 变成 ma1 < ma2”。
        death_cross = (
            ma1_prev >= ma2_prev and
            ma1_now < ma2_now
        )

        # 默认无操作
        signal = 0

        # 空仓时，只允许产生买入信号
        if position == 0:
            # Saucer 信号优先，因为它可能比均线交叉更早出现
            if bullish_saucer:
                signal = 1
                position = 1
            elif golden_cross:
                signal = 1
                position = 1

        # 持有多头仓位时，只允许产生卖出信号
        elif position == 1:
            # Saucer 信号优先，因为它可能比均线交叉更早出现
            if bearish_saucer:
                signal = -1
                position = 0
            elif death_cross:
                signal = -1
                position = 0

        signals.at[i, 'awesome signals'] = signal
        signals.at[i, 'cumsum'] = position

    # 为了和后面的 portfolio() 函数兼容，最终再用信号累计和生成一次 cumsum
    # 在这个只做多版本里，cumsum 应该只会是 0 或 1
    signals['cumsum'] = signals['awesome signals'].cumsum()

    return signals

# In[4]:
    
#we plot the results to compare
#basically the same as macd
#im not gonna explain much
def plot(new,ticker):

    output_dir=Path(__file__).resolve().parent/'Awesome Oscillator'
    output_dir.mkdir(exist_ok=True)
    
    #positions
    fig=plt.figure()
    ax=fig.add_subplot(211)

    new['Close'].plot(label=ticker)
    ax.plot(new.loc[new['awesome signals']==1].index,new['Close'][new['awesome signals']==1],label='AWESOME LONG',lw=0,marker='^',c='g')
    ax.plot(new.loc[new['awesome signals']==-1].index,new['Close'][new['awesome signals']==-1],label='AWESOME SHORT',lw=0,marker='v',c='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')

    bx=fig.add_subplot(212,sharex=ax)
    new['Close'].plot(label=ticker)
    bx.plot(new.loc[new['macd signals']==1].index,new['Close'][new['macd signals']==1],label='MACD LONG',lw=0,marker='^',c='g')
    bx.plot(new.loc[new['macd signals']==-1].index,new['Close'][new['macd signals']==-1],label='MACD SHORT',lw=0,marker='v',c='r')

    plt.legend(loc='best')
    plt.grid(True)
    fig.savefig(output_dir/f'{ticker}_positions.png',
                dpi=300,bbox_inches='tight')
    plt.close(fig)

    
    #oscillator
    fig=plt.figure()
    cx=fig.add_subplot(211)

    c=np.where(new['Open']>new['Close'],'r','g')
    cx.bar(range(len(new)),new['awesome oscillator'],color=c,label='awesome oscillator')

    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Oscillator')

    dx=fig.add_subplot(212,sharex=cx)

    new['macd oscillator'].plot(kind='bar',label='macd oscillator')

    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('')
    plt.xticks([])
    fig.savefig(output_dir/f'{ticker}_oscillator.png',
                dpi=300,bbox_inches='tight')
    plt.close(fig)



    #moving average
    fig=plt.figure()
    ex=fig.add_subplot(211)

    new['awesome ma1'].plot(label='awesome ma1')
    new['awesome ma2'].plot(label='awesome ma2',linestyle=':')

    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks([])
    plt.xlabel('')
    plt.title('Moving Average')

    fx=fig.add_subplot(212,sharex=ex)
    
    new['macd ma1'].plot(label='macd ma1')
    new['macd ma2'].plot(label='macd ma2',linestyle=':')

    plt.legend(loc='best')
    plt.grid(True)
    fig.savefig(output_dir/f'{ticker}_moving_average.png',
                dpi=300,bbox_inches='tight')
    plt.close(fig)


# In[5]:

# 通常我不会加入回测统计指标
# 但为了和 MACD 策略做对比，这里破例加入
# capital0 表示初始资金
# positions 表示每次交易买入或卖出的股票数量
def portfolio(signals):

    capital0 = 5000
    positions = 100

    portfolio = pd.DataFrame()
    portfolio['Close'] = signals['Close']
    
    # cumsum 用来表示当前持仓状态
    # 持仓市值 = 当前持仓数量 × 收盘价 × 每次交易股数
    portfolio['awesome holding'] = signals['cumsum'] * portfolio['Close'] * positions
    portfolio['macd holding'] = signals['macd positions'] * portfolio['Close'] * positions

    # cash 表示账户中的现金
    # 现金 = 初始资金 - 累计买入支出 + 累计卖出收入
    # 这里用 cumsum 对每次交易产生的现金流进行累加
    portfolio['awesome cash'] = capital0 - (signals['awesome signals'] * portfolio['Close'] * positions).cumsum()
    portfolio['macd cash'] = capital0 - (signals['macd signals'] * portfolio['Close'] * positions).cumsum()

    # 总资产 = 持仓市值 + 现金
    portfolio['awesome asset'] = portfolio['awesome holding'] + portfolio['awesome cash']
    portfolio['macd asset'] = portfolio['macd holding'] + portfolio['macd cash']

    # 根据总资产计算每一期收益率
    portfolio['awesome return'] = portfolio['awesome asset'].pct_change()
    portfolio['macd return'] = portfolio['macd asset'].pct_change()
    
    return portfolio


# In[6]:

# 绘制两个策略的资产变化曲线，用来比较 Awesome Oscillator 和 MACD 的表现
def profit(portfolio):

    output_dir = Path(__file__).resolve().parent / 'Awesome Oscillator'
    output_dir.mkdir(exist_ok=True)
        
    gx = plt.figure()
    gx.add_subplot(111)

    portfolio['awesome asset'].plot()
    portfolio['macd asset'].plot()

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Awesome VS MACD')
    gx.savefig(
        output_dir / 'awesome_vs_macd_profit.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(gx)


# In[7]:

# 计算最大回撤 Maximum Drawdown, MDD
# 思路很简单：
# 对于每一天，取当前资产价值，
# 和之前出现过的最高资产价值进行比较，
# 得到当天相对于历史高点的回撤比例。
#
# 如果当前资产不是截至目前的最高值，
# 那么这个回撤值应该是负数。
#
# 我们用一个临时变量 temp 记录遍历过程中出现过的最小回撤，
# 也就是最大回撤。
# 每当发现新的回撤比 temp 更小，就更新 temp。
# 遍历结束后，返回 temp 作为最大回撤。
# 计算的是从历史最高点跌下来的最大百分比
def mdd(series):

    series = series.dropna()
    drawdown = series / series.cummax() - 1

    return drawdown.min()


def stats(portfolio):
    
    stats = pd.DataFrame([0])

    # 计算 Sharpe Ratio
    # 为了简化，这里假设无风险收益率 risk-free rate 为 0
    # 也可以选择使用 S&P 500 等市场指数作为 benchmark
    stats['awesome sharpe'] = (
        portfolio['awesome asset'].iloc[-1] / 5000 - 1
    ) / np.std(portfolio['awesome return'])

    stats['macd sharpe'] = (
        portfolio['macd asset'].iloc[-1] / 5000 - 1
    ) / np.std(portfolio['macd return'])

    # 计算最大回撤 MDD
    stats['awesome mdd'] = mdd(portfolio['awesome asset'])
    stats['macd mdd'] = mdd(portfolio['macd asset'])

    # 输出统计结果
    print(stats)


# In[8]:   

def main():
    
    #awesome oscillator uses 5 lags as short ma
    #34 lags as long ma
    #for the consistent comparison
    #i apply the same to macd oscillator
    ma1=5
    ma2=34

    #downloading
    stdate=input('start date in format yyyy-mm-dd[default: 2020-01-01]:') or '2020-01-01'
    eddate=input('end date in format yyyy-mm-dd[default: 2021-01-01]:') or '2021-01-01'
    ticker=input('ticker[default: AAPL]:') or 'AAPL'
    df=yf.download(ticker,start=stdate,end=eddate)
    df=normalize_price_columns(df,ticker)

    if df.empty:
        print(f'No price data found for {ticker}. Please check the ticker symbol and date range.')
        return

    #slicing the downloaded dataset
    #if the dataset is too large
    #backtesting plot would look messy
    slicer=int(input('slicing[default: 0]:') or '0')
    signals=signal_generation(df,ewmacd,ma1,ma2)
    sig=awesome_signal_generation(signals,awesome_ma)
    new=sig[slicer:]
    plot(new,ticker)
    
    portfo=portfolio(sig)
    profit(portfo)
    
    stats(portfo)
    
    #from my tests
    #macd has demonstrated a higher sharpe ratio
    #it executes fewer trades but brings more profits
    #however its maximum drawdown is higher than awesome oscillator
    #which one is better?
    #it depends on your risk averse level

if __name__ == '__main__':
    main()
