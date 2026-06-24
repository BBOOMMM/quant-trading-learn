
# coding: utf-8

# In[1]:


# Shooting Star（射击之星）是一种看跌 K 线形态。
# 它可以理解成上下翻转后的锤子线：实体较小，上影线很长，下影线很短或几乎没有。
# 直观含义是：价格盘中一度冲高，但收盘被压下来，说明上方抛压较强。
# 形态细节可以参考 Investopedia：
# https://www.investopedia.com/terms/s/shootingstar.asp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance
from pathlib import Path


# In[2]:


def _flatten_yfinance_columns(df,ticker=None):

    if not isinstance(df.columns,pd.MultiIndex):
        return df

    data=df.copy()

    if ticker is not None and ticker in data.columns.get_level_values(-1):
        data=data.xs(ticker,axis=1,level=-1)
    else:
        data.columns=data.columns.get_level_values(0)

    return data


# 判断每根 K 线是否满足 Shooting Star 的各项条件。
def shooting_star(data,lower_bound,body_size):

    df=data.copy()

    # 条件 1：收跌 K 线，开盘价高于或等于收盘价。
    df['condition1']=np.where(df['Open']>=df['Close'],1,0)

    # 条件 2：下影线很短，或者几乎没有下影线。
    df['condition2']=np.where(
        (df['Close']-df['Low'])<lower_bound*abs(df['Close']-df['Open']),1,0)

    # 条件 3：实体较小。
    # 用平均实体长度作为参照，避免绝对价格水平影响判断。
    df['condition3']=np.where(abs(df['Open']-df['Close']) < 
                              np.mean(abs(df['Open']-df['Close']))*body_size,
                              1,0)

    # 条件 4：上影线较长，至少是实体长度的 2 倍。
    df['condition4']=np.where(
        (df['High']-df['Open'])>=2*(
            df['Open']-df['Close']),1,0)

    # 条件 5 和 6：形态出现前价格处于短线上涨趋势。
    df['condition5']=np.where(
        df['Close']>=df['Close'].shift(1),1,0)
    df['condition6']=np.where(
        df['Close'].shift(1)>=df['Close'].shift(2),1,0)

    # 条件 7：下一根 K 线的最高价不能超过 Shooting Star 的最高价。
    df['condition7']=np.where(
        df['High'].shift(-1)<=df['High'],1,0)

    # 条件 8：下一根 K 线收盘价低于或等于 Shooting Star 的收盘价。
    df['condition8']=np.where(
        df['Close'].shift(-1)<=df['Close'],1,0)
    
    return df


# In[3]:


# 根据信号形态生成交易信号。
# signals=-1 表示开空，signals=1 表示平空。
def signal_generation(df,method,
                      lower_bound=0.2,body_size=0.5,
                      stop_threshold=0.05,
                      holding_period=7):

    # 先计算 Shooting Star 的 8 个条件。
    data=method(df,lower_bound,body_size)

    # 只有同时满足 8 个条件，才认为出现 Shooting Star。
    # 实盘中这个定义可能过于严格，可以适当放宽 body_size 等参数。
    data['signals']=data['condition1']*data[
        'condition2']*data['condition3']*data[
        'condition4']*data['condition5']*data[
        'condition6']*data['condition7']*data[
        'condition8']

    # Shooting Star 是看跌信号，因此记为开空信号 -1。
    data['signals']=-data['signals']
    
    # 为每个开空信号寻找平仓位置。
    idxlist=list(data.index[data['signals']==-1])
    for entry_ind in idxlist:

        # 如果这个信号后来已经被其他逻辑覆盖，则跳过。
        if data.at[entry_ind,'signals']!=-1:
            continue

        # 入场价格。
        entry_pos=data.at[entry_ind,'Close']

        for counter,exit_ind in enumerate(
            range(entry_ind+1,len(data)),
            start=1
        ):

            price_change=abs(data.at[exit_ind,'Close']/entry_pos-1)

            # 盈亏达到阈值，或者持仓超过最大天数，就平空。
            if price_change>stop_threshold or counter>=holding_period:
                data.at[exit_ind,'signals']=1
                break

    # 累计信号得到持仓状态。负数表示持有空头。
    data['positions']=data['signals'].cumsum()
    
    return data


# In[4]:


# matplotlib 新版本已经没有旧的 candlestick 函数。
# 为了不额外依赖 mpl_finance，这里手写一个简化版 K 线绘图函数：
# 用 fill_between 画实体，用竖线画上下影线。
def candlestick(df,ax=None,highlight=None,titlename='',
                highcol='High',lowcol='Low',
                opencol='Open',closecol='Close',xcol='Date',
                colorup='r',colordown='g',highlightcolor='y',
                **kwargs):  
    
    # K 线实体宽度约为 0.6。
    dif=[(-3+i)/10 for i in range(7)]
    
    if ax is None:
        ax=plt.figure(figsize=(10,5)).add_subplot(111)
    
    # 逐根绘制 K 线。
    for i in range(len(df)):
        
        # 用 7 个横向点构造 K 线实体宽度。
        x=[i+j for j in dif]
        y1=[df[opencol].iloc[i]]*7
        y2=[df[closecol].iloc[i]]*7

        barcolor=colorup if y2[0]>=y1[0] else colordown
        
        # 如果实体顶部不是最高价，则绘制上影线。
        if df[highcol].iloc[i]!=max(df[opencol].iloc[i],df[closecol].iloc[i]):
            
            ax.plot([i,i],
                    [df[highcol].iloc[i],
                     max(df[opencol].iloc[i],
                         df[closecol].iloc[i])*1.001],c='k',**kwargs)
    
        # 如果实体底部不是最低价，则绘制下影线。
        if df[lowcol].iloc[i]!=min(df[opencol].iloc[i],df[closecol].iloc[i]):             
            
            ax.plot([i,i],
                    [df[lowcol].iloc[i],
                     min(df[opencol].iloc[i],
                         df[closecol].iloc[i])*0.999],c='k',**kwargs)
        
        # 绘制 K 线实体。
        ax.fill_between(x,y1,y2,
                        edgecolor='k',
                        facecolor=barcolor,**kwargs)
        
        if highlight:
            if df[highlight].iloc[i]==-1:
                ax.fill_between(x,y1,y2,
                                edgecolor='k',
                                facecolor=highlightcolor,**kwargs)

    # K 线子图不显示横轴日期，避免太拥挤。
    ax.set_xticks([])
    ax.grid(True)
    ax.set_title(titlename)

    return ax


# In[5]:


# 绘制回测结果。
def plot(data,name):   

    output_dir=Path(__file__).resolve().parent/'Shooting Star'
    output_dir.mkdir(exist_ok=True)
    
    fig=plt.figure(figsize=(10,6))

    # 第一张子图：K 线，并高亮 Shooting Star。
    ax1=plt.subplot2grid((250,1),(0,0),
                         rowspan=120,
                         fig=fig,
                         ylabel='Candlestick')
    candlestick(data,ax1,
                highlight='signals',
                highlightcolor='#FFFF00')

    # 第二张子图：真实收盘价，并标出开空和平空位置。
    ax2=plt.subplot2grid((250,1),(130,0),
                         rowspan=120,
                         fig=fig,
                         ylabel='£ per share',
                         xlabel='Date')
    ax2.plot(data.index,
             data['Close'],
             label=name)

    # signals=-1 是开空，signals=1 是平空。
    ax2.plot(data.loc[data['signals']==-1].index,
             data['Close'].loc[data['signals']==-1],
             marker='v',lw=0,c='r',label='short',
             markersize=10)
    ax2.plot(data.loc[data['signals']==1].index,
             data['Close'].loc[data['signals']==1],
             marker='^',lw=0,c='g',label='long',
             markersize=10)

    # 只显示约 5 个横轴日期刻度。
    step=max(1,len(data)//5)
    ax2.set_xticks(range(0,len(data),step))
    ax2.set_xticklabels(data['Date'].iloc[0::step].dt.date)
    
    ax2.grid(True)
    ax2.legend(loc='lower left')
    fig.tight_layout(pad=0.1)
    fig.savefig(output_dir/f'{name}_shooting_star.png',
                dpi=300,bbox_inches='tight')
    plt.close(fig)


# In[6]:


def main():
    
    # 初始化回测参数。
    stdate='2000-01-01'
    eddate='2021-11-04'
    name='Vodafone'
    ticker='VOD.L'

    df=yfinance.download(ticker,start=stdate,end=eddate)
    df=_flatten_yfinance_columns(df,ticker)

    if df.empty:
        raise ValueError(f'{ticker} 在 {stdate} 到 {eddate} 之间没有下载到数据。')

    df.reset_index(inplace=True)
    df['Date']=pd.to_datetime(df['Date'])

    # 生成交易信号。
    new=signal_generation(df,shooting_star)

    # 截取一小段数据用于展示，方便突出 Shooting Star。
    subset=new.loc[5260:5283].copy()
    subset.reset_index(inplace=True,drop=True)

    # 可视化。
    plot(subset,name)


# In[7]:


if __name__ == '__main__':
    main()
