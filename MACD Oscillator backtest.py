# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:57:46 2018

@author: Administrator
"""

# In[1]:

#need to get fix yahoo finance package first

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests
from pathlib import Path


# In[2]:

#simple moving average
def macd(signals):
    
    
    signals['ma1']=signals['Close'].rolling(window=ma1,min_periods=1,center=False).mean()
    signals['ma2']=signals['Close'].rolling(window=ma2,min_periods=1,center=False).mean()
    
    return signals



# In[3]:

#signal generation
#when the short moving average is larger than long moving average, we long and hold
#when the short moving average is smaller than long moving average, we clear positions
#the logic behind this is that the momentum has more impact on short moving average
#we can subtract short moving average from long moving average
#the difference between is sometimes positive, it sometimes becomes negative
#thats why it is named as moving average converge/diverge oscillator
def signal_generation(df,method):
    
    signals=method(df)
    signals['positions']=0

    #oscillator is the difference between two moving average
    #when it is positive, we long, vice versa
    signals['oscillator']=signals['ma1']-signals['ma2']

    #positions becomes and stays one once the short moving average is above long moving average
    signals.loc[signals.index[ma1:], 'positions'] = np.where(
        signals['oscillator'].iloc[ma1:] >= 0,
        1,
        0
    )

    #as positions only imply the holding
    #we take the difference to generate real trade signal
    signals['signals']=signals['positions'].diff()

    return signals



# In[4]:

#plotting the backtesting result
def plot(new, ticker):

    print(new['signals'].value_counts(dropna=False))
    output_dir = Path(__file__).resolve().parent / 'MACD'
    output_dir.mkdir(exist_ok=True)
    file_prefix = f'{ticker}_{stdate}_{eddate}'
    
    #the first plot is the actual close price with long/short positions
    fig=plt.figure()
    ax=fig.add_subplot(111)
    long_signals = new[new['signals'] == 1]
    short_signals = new[new['signals'] == -1]
    
    new['Close'].plot(ax=ax, label=ticker)
    ax.plot(long_signals.index,long_signals['Close'],label='LONG',lw=0,marker='^',c='g')
    ax.plot(short_signals.index,short_signals['Close'],label='SHORT',lw=0,marker='v',c='r')

    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title('Positions')
    fig.savefig(output_dir / f'{file_prefix}_positions.png', dpi=300, bbox_inches='tight')
    
    #the second plot is long/short moving average with oscillator
    #note that i use bar chart for oscillator
    fig=plt.figure()
    cx=fig.add_subplot(211)

    new['oscillator'].plot(ax=cx, kind='bar', color='r')

    cx.legend(loc='best')
    cx.grid(True)
    cx.set_xticks([])
    cx.set_xlabel('')
    cx.set_title('MACD Oscillator')

    bx=fig.add_subplot(212)

    new['ma1'].plot(ax=bx, label='ma1')
    new['ma2'].plot(ax=bx, label='ma2', linestyle=':')
    
    bx.legend(loc='best')
    bx.grid(True)
    fig.savefig(output_dir / f'{file_prefix}_macd_oscillator.png', dpi=300, bbox_inches='tight')

    
# In[5]:

def main():
    
    #input the long moving average and short moving average period
    #for the classic MACD, it is 12 and 26
    #once a upon a time you got six trading days in a week
    #so it is two week moving average versus one month moving average
    #for now, the ideal choice would be 10 and 21
    
    global ma1,ma2,stdate,eddate,ticker,slicer

    #macd is easy and effective
    #there is just one issue
    #entry signal is always late
    #watch out for downward EMA spirals!
    ma1 = int(input('ma1 [default: 10]: ') or 10)
    ma2 = int(input('ma2 [default: 21]: ') or 21)
    stdate = input('start date in format yyyy-mm-dd [default: 2020-01-01]: ') or '2020-01-01'
    eddate = input('end date in format yyyy-mm-dd [default: 2021-01-01]: ') or '2021-01-01'
    ticker = input('ticker [default: AAPL]: ') or 'AAPL'

    #slicing the downloaded dataset
    #if the dataset is too large, backtesting plot would look messy
    #you get too many markers cluster together
    slicer = int(input('slicing [default: 0]: ') or 0)

    #downloading data
    session = requests.Session(impersonate="chrome")

    df = yf.download(
        ticker,
        start=stdate,
        end=eddate,
        progress=False,
        auto_adjust=False,
        session=session,
        timeout=30,
    )
    print(df.columns)
    
    new=signal_generation(df,macd)
    new=new[slicer:]
    plot(new, ticker)


#how to calculate stats could be found from my other code called Heikin-Ashi
# https://github.com/je-suis-tm/quant-trading/blob/master/heikin%20ashi%20backtest.py


if __name__ == '__main__':
    main()
