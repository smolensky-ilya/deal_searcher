import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import timedelta
import numpy as np
from tradingview_ta import TA_Handler, Interval


def get_analysis(ticker, interval):
    output = TA_Handler(symbol=ticker, screener='Crypto', exchange='bitget', interval=interval)
    return output.get_analysis().summary['RECOMMENDATION']


def get_all_tickers():
    return [each['symbol'] for each in requests.
            get("https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES").json()['data']]


def get_candles(ticker, gran='1H', limit=500):
    response = requests.get(
        f"https://api.bitget.com/api/v2/mix/market/candles?symbol={ticker}"
        f"&granularity={gran}&productType=usdt-futures&limit={limit}")
    df = pd.DataFrame(response.json()['data'],
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'base_coin_value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms') + timedelta(hours=3)
    for each in df.columns[1:]:
        df[each] = pd.to_numeric(df[each])
    df.index = df['timestamp']
    return df


def plot_my_thing(ticker, signal):
    def lines_filter(levell):
        temp = levell.copy()
        start = 1
        maximum = 7
        while True:
            temp = levell[abs(levell.diff()) > start]
            start += 1
            if len(temp) <= maximum:
                return temp

    all_figs = {}
    grans = {'5m': ['yellow', 0.4], '15m': ['blue', 0.4], '1H': ['red', 0.8], '1D': ['black', 1]}
    dfs = {}
    lvls = {}

    for each in grans.keys():
        candles = get_candles(ticker, gran=each)
        dfs[each] = candles
        sup = candles[candles.close == candles.close.rolling(5, center=True).min()].close
        res = candles[candles.open == candles.open.rolling(5, center=True).max()].open
        lvls[each] = lines_filter(pd.concat([sup, res]))

    for gran in grans.keys():
        added_labels = set()
        fig, ax = plt.subplots(figsize=(12, 5))
        mpf.plot(dfs[gran], type='candle', ax=ax, style='charles')
        for level in lvls.keys():
            for each in lvls[level]:
                if (dfs[gran]['high'].max()) * 1.002 >= each >= (dfs[gran]['low'].min() * 0.98):
                    if level not in added_labels:
                        ax.axhline(each, color=grans[level][0], linestyle='-.', alpha=grans[level][1], label=level)
                        added_labels.add(level)
                    else:
                        ax.axhline(each, color=grans[level][0], linestyle='-.', alpha=grans[level][1])
        ax.set_title(f"{ticker} // {gran} // {signal.upper()}")
        plt.legend()
        all_figs[gran] = fig
    return all_figs


def main():
    st.set_page_config(layout="wide")
    all_tickers = get_all_tickers()
    #how_many = st.number_input('How many pairs', min_value=1, max_value=len(all_tickers), value=1)
    #limit = st.number_input('Candles', min_value=100, max_value=1000, value=100)

    col1, col2 = st.columns(2)
    for each in all_tickers:
        try:
            res_5_min = get_analysis(each, interval=Interval.INTERVAL_5_MINUTES)[-4:]
            if res_5_min == get_analysis(each, interval=Interval.INTERVAL_15_MINUTES)[-4:]:
                if res_5_min == get_analysis(each, interval=Interval.INTERVAL_1_HOUR)[-4:]:
                    if res_5_min == get_analysis(each, interval=Interval.INTERVAL_1_DAY)[-4:]:
                        #st.write(each + ": " + res_5_min)
                        figs = plot_my_thing(each, signal=res_5_min)
                        col1.pyplot(figs['5m'], use_container_width=True)
                        col2.pyplot(figs['15m'], use_container_width=True)
                        col1.pyplot(figs['1H'], use_container_width=True)
                        col2.pyplot(figs['1D'], use_container_width=True)
                        st.write('---')
        except:
            print('error***: ' + each)


if __name__ == "__main__":
    main()
