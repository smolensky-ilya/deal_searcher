import streamlit as st
import requests
from tradingview_ta import TA_Handler, Interval
import pandas as pd
import mplfinance as mpf
from datetime import timedelta
import matplotlib.pyplot as plt
from numpy import int64


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
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int64), unit='ms') + timedelta(hours=3)
    for each in df.columns[1:]:
        df[each] = pd.to_numeric(df[each])
    df.index = df['timestamp']
    return df


def plot_my_thing(ticker, signal):
    def lines_filter(lvl):
        temp = lvl.copy()
        start = 0.00001
        maximum = 7
        while True:
            temp = lvl[abs(lvl.diff()) > start]
            start += 0.00001
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
                if (dfs[gran]['high'].max()) * 1.005 >= each >= (dfs[gran]['low'].min() * 0.95):
                    if level not in added_labels:
                        ax.axhline(each, color=grans[level][0], linestyle='-.', alpha=grans[level][1], label=level)
                        added_labels.add(level)
                    else:
                        ax.axhline(each, color=grans[level][0], linestyle='-.', alpha=grans[level][1])
        ax.set_title(f"{ticker} // {gran} // {signal.upper()}")
        plt.legend()
        all_figs[gran] = fig
    return all_figs


def rerun():
    st.experimental_rerun()


def main():
    params = st.query_params.get_all()  # LINK PARAMS
    st.set_page_config(layout="wide")
    all_tickers = get_all_tickers()
    st.title('Bitget deal searcher')
    run = st.button('Refresh')
    if run:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.header('STRONG_BUY')
        col2.header('BUY')
        col3.header('SELL')
        col4.header('STRONG_SELL')
        col5.header('***errors')

        signals = []

        for each in all_tickers:
            try:
                res_5_min = get_analysis(each, interval=Interval.INTERVAL_5_MINUTES)
                res_15_min = get_analysis(each, interval=Interval.INTERVAL_15_MINUTES)
                res_1_hour = get_analysis(each, interval=Interval.INTERVAL_1_HOUR)
                res_1_day = get_analysis(each, interval=Interval.INTERVAL_1_DAY)
                if res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_BUY':
                    signals.append([each, res_5_min])
                    col1.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'BUY':
                    signals.append([each, res_5_min])
                    col2.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'SELL':
                    signals.append([each, res_5_min])
                    col3.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_SELL':
                    signals.append([each, res_5_min])
                    col4.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
            except Exception as error:
                col5.write(each)

        chart = params['chart'] if 'chart' in params else st.text_input('Insert a ticker to build charts')
        if chart:
            col1, col2 = st.columns(2)
            for each in signals:
                figs = plot_my_thing(each[0], each[1])
                for i, fig in enumerate(figs.keys()):
                    if (i + 1) % 2 == 0:
                        col1.pyplot(figs[fig], use_container_width=True)
                    else:
                        col2.pyplot(figs[fig], use_container_width=True)


if __name__ == "__main__":
    main()
