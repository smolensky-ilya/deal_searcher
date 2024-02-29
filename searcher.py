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
        tick = start = lvl.mean() * 0.001
        maximum = 7
        while True:
            temp = lvl[abs(lvl.diff()) > start]
            start += tick
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
        plt.legend(loc='center left')
        all_figs[gran] = fig
    return all_figs


def rerun():
    st.experimental_rerun()


def get_link(ticker):
    return f"[{ticker}](https://www.bitget.com/futures/usdt/{ticker}) " + f"[C](?chart={ticker})"


def main():
    params = st.query_params.to_dict()  # LINK PARAMS
    ticker = (params['chart'].upper() + " | ") if 'chart' in params else ""
    st.set_page_config(layout="wide", page_title=ticker + "Deal searcher | SMK", initial_sidebar_state='collapsed')
    all_tickers = get_all_tickers()
    chart = params.get('chart', st.text_input('Check a ticker', placeholder='BTCUSDT'))
    st.title('Bitget deal searcher')
    run = st.button('Scan the market')

    if run:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        errors = []
        signals_stat = {'STRONG_BUY': 0, 'MORE_BUY': 0, 'BUY': 0,
                        'SELL': 0, 'MORE_SELL': 0, 'STRONG_SELL': 0, 'NEUTRAL': 0}
        col1.header('STR_BUY')
        col2.header('More-BUY')
        col3.header('BUY')
        col4.header('SELL')
        col5.header('More-SELL')
        col6.header('STR_SELL')
        col7.header('other')
        for each in all_tickers:
            try:
                res_5_min = get_analysis(each, interval=Interval.INTERVAL_5_MINUTES)
                res_15_min = get_analysis(each, interval=Interval.INTERVAL_15_MINUTES)
                res_1_hour = get_analysis(each, interval=Interval.INTERVAL_1_HOUR)
                res_1_day = get_analysis(each, interval=Interval.INTERVAL_1_DAY)
                if res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_BUY':
                    col1.markdown(get_link(each))
                    signals_stat['STRONG_BUY'] += 1
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'BUY':
                    col3.markdown(get_link(each))
                    signals_stat['BUY'] += 1
                elif (res_5_min in ['BUY', 'STRONG_BUY'] and res_15_min in ['BUY', 'STRONG_BUY'] and
                        res_1_hour in ['BUY', 'STRONG_BUY'] and res_1_day in ['BUY', 'STRONG_BUY']):
                    col2.markdown(get_link(each))
                    signals_stat['MORE_BUY'] += 1
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'SELL':
                    col4.markdown(get_link(each))
                    signals_stat['SELL'] += 1
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_SELL':
                    col6.markdown(get_link(each))
                    signals_stat['STRONG_SELL'] += 1
                elif (res_5_min in ['SELL', 'STRONG_SELL'] and res_15_min in ['SELL', 'STRONG_SELL'] and
                        res_1_hour in ['SELL', 'STRONG_SELL'] and res_1_day in ['SELL', 'STRONG_SELL']):
                    col5.markdown(get_link(each))
                    signals_stat['MORE_SELL'] += 1
                else:
                    col7.markdown(get_link(each))
                    signals_stat['NEUTRAL'] += 1
            except Exception as error:
                errors.append(each)
        st.sidebar.header('STATS:')
        overall = sum(signals_stat.values())
        st.sidebar.write(f"Buy: {((signals_stat['BUY'] + signals_stat['STRONG_BUY'] + signals_stat['MORE_BUY']) / overall) * 100:.2f}%")
        st.sidebar.write(f"--- Strong Buy: {(signals_stat['STRONG_BUY'] / overall) * 100:.2f}%")
        st.sidebar.write(f"--- More Buy: {(signals_stat['MORE_BUY'] / overall) * 100:.2f}%")
        st.sidebar.write(f"--- Buy: {(signals_stat['BUY'] / overall) * 100:.2f}%")
        st.sidebar.write(f"Sell: {((signals_stat['SELL'] + signals_stat['STRONG_SELL'] + signals_stat['MORE_SELL']) / overall) * 100:.2f}%")
        st.sidebar.write(f"--- Strong Sell: {(signals_stat['STRONG_SELL'] / overall) * 100:.2f}%")
        st.sidebar.write(f"--- More Sell: {(signals_stat['MORE_SELL'] / overall) * 100:.2f}%")
        st.sidebar.write(f"--- Sell: {(signals_stat['SELL'] / overall) * 100:.2f}%")
        st.sidebar.write(f"Neutral: {(signals_stat['NEUTRAL'] / overall) * 100:.2f}%")
        st.sidebar.write(f"Overall pairs checked: {overall}")
        st.sidebar.write(f'*Errors*: {", ".join(errors)}')
    if chart:
        chart = chart + "USDT" if chart[-4:] != "USDT" else chart
        st.header(chart)
        try:
            col1, col2 = st.columns(2)
            col1.markdown(f"[Bitget](https://www.bitget.com/futures/usdt/{chart})")
            col1.write("5min: " + get_analysis(chart, interval=Interval.INTERVAL_5_MINUTES))
            col1.write("15min: " + get_analysis(chart, interval=Interval.INTERVAL_15_MINUTES))
            col1.write("1H: " + get_analysis(chart, interval=Interval.INTERVAL_1_HOUR))
            col1.write("1D: " + get_analysis(chart, interval=Interval.INTERVAL_1_DAY))

            col2.write("1min: " + get_analysis(chart, interval=Interval.INTERVAL_1_MINUTE))
            col2.write("30min: " + get_analysis(chart, interval=Interval.INTERVAL_30_MINUTES))
            col2.write("2H: " + get_analysis(chart, interval=Interval.INTERVAL_2_HOURS))
            col2.write("4H: " + get_analysis(chart, interval=Interval.INTERVAL_4_HOURS))
            col2.write("1W: " + get_analysis(chart, interval=Interval.INTERVAL_1_WEEK))
            col2.write("1M: " + get_analysis(chart, interval=Interval.INTERVAL_1_MONTH))
            col1, col2 = st.columns(2)
            figs = plot_my_thing(chart, 'figure this out')
            for i, fig in enumerate(figs.keys()):
                if (i + 1) % 2 == 0:
                    col1.pyplot(figs[fig], use_container_width=True)
                else:
                    col2.pyplot(figs[fig], use_container_width=True)
        except Exception as e:
            if str(e) == "Exchange or symbol not found.":
                st.write("The ticker wasn't found or doesn't exist.")


if __name__ == "__main__":
    main()
