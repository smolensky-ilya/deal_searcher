import streamlit as st
import requests
from tradingview_ta import TA_Handler, Interval


def get_analysis(ticker, interval):
    output = TA_Handler(symbol=ticker, screener='Crypto', exchange='bitget', interval=interval)
    return output.get_analysis().summary['RECOMMENDATION']


def get_all_tickers():
    return [each['symbol'] for each in requests.
            get("https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES").json()['data']]


def rerun():
    st.experimental_rerun()


def main():
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

        for each in all_tickers:
            try:
                res_5_min = get_analysis(each, interval=Interval.INTERVAL_5_MINUTES)
                res_15_min = get_analysis(each, interval=Interval.INTERVAL_15_MINUTES)
                res_1_hour = get_analysis(each, interval=Interval.INTERVAL_1_HOUR)
                res_1_day = get_analysis(each, interval=Interval.INTERVAL_1_DAY)
                if res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_BUY':
                    col1.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'BUY':
                    col2.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'SELL':
                    col3.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
                elif res_5_min == res_15_min == res_1_hour == res_1_day == 'STRONG_SELL':
                    col4.markdown(f"[{each}](https://www.bitget.com/futures/usdt/{each})")
            except Exception as error:
                col5.write(each)


if __name__ == "__main__":
    main()
