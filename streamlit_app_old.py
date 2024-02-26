import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks


def get_all_tickers():
    return [each['symbol'] for each in requests.
            get("https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES").json()['data']]


def get_candles(ticker, gran='1H', limit=100):
    response = requests.get(
        f"https://api.bitget.com/api/v2/mix/market/candles?symbol={ticker}"
        f"&granularity={gran}&productType=usdt-futures&limit={limit}")
    df = pd.DataFrame(response.json()['data'],
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'base_coin_value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms') + timedelta(hours=3)
    return df


def plot_candles(ticker, gran, df):
    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    #pivots = find_pivots(df)

    fig.update_layout(title=ticker + " // " + gran,
                      xaxis_title='Time',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)  # Hides the range slider

    support_levels, resistance_levels = find_support_resistance(df)
    for level in support_levels:
        fig.add_shape(type="line",
                      x0=df['timestamp'].iloc[0], y0=level,  # Start of the line
                      x1=df['timestamp'].iloc[-1], y1=level,  # End of the line
                      line=dict(color="RoyalBlue", width=2, dash="dash"))

    # Add horizontal lines for resistance levels
    for level in resistance_levels:
        fig.add_shape(type="line",
                      x0=df['timestamp'].iloc[0], y0=level,  # Start of the line
                      x1=df['timestamp'].iloc[-1], y1=level,  # End of the line
                      line=dict(color="Red", width=2, dash="dash"))



    """fig.add_shape(type="line",
        x0=df['timestamp'].iloc[0], y0=df['low'].iloc[0],  # replace with your actual data
        x1=df['timestamp'].iloc[-1], y1=df['low'].iloc[-1],  # replace with your actual data
        line=dict(color="RoyalBlue", width=2))"""

    return fig


def find_pivots(df):
    # Define the window size for finding pivots
    lookahead = 5

    # Find local maxima
    df['pivot_high'] = df['high'][(df['high'].shift(1) < df['high']) &
                                  (df['high'].shift(-1) < df['high'])]

    # Find local minima
    df['pivot_low'] = df['low'][(df['low'].shift(1) > df['low']) &
                                (df['low'].shift(-1) > df['low'])]

    # Forward-fill the NaNs for the window size, then back to NaN
    df['pivot_high'] = df['pivot_high'].replace(to_replace=0, method='ffill', limit=lookahead - 1)
    df['pivot_low'] = df['pivot_low'].replace(to_replace=0, method='ffill', limit=lookahead - 1)

    # Only keep the actual pivots
    df['pivot_high'] = df['pivot_high'].where(df['pivot_high'] == df['high'])
    df['pivot_low'] = df['pivot_low'].where(df['pivot_low'] == df['low'])

    return df


def find_support_resistance(df, distance=5):
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    # Finding local maxima for resistance
    peaks, _ = find_peaks(df['high'], distance=distance)
    resistance_levels = df['high'][peaks].value_counts().index.sort_values()

    # Finding local minima for support (invert the lows to find peaks)
    troughs, _ = find_peaks(-df['low'], distance=distance)
    support_levels = df['low'][troughs].value_counts().index.sort_values()

    return support_levels, resistance_levels

def main():
    st.set_page_config(layout="wide")
    all_tickers = get_all_tickers()
    how_many = st.number_input('How many pairs', min_value=1, max_value=len(all_tickers), value=1)
    limit = st.number_input('Candles', min_value=100, max_value=1000, value=100)

    col1, col2 = st.columns(2)
    for i in range(how_many):
        candles_1m = get_candles(all_tickers[i], gran='1m', limit=limit)
        col1.plotly_chart(plot_candles(all_tickers[i], '1m', candles_1m),
                          use_container_width=True)
        candles_5m = get_candles(all_tickers[i], gran='5m', limit=limit)
        col2.plotly_chart(plot_candles(all_tickers[i], '5m', candles_5m),
                          use_container_width=True)
        #col1.plotly_chart(plot_candles(all_tickers[i], '15m', get_candles(all_tickers[i], gran='15m', limit=limit)),
        #                  use_container_width=True)
        #col2.plotly_chart(plot_candles(all_tickers[i], '1H', get_candles(all_tickers[i], gran='1H', limit=limit)),
        #                  use_container_width=True)
        st.divider()

        st.dataframe(find_pivots(candles_5m))
        st.write(find_support_resistance(candles_5m))

if __name__ == "__main__":
    main()
