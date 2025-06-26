import yfinance as yf
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime

# 初始化 Dash 應用
app = Dash(__name__)

# 下載資料（僅一天）
symbol = "SOL-USD"
start_date = "2025-06-15"
end_date = "2025-06-19"

df_5m = yf.download(symbol, start=start_date, end=end_date, interval='5m').reset_index()
df_4h = yf.download(symbol, start=start_date, end=end_date, interval='4h').reset_index()

# 模擬時間 index（每筆代表5分鐘）
total_ticks = len(df_5m)

# 頁面佈局
app.layout = html.Div([
    html.H1(f"{symbol} 模擬交易環境"),
    dcc.Graph(id='candlestick-graph'),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0)  # 每秒更新
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):

    window_5m = 30
    window_4h = 30
    end_idx = min(n + 1, total_ticks)
    
    start_5m = max(end_idx - window_5m, 0)
    partial_5m = df_5m.iloc[start_5m:end_idx]
    
    latest_time = df_5m['Datetime'].iloc[end_idx - 1]
    partial_4h_all = df_4h[df_4h['Datetime'] <= latest_time]
    partial_4h = partial_4h_all.tail(window_4h)

    fig = make_candlestick_figure(partial_5m, partial_4h)
    return fig

    '''
    # 模擬進度的資料
    partial_5m = df_5m.iloc[:end_idx]
    partial_time = partial_5m['Datetime'].iloc[-1]
    partial_4h = df_4h[df_4h['Datetime'] <= partial_time]

    fig = make_candlestick_figure(partial_5m, partial_4h)
    return fig
    '''

def make_candlestick_figure(df5, df4):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        vertical_spacing=0.05,
                        subplot_titles=('5-Minute Chart', '4-Hour Chart'),
                        row_heights=[0.5, 0.5])

    fig.add_trace(go.Candlestick(
        x=df5['Datetime'],
        open=df5[('Open', 'SOL-USD')], high=df5[('High', 'SOL-USD')],
        low=df5[('Low', 'SOL-USD')], close=df5[('Close', 'SOL-USD')],
        name="5m"
    ), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=df4['Datetime'],
        open=df4[('Open', 'SOL-USD')], high=df4[('High', 'SOL-USD')],
        low=df4[('Low', 'SOL-USD')], close=df4[('Close', 'SOL-USD')],
        name="4h"
    ), row=2, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
