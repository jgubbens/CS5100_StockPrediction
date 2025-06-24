import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from DataExtract import fetch_binanceus_ohlcv
from SNR_function import SupportResistanceManager

#Data extraction
df_5m = fetch_binanceus_ohlcv('SOL/USDT', '5m', start_time='2025-01-18T00:00:00Z', end_time='2025-06-20T00:00:00Z')
df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', start_time='2025-01-18T00:00:00Z', end_time='2025-06-20T00:00:00Z')

# initialize SNR data
sr_manager_5m = SupportResistanceManager()
sr_manager_4h = SupportResistanceManager()

# plotly candle stick
def make_candlestick_figure(df5, df4, supports_5m, resistances_5m, supports_4h, resistances_4h):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        vertical_spacing=0.05,
                        subplot_titles=('5-Minute Chart', '4-Hour Chart'),
                        row_heights=[0.5, 0.5])

    fig.add_trace(go.Candlestick(x=df5.index, open=df5['open'], high=df5['high'],
                                 low=df5['low'], close=df5['close'], name="5m"), row=1, col=1)
    for s in supports_5m:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=1, col=1,
                      annotation_text=f"S: {s:.2f}", annotation_position="bottom left")
    for r in resistances_5m:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=1, col=1,
                      annotation_text=f"R: {r:.2f}", annotation_position="top left")

    fig.add_trace(go.Candlestick(x=df4.index, open=df4['open'], high=df4['high'],
                                 low=df4['low'], close=df4['close'], name="4h"), row=2, col=1)
    for s in supports_4h:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=2, col=1,
                      annotation_text=f"S: {s:.2f}", annotation_position="bottom left")
    for r in resistances_4h:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=2, col=1,
                      annotation_text=f"R: {r:.2f}", annotation_position="top left")

    fig.update_layout(height=800, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

# initialize dash
app = Dash(__name__)
total_ticks = len(df_5m)
RECALC_EVERY = 30

# Simulation enviorment param
app.layout = html.Div([
    html.H1("SOL Simulation"),
    dcc.Graph(id='candlestick-graph'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)

def update_graph(n):
    end_idx = min(n + 1, total_ticks)
    #partial_5m = df_5m.iloc[:end_idx]
    #partial_4h = df_4h[df_4h.index <= partial_5m.index[-1]]
    partial_5m = df_5m.iloc[max(0, end_idx - 300):end_idx]
    partial_4h = df_4h[df_4h.index <= partial_5m.index[-1]].iloc[-300:]


    if partial_5m.empty:
        return go.Figure()

    latest_candle_5m = partial_5m.iloc[-1]
    latest_candle_4h = partial_4h.iloc[-1] if not partial_4h.empty else None

    if n == 0 or n % RECALC_EVERY == 0:
        sr_manager_5m.reset_levels()
        sr_manager_5m.calculate_levels(partial_5m)
        sr_manager_5m.update_levels(latest_candle_5m)

        if latest_candle_4h is not None:
            sr_manager_4h.reset_levels()
            sr_manager_4h.calculate_levels(partial_4h)
            sr_manager_4h.update_levels(latest_candle_4h)
    else:
        sr_manager_5m.update_levels(latest_candle_5m)
        if latest_candle_4h is not None:
            sr_manager_4h.update_levels(latest_candle_4h)

    levels_5m = sr_manager_5m.get_levels()
    levels_4h = sr_manager_4h.get_levels()

    return make_candlestick_figure(
        partial_5m, partial_4h,
        levels_5m['supports'], levels_5m['resistances'],
        levels_4h['supports'], levels_4h['resistances']
    )

if __name__ == '__main__':
    app.run(debug=True)
    # https://120.0.0.1:8050/
