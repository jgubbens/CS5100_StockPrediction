import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from DataExtract import fetch_binanceus_ohlcv
from SNR_function import SupportResistanceManager

# data extract
df_5m = fetch_binanceus_ohlcv('SOL/USDT', '5m', '2025-06-10T00:00:00Z', '2025-06-20T00:00:00Z')
df_5m['timestamp'] = df_5m.index

df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', '2025-06-10T00:00:00Z', '2025-06-20T00:00:00Z')
df_4h['timestamp'] = df_4h.index

# load q table
with open('5m_q_table.pkl', 'rb') as f:
    q_table_5m = pickle.load(f)

with open('4h_q_table.pkl', 'rb') as f:
    q_table_4h = pickle.load(f)

sr_manager_5m = SupportResistanceManager()
sr_manager_4h = SupportResistanceManager()

position = 0
entry_price = None
entry_index = None
entry_points = []  # entry point(index, price, direction)
last_4h_timestamp = None
last_4h_onehot = [0, 0, 0, 0]

def discretize_state(state, bins=[10, 10, 10, 3, 4]):
    return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

app = Dash(__name__)
total_ticks = len(df_5m)
RECALC_EVERY = 30

app.layout = html.Div([
    html.H1("SOL 5m Simulation with 4h Q-table Guidance"),
    dcc.Graph(id='candlestick-graph'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global position, entry_price, entry_index, last_4h_timestamp, last_4h_onehot, entry_points

    idx_5m = min(n, total_ticks - 1)
    row_5m = df_5m.iloc[idx_5m]
    time_5m = row_5m['timestamp']

    
    df_4h_before = df_4h[df_4h['timestamp'] <= time_5m]
    if not df_4h_before.empty:
        row_4h = df_4h_before.iloc[-1]
        time_4h = row_4h['timestamp']
        if time_4h != last_4h_timestamp:
            last_4h_timestamp = time_4h
            sr_manager_4h.reset_levels()
            sr_manager_4h.calculate_levels(df_4h[df_4h['timestamp'] <= time_4h])
            sr_manager_4h.update_levels(row_4h)

            price_4h = row_4h['close']
            volume_4h = row_4h['volume']
            support_4h = sr_manager_4h.get_closest_support(price_4h)
            resistance_4h = sr_manager_4h.get_closest_resistance(price_4h)

            support_pct_4h = (price_4h - support_4h) / support_4h if support_4h != 0 else 0.0
            resistance_pct_4h = (resistance_4h - price_4h) / resistance_4h if resistance_4h != 0 else 0.0
            volume_pct_4h = volume_4h / df_4h['volume'].max()

            state_4h = discretize_state([support_pct_4h, resistance_pct_4h, volume_pct_4h])
            q4 = q_table_4h.get(state_4h, [0, 0, 1])
            probs = np.exp(q4) / np.sum(np.exp(q4))
            action_4h = np.argmax(probs) if np.max(probs) > 0.3 else 2

            last_4h_onehot = [0, 0, 0, 0]
            last_4h_onehot[action_4h] = 1

    if n % RECALC_EVERY == 0:
        sr_manager_5m.reset_levels()
        sr_manager_5m.calculate_levels(df_5m.iloc[:idx_5m + 1])
    sr_manager_5m.update_levels(row_5m)

    price_5m = row_5m['close']
    volume_5m = row_5m['volume']
    support_5m = sr_manager_5m.get_closest_support(price_5m)
    resistance_5m = sr_manager_5m.get_closest_resistance(price_5m)
    support_pct_5m = (price_5m - support_5m) / support_5m if support_5m != 0 else 0.0
    resistance_pct_5m = (resistance_5m - price_5m) / resistance_5m if resistance_5m != 0 else 0.0
    volume_pct_5m = volume_5m / df_5m['volume'].max()

    combined_state = [support_pct_5m, resistance_pct_5m, volume_pct_5m] + last_4h_onehot
    state_5m = discretize_state(combined_state)

    q5 = q_table_5m.get(state_5m, [0, 0, 1])
    action = int(np.argmax(q5))

    if position == 0:
        if action == 0:
            position = 1
            entry_price = price_5m
            entry_index = idx_5m
            entry_points.append((idx_5m, price_5m, 1))
            print(f"[ENTRY] Long @ {entry_price:.2f}, Index: {entry_index}, Time: {time_5m}")
        elif action == 1:
            position = -1
            entry_price = price_5m
            entry_index = idx_5m
            entry_points.append((idx_5m, price_5m, -1))
            print(f"[ENTRY] Short @ {entry_price:.2f}, Index: {entry_index}, Time: {time_5m}")
    else:
        ret = (price_5m - entry_price) / entry_price
        if position == -1:
            ret = -ret
        if ret >= 0.02 or ret <= -0.02 or action == 2:
            print(f"[EXIT] Close @ {price_5m:.2f}, PnL: {ret:.2%}, Index: {idx_5m}")
            position = 0
            entry_price = None
            entry_index = None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.05,
                        subplot_titles=("5m Chart", "4h Chart"), row_heights=[0.5, 0.5])

    start_idx_5m = max(0, idx_5m - 299)
    df_5m_window = df_5m.iloc[start_idx_5m:idx_5m + 1]
    entry_points = [entry for entry in entry_points if entry[0] >= start_idx_5m]

    fig.add_trace(go.Candlestick(
        x=df_5m_window['timestamp'],
        open=df_5m_window['open'],
        high=df_5m_window['high'],
        low=df_5m_window['low'],
        close=df_5m_window['close'],
        name="5m"
    ), row=1, col=1)

    for s in sr_manager_5m.get_levels()['supports']:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=1, col=1)
    for r in sr_manager_5m.get_levels()['resistances']:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=1, col=1)

    df_4h_before = df_4h[df_4h['timestamp'] <= time_5m]
    fig.add_trace(go.Candlestick(
        x=df_4h_before['timestamp'],
        open=df_4h_before['open'],
        high=df_4h_before['high'],
        low=df_4h_before['low'],
        close=df_4h_before['close'],
        name="4h"
    ), row=2, col=1)

    for s in sr_manager_4h.get_levels()['supports']:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=2, col=1)
    for r in sr_manager_4h.get_levels()['resistances']:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=2, col=1)

    for idx, price, direction in entry_points:
        if idx <= idx_5m:
            x_entry = df_5m['timestamp'].iloc[idx]
            arrow_color = "green" if direction == 1 else "red"
            symbol = "triangle-up" if direction == 1 else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[x_entry],
                y=[price],
                mode="markers+text",
                marker=dict(color=arrow_color, size=12, symbol=symbol),
                text=["Entry"],
                textposition="bottom center",
                name="Entry Point"
            ), row=1, col=1)

    fig.update_layout(
        height=800,
        template='plotly_white',
        xaxis_rangeslider=dict(visible=False),
        xaxis2_rangeslider=dict(visible=False)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
    # https://120.0.0.1:8050/