import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from DataExtract import fetch_binanceus_ohlcv
from SNR_function import SupportResistanceManager

# Load Q-tables once at start
with open('4h_q_table.pkl', 'rb') as f:
    q_table_4h = pickle.load(f)
with open('5m_q_table.pkl', 'rb') as f:
    q_table_5m = pickle.load(f)

# Fetch OHLCV data once
df_5m = fetch_binanceus_ohlcv('SOL/USDT', '5m', start_time='2025-01-18T00:00:00Z', end_time='2025-06-20T00:00:00Z')
df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', start_time='2025-01-18T00:00:00Z', end_time='2025-06-20T00:00:00Z')

# Reset index for integer access
df_5m = df_5m.reset_index(drop=False)
df_4h = df_4h.reset_index(drop=False)

max_volume_5m = df_5m['volume'].max()
max_volume_4h = df_4h['volume'].max()

# SupportResistanceManager instances
sr_manager_5m = SupportResistanceManager()
sr_manager_4h = SupportResistanceManager()

# Discretize functions
def discretize_state_4h(state, bins=[10,10,10,3]):
    return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))
def discretize_state_5m(state, bins=[10,10,10,3,4]):
    return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

def get_closest_support_resistance(sr_manager, price):
    return sr_manager.get_closest_support(price), sr_manager.get_closest_resistance(price)

def predict_4h_action(timestamp, df_4h_partial, sr_manager_4h, max_volume_4h, q_table_4h):
    df_4h_sub = df_4h_partial[df_4h_partial['timestamp'] <= timestamp]
    if df_4h_sub.empty:
        return 2
    row = df_4h_sub.iloc[-1]
    price, volume = row['close'], row['volume']
    support, resistance = get_closest_support_resistance(sr_manager_4h, price)
    support_pct = (price - support) / support if support != 0 else 0.0
    resistance_pct = (resistance - price) / resistance if resistance != 0 else 0.0
    volume_pct = volume / max_volume_4h if max_volume_4h != 0 else 0.0
    state_4h = np.array([support_pct, resistance_pct, volume_pct, 0])
    state_d = discretize_state_4h(state_4h)
    if state_d in q_table_4h:
        q_values = q_table_4h[state_d]
        probs = np.exp(q_values) / np.sum(np.exp(q_values))
        action = np.argmax(probs) if np.max(probs) > 0.3 else 2
    else:
        action = 2
    return action

def make_candlestick_figure(df5, df4, supports_5m, resistances_5m, supports_4h, resistances_4h,
                            entry_5m_idx, entry_5m_prices, entry_5m_actions,
                            entry_4h_idx, entry_4h_prices, entry_4h_actions):

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        vertical_spacing=0.07,
                        subplot_titles=('5-Minute Chart', '4-Hour Chart'),
                        row_heights=[0.5, 0.5])

    # 5m candles
    fig.add_trace(go.Candlestick(x=df5['timestamp'], open=df5['open'], high=df5['high'],
                                 low=df5['low'], close=df5['close'], name='5m Candles'),
                  row=1, col=1)
    # 5m S/R lines
    for s in supports_5m:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=1, col=1)
    for r in resistances_5m:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=1, col=1)
    # 5m entries
    for ts, price, action in zip(entry_5m_idx, entry_5m_prices, entry_5m_actions):
        symbol = 'triangle-up' if action == 0 else 'triangle-down'
        color = 'green' if action == 0 else 'red'
        fig.add_trace(go.Scatter(x=[ts], y=[price], mode='markers',
                                 marker=dict(symbol=symbol, color=color, size=12),
                                 name='5m Entry'))

    # 4h candles
    fig.add_trace(go.Candlestick(x=df4['timestamp'], open=df4['open'], high=df4['high'],
                                 low=df4['low'], close=df4['close'], name='4h Candles'),
                  row=2, col=1)
    # 4h S/R lines
    for s in supports_4h:
        fig.add_hline(y=s, line_dash='dot', line_color='green', row=2, col=1)
    for r in resistances_4h:
        fig.add_hline(y=r, line_dash='dash', line_color='red', row=2, col=1)
    # 4h entries
    for ts, price, action in zip(entry_4h_idx, entry_4h_prices, entry_4h_actions):
        symbol = 'triangle-up' if action == 0 else 'triangle-down'
        color = 'green' if action == 0 else 'red'
        fig.add_trace(go.Scatter(x=[ts], y=[price], mode='markers',
                                 marker=dict(symbol=symbol, color=color, size=12),
                                 name='4h Entry'))

    fig.update_layout(height=900, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

# Dash app setup
app = Dash(__name__)
RECALC_EVERY = 30
MAX_CANDLES = 200  # Max candles to display/update per update tick

app.layout = html.Div([
    html.H1("SOL/USDT Trading Simulation with Q-Table Entry Points"),
    dcc.Graph(id='candlestick-graph'),
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)  # update every 2 seconds
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    # Limit how much data we visualize to last MAX_CANDLES candles for performance
    end_idx_5m = min(n+1, len(df_5m))
    start_idx_5m = max(0, end_idx_5m - MAX_CANDLES)
    partial_5m = df_5m.iloc[start_idx_5m:end_idx_5m]

    if partial_5m.empty:
        return go.Figure()

    # Filter 4h candles up to latest 5m timestamp
    latest_ts_5m = partial_5m['timestamp'].iloc[-1]
    partial_4h = df_4h[df_4h['timestamp'] <= latest_ts_5m]

    # On recalculation interval reset SR managers and recalc levels
    if n == 0 or n % RECALC_EVERY == 0:
        sr_manager_5m.reset_levels()
        sr_manager_5m.calculate_levels(partial_5m)
        sr_manager_4h.reset_levels()
        sr_manager_4h.calculate_levels(partial_4h)

    # Update SR levels with last candle only
    sr_manager_5m.update_levels(partial_5m.iloc[-1])
    if not partial_4h.empty:
        sr_manager_4h.update_levels(partial_4h.iloc[-1])

    levels_5m = sr_manager_5m.get_levels()
    levels_4h = sr_manager_4h.get_levels()

    # Find entry points on 5m
    entry_5m_idx = []
    entry_5m_prices = []
    entry_5m_actions = []

    for _, row in partial_5m.iterrows():
        price = row['close']
        timestamp = row['timestamp']
        volume = row['volume']

        support, resistance = get_closest_support_resistance(sr_manager_5m, price)
        support_pct = (price - support) / support if support != 0 else 0.0
        resistance_pct = (resistance - price) / resistance if resistance != 0 else 0.0
        volume_pct = volume / max_volume_5m if max_volume_5m != 0 else 0.0

        # Predict 4h action for 5m state
        action_4h = predict_4h_action(timestamp, partial_4h, sr_manager_4h, max_volume_4h, q_table_4h)
        one_hot_4h = np.zeros(4)
        if action_4h in [0, 1, 2, 3]:
            one_hot_4h[action_4h] = 1

        state_5m = np.array([support_pct, resistance_pct, volume_pct, *one_hot_4h])
        state_d_5m = discretize_state_5m(state_5m)

        if state_d_5m in q_table_5m:
            q_values = q_table_5m[state_d_5m]
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            predicted_action = np.argmax(probs) if np.max(probs) > 0.3 else 2
        else:
            predicted_action = 2

        if predicted_action in [0, 1]:  # only mark call/short
            entry_5m_idx.append(timestamp)
            entry_5m_prices.append(price)
            entry_5m_actions.append(predicted_action)

    # Find entry points on 4h
    entry_4h_idx = []
    entry_4h_prices = []
    entry_4h_actions = []

    for _, row in partial_4h.iterrows():
        price = row['close']
        timestamp = row['timestamp']
        volume = row['volume']

        support, resistance = get_closest_support_resistance(sr_manager_4h, price)
        support_pct = (price - support) / support if support != 0 else 0.0
        resistance_pct = (resistance - price) / resistance if resistance != 0 else 0.0
        volume_pct = volume / max_volume_4h if max_volume_4h != 0 else 0.0

        state_4h = np.array([support_pct, resistance_pct, volume_pct, 0])
        state_d_4h = discretize_state_4h(state_4h)

        if state_d_4h in q_table_4h:
            q_values = q_table_4h[state_d_4h]
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            predicted_action = np.argmax(probs) if np.max(probs) > 0.3 else 2
        else:
            predicted_action = 2

        if predicted_action in [0, 1]:
            entry_4h_idx.append(timestamp)
            entry_4h_prices.append(price)
            entry_4h_actions.append(predicted_action)

    return make_candlestick_figure(
        partial_5m, partial_4h,
        levels_5m['supports'], levels_5m['resistances'],
        levels_4h['supports'], levels_4h['resistances'],
        entry_5m_idx, entry_5m_prices, entry_5m_actions,
        entry_4h_idx, entry_4h_prices, entry_4h_actions
    )

if __name__ == '__main__':
    app.run(debug=True)
