import ccxt
import pandas as pd
'''
! pip install ccxt
'''

def fetch_binanceus_ohlcv(symbol='SOL/USDT', timeframe='5m', start_time='2023-01-01T00:00:00Z', end_time=None, limit=1000):
    # history data extract and concat
    exchange = ccxt.binanceus()
    since = exchange.parse8601(start_time)
    end_ts = exchange.parse8601(end_time) if end_time else None
    
    all_ohlcv = []
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        
        last_ts = ohlcv[-1][0]
        since = last_ts + 1 
        
        if end_ts and last_ts >= end_ts:
            break
        if len(ohlcv) < limit:
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    if end_ts:
        df.index = pd.to_datetime(df.index).tz_localize(None)
        end_time_ts = pd.to_datetime(end_time).tz_localize(None)
        df = df[df.index <= pd.to_datetime(end_time_ts)]
    
    return df
