import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pattern_recognition')))

from pattern_recognition import SupportResistanceCNN



class SupportResistanceManager:

    def __init__(self, max_levels=5,
                 strong_peak_distance=60, 
                 strong_peak_prominence=20,
                 peak_distance=5, 
                 peak_rank_width=2, 
                 resistance_min_pivot_rank=3):

        self.max_levels = max_levels

        self.strong_peak_distance = strong_peak_distance
        self.strong_peak_prominence = strong_peak_prominence

        self.peak_distance = peak_distance
        self.peak_rank_width = peak_rank_width
        self.resistance_min_pivot_rank = resistance_min_pivot_rank

        self.model = SupportResistanceCNN(sequence_length=300)
        self.model.load_state_dict(torch.load("../pattern_recognition/support_resistance_cnn_300candlesticks.pt", map_location=torch.device("cpu")))

        self.supports = []
        self.resistances = []

    def calculate_levels(self, df):
        if len(df) < 300:
            print("Not enough candlesticks for model input — skipping prediction.")
            self.supports = []
            self.resistances = []
            return
        df_norm = (df - df.mean()) / df.std()
        features = df_norm[['open', 'high', 'low', 'close', 'volume']].values.T
        x = torch.tensor(features[:, -300:], dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred_h, _ = self.model(x)
            probs = torch.sigmoid(pred_h).squeeze(0).numpy()

        levels = np.linspace(df['low'].min(), df['high'].max(), 100)
        last_close = df['close'].iloc[-1]

        support_levels = [(lvl, prob) for lvl, prob in zip(levels, probs) if lvl < last_close]
        resistance_levels = [(lvl, prob) for lvl, prob in zip(levels, probs) if lvl >= last_close]

        top_supports = sorted(support_levels, key=lambda x: x[1], reverse=True)[:self.max_levels]
        top_resistances = sorted(resistance_levels, key=lambda x: x[1], reverse=True)[:self.max_levels]

        self.supports = [lvl for lvl, _ in top_supports]
        self.resistances = [lvl for lvl, _ in top_resistances]

    '''
    def calculate_levels(self, df):
        # Strong Resistance
        strong_peaks, _ = find_peaks(df['high'], distance=self.strong_peak_distance, prominence=self.strong_peak_prominence)
        strong_peaks_values = df.iloc[strong_peaks]['high'].tolist()
        yearly_high = df['high'].iloc[-252:].max()
        strong_peaks_values.append(yearly_high)

        # Strong Support
        strong_troughs, _ = find_peaks(-df['low'], distance=self.strong_peak_distance, prominence=self.strong_peak_prominence)
        strong_troughs_values = df.iloc[strong_troughs]['low'].tolist()
        yearly_low = df['low'].iloc[-252:].min()
        strong_troughs_values.append(yearly_low)

        # Normal Resistance
        peaks, _ = find_peaks(df['high'], distance=self.peak_distance)
        peak_to_rank = {peak: 0 for peak in peaks}
        for i, current_peak in enumerate(peaks):
            current_high = df.iloc[current_peak]['high']
            for previous_peak in peaks[:i]:
                if abs(current_high - df.iloc[previous_peak]['high']) <= self.peak_rank_width:
                    peak_to_rank[current_peak] += 1 # claculate the the amount price has touch the resistance

        resistances = strong_peaks_values.copy()
        for peak, rank in peak_to_rank.items():
            if rank >= self.resistance_min_pivot_rank: # if the touch amount exceed the threshold then considered as a resistance
                resistances.append(df.iloc[peak]['high'] + 1e-3)
        resistances.sort()

        resistance_bins = []
        current_bin = [resistances[0]]

        # merge the nearby indicator
        for r in resistances[1:]:
            if r - current_bin[-1] < self.peak_rank_width:
                current_bin.append(r)
            else:
                resistance_bins.append(current_bin)
                current_bin = [r]
        resistance_bins.append(current_bin)
        final_resistances = [np.mean(bin) for bin in resistance_bins]

        # Normal Support
        troughs, _ = find_peaks(-df['low'], distance=self.peak_distance)
        trough_to_rank = {trough: 0 for trough in troughs}
        for i, current_trough in enumerate(troughs):
            current_low = df.iloc[current_trough]['low']
            for previous_trough in troughs[:i]:
                if abs(current_low - df.iloc[previous_trough]['low']) <= self.peak_rank_width:
                    trough_to_rank[current_trough] += 1  # claculate the the amount price has touch the support


        supports = strong_troughs_values.copy()
        for trough, rank in trough_to_rank.items():
            if rank >= self.resistance_min_pivot_rank:
                supports.append(df.iloc[trough]['low'] - 1e-3)# if the touch amount exceed the threshold then considered as a support
        supports.sort()

        support_bins = []
        current_bin = [supports[0]]

        # merge the nearby indicator
        for s in supports[1:]:
            if s - current_bin[-1] < self.peak_rank_width:
                current_bin.append(s)
            else:
                support_bins.append(current_bin)
                current_bin = [s]
        support_bins.append(current_bin)
        final_supports = [np.mean(bin) for bin in support_bins]

        self.supports = final_supports[:self.max_levels] # get the five nearest resistance line
        self.resistances = final_resistances[-self.max_levels:] # get the five nearest support line
    '''

    def update_levels(self, latest_candle):
        # if the price penetrate support line support will be the new resistance
        # if the price penetrate resistance line resistance will be the new support
        close = latest_candle['close']
        open_ = latest_candle['open']

        new_supports = []
        new_resistances = []

        for s in self.supports:
            if close < s:
                new_resistances.append(s)
            else:
                new_supports.append(s)

        for r in self.resistances:
            if open_ > r:
                new_supports.append(r)
            else:
                new_resistances.append(r)

        # merge the duplicate value
        self.supports = self._unique_keep_order(new_supports)
        self.resistances = self._unique_keep_order(new_resistances)

        supports_near = [s for s in self.supports if s <= close]
        self.supports = sorted(supports_near, key=lambda x: abs(close - x))[:self.max_levels]

        resistances_near = [r for r in self.resistances if r >= close]
        self.resistances = sorted(resistances_near, key=lambda x: abs(close - x))[:self.max_levels]

    def _unique_keep_order(self, seq):
        seen = set()
        result = []
        # data cleanse
        for x in seq:
            if isinstance(x, (float, int)) and not pd.isna(x) and x not in seen:
                seen.add(x)
                result.append(x)
        return result

    def get_levels(self):
        return {'supports': self.supports, 'resistances': self.resistances}

    def get_closest_support(self, price):
        supports = [s for s in self.supports if s <= price]
        # return current price if support doesn't exist
        if not supports:
            return price
        return min(supports, key=lambda x: abs(price - x))

    def get_closest_resistance(self, price):
        resistances = [r for r in self.resistances if r >= price]
        # return current price if resistace doesn't exist
        if not resistances:
            return price
        return min(resistances, key=lambda x: abs(price - x))

    def reset_levels(self):
        self.supports = []
        self.resistances = []

