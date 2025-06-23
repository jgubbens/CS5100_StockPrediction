import numpy as np
import gym
from gym import spaces
from SNR_function import SupportResistanceManager

class TradingEnv_4h(gym.Env):
    def __init__(self, df_4h):
        super().__init__()

        self.df = df_4h.reset_index(drop=True)
        self.max_volume = self.df['volume'].max()

        self.sr_manager = SupportResistanceManager()
        self.sr_manager.calculate_levels(self.df)

        # action space [0: call, 1: short, 2: hold]
        '''
        remove close position in action space
        '''
        self.action_space = spaces.Discrete(3)

        # observation: [resistance_diff_pct, support_diff_pct, volume%, current position]
        low = np.array([-np.inf, -np.inf, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # initial setting
        self.position = 0 
        self.current_step = 0
        self.balance = 100.0  # initial balance
        self.position_price = 0.0  
        self.stop_loss_pct = -0.1  # Stop loss at -10%
        self.take_profit_pct = 0.1  # target price at +20%

    def reset(self):
        self.position = 0
        self.current_step = 6 
        self.balance = 100.0
        self.position_price = 0.0
        self.sr_manager.reset_levels()
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = row['close']
        volume = row['volume']

        self.sr_manager.update_levels(row)

        support = self.sr_manager.get_closest_support(price)
        resistance = self.sr_manager.get_closest_resistance(price)

        support_pct = (price - support) / support if support != 0 else 0.0
        resistance_pct = (resistance - price) / resistance if resistance != 0 else 0.0
        volume_pct = volume / self.max_volume if self.max_volume != 0 else 0.0

        obs = np.array([support_pct, resistance_pct, volume_pct], dtype=np.float32)
        return obs

    def step(self, action):
        done = False
        realized_reward = 0.0  # rewards after close
        unrealized_reward = 0.0  # on hold unrealized reward

        row = self.df.iloc[self.current_step]
        price = row['close']

        # close every thing if the balance drop below 0 (in debt)
        if self.balance <= 0:
            action = 2

        # SL/TP
        if self.position != 0:
            ret_pct = (price - self.position_price) / self.position_price
            if self.position == -1:
                ret_pct = -ret_pct

            if ret_pct <= self.stop_loss_pct or ret_pct >= self.take_profit_pct:
                realized_reward = self._close_position(price)

            else:
                unrealized_reward = ret_pct


        # action space
        if action == 0: # call
            if self.position == 0:
                self.position = 1
                self.position_price = price
                print("Call Enter Price:", price)
        elif action == 1:  # short
            if self.position == 0:
                self.position = -1
                self.position_price = price
                print("Short Enter Price:", price)
        elif action == 2:  # hold
            pass

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            if self.position != 0:
                realized_reward += self._close_position(price) # close all the position before epoch ends

        obs = self._get_observation()
        info = {'balance': self.balance, 'position': self.position}
        reward = realized_reward + unrealized_reward

        return obs, reward, done, info

    def _close_position(self, price):
        if self.position == 1:
            profit = price - self.position_price
        elif self.position == -1:
            profit = self.position_price - price
        else:
            profit = 0.0
        print("Close Price:", price)
        self.balance += profit
        self.position = 0
        self.position_price = 0.0
        return profit