import numpy as np
import gym
from gym import spaces
from SNR_function import SupportResistanceManager

# Class to simulate the stock and/or trading crypto market for 4 hour candlesticks
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

        # Observation Space: [resistance_diff_pct, support_diff_pct, volume%, current position]
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

        # misc parameters
        self.trade_count = 0
        self.total_profit = 0.0

    # Method to reset the environment
    def reset(self):
        self.position = 0
        self.current_step = 6
        self.balance = 100.0
        self.position_price = 0.0
        self.sr_manager.reset_levels()

        self.trade_count = 0
        self.total_profit = 0.0

        return self._get_observation()
    
    # Likely can remove - occurs in training
    def discretize_state(self, state, bins=[10, 10, 10]):
        # bucket
        return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

    # Getting observations from state with the help of the support and resistance manager 
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

        obs = np.array([support_pct, resistance_pct, volume_pct, self.position], dtype=np.float32)
        return obs

    # Step function
    def step(self, action):
        done = False
        realized_reward = 0.0  # rewards after close
        unrealized_reward = 0.0  # on hold unrealized reward

        row = self.df.iloc[self.current_step]
        price = row['close']

        # Close all positions if the balance drops below 0 (in debt)
        if self.balance <= 0:
            action = 2

        # Stop loss (SL) and Take profit (TP) calculations
        if self.position != 0:
            ret_pct = (price - self.position_price) / self.position_price
            if self.position == -1:
                ret_pct = -ret_pct

            if ret_pct <= self.stop_loss_pct or ret_pct >= self.take_profit_pct:
                realized_reward = self._close_position(price)

            else:
                unrealized_reward = ret_pct

        # Execute action
        if action == 0:  # call / long
            self.position = 1
            self.position_price = price
        elif action == 1:  # short
            self.position = -1
            self.position_price = price
        elif action == 2:  # hold
            pass

        # Increment to next step and close all positions before the epoch ends
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            if self.position != 0:
                realized_reward += self._close_position(price)

        # Integration of the _simulate_trade_until_close method in our step function
        if self.position == 0 and action in [0, 1]:
            simulated_position = 1 if action == 0 else -1
            simulated_reward = self._simulate_trade_until_close(self.current_step, self.df.iloc[self.current_step]['close'], simulated_position)
        else:
            simulated_reward = 0.0  # If already in position, don't simulate new one

        # Reward calculation 
        # - Unrealized_reward and simulated_reward were utilized in training, effectively allowing our agent to "cheat"
        # and look ahead. 
        # - In actual simulation, only realized reward matters - cannot look ahead in real time market

        # For n steps: 
        # n_steps = self._simulate_trade_for_n_steps(10)
        # reward = realized_reward + unrealized_reward + (0.95 * frs)

        # For simulate until close:
        reward = realized_reward + unrealized_reward + (0.95 * simulated_reward)

        # Actual real-time market reward structure:
        # reward = realized_reward

        obs = self._get_observation()
        info = {'balance': self.balance, 
                'position': self.position, 
                'trade_count': self.trade_count,
                'avg_profit_per_trade': self.total_profit / self.trade_count if self.trade_count > 0 else 0.0}

        return obs, reward, done, info
    
    # A lookahead feature for training - checks proft at next 'n' steps
    def _simulate_trade_for_n_steps(self, n):
        future_steps = self.df.iloc[self.current_step : self.current_step + n]
        if len(future_steps) == 0:
            return 0.0
    
        future_prices = future_steps['close'].values
        if self.position == 1:  # Long
            returns = (future_prices - self.position_price) / self.position_price
        elif self.position == -1:  # Short
            returns = (self.position_price - future_prices) / self.position_price
        else:
            return 0.0

        return np.max(returns)  # or np.mean(returns), or weighted
    
    # A lookahead feature used in training - checks profit at next close
    def _simulate_trade_until_close(self, entry_step, entry_price, position):
        stop_loss = self.stop_loss_pct
        take_profit = self.take_profit_pct
        df_len = len(self.df)

        for i in range(entry_step + 1, df_len):
            price = self.df.iloc[i]['close']
            ret_pct = (price - entry_price) / entry_price
            if position == -1:
                ret_pct = -ret_pct

            if ret_pct <= stop_loss or ret_pct >= take_profit:
                return ret_pct  # SL or TP hit

        # If never closed, return P/L at final price
        final_price = self.df.iloc[-1]['close']
        ret_pct = (final_price - entry_price) / entry_price
        if position == -1:
            ret_pct = -ret_pct
        return ret_pct

    # No longer have close as an action - now close at SL/TP
    # def _close_position(self, price):
    #     if self.position == 1:
    #         profit = price - self.position_price
    #     elif self.position == -1:
    #         profit = self.position_price - price
    #     else:
    #         profit = 0.0
    #     # print("Close Price:", price)
    #     self.balance += profit
    #     self.position = 0
    #     self.position_price = 0.0
    #     return profit
    
    # Method to close the position
    def _close_position(self, price):
        if self.position == 1:
            profit = ((price - self.position_price)/self.position_price) * self.balance * 0.1
        elif self.position == -1:
            profit = ((self.position_price - price)/self.position_price) * self.balance * 0.1
        else:
            profit = 0.0

        self.balance += profit
        self.total_profit += profit
        self.trade_count += 1

        self.position = 0
        self.position_price = 0.0
        return profit
