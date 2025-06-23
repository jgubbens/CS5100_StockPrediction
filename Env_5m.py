import numpy as np
import gym
from gym import spaces
from SNR_function import SupportResistanceManager

class TradingEnv_5m(gym.Env):
    def __init__(self, df_5m, df_4h, q_table_4h):
        super().__init__()

        self.df_4h = df_4h.reset_index(drop=True)
        self.df = df_5m.reset_index(drop=True)
        self.max_volume = self.df['volume'].max()
        self.max_volume_4h = self.df_4h['volume'].max()

        self.q_table_4h = q_table_4h
        self.sr_manager = SupportResistanceManager()
        self.sr_manager.calculate_levels(self.df)

        self.sr_manager_4h = SupportResistanceManager()
        self.sr_manager_4h.calculate_levels(self.df_4h)

        # action space [0: call, 1: short, 2: hold]
        self.action_space = spaces.Discrete(3)

        # observation: [resistance_diff_pct, support_diff_pct, volume%, 4h_decition]
        low = np.array([-np.inf, -np.inf, 0] + [0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 1] + [1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.position = 0
        self.current_step = 0
        self.balance = 100.0 # initial balance
        self.position_price = 0.0
        self.stop_loss_pct = -0.02 # Stop loss at -2%
        self.take_profit_pct = 0.02 # Target Price at +2%

    def reset(self):
        self.position = 0
        self.current_step = 6 # start with 6 k bar to get the first SNR
        self.balance = 100.0
        self.position_price = 0.0
        self.sr_manager.reset_levels()
        return self._get_observation()

    def discretize_state(self, state, bins=[10, 10, 10]):
        # bucket
        return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

    def _get_4h_state(self, row):
        # get 4hr state
        price = row['close']
        volume = row['volume']
        support = self.sr_manager_4h.get_closest_support(price)
        resistance = self.sr_manager_4h.get_closest_resistance(price)
        support_pct = (price - support) / support if support != 0 else 0.0
        resistance_pct = (resistance - price) / resistance if resistance != 0 else 0.0
        volume_pct = volume / self.max_volume_4h if self.max_volume_4h != 0 else 0.0
        return np.array([support_pct, resistance_pct, volume_pct, self.position])

    def _predict_4h_action(self, timestamp):
        row_4h = self.df_4h[self.df_4h['timestamp'] <= timestamp].iloc[-1]
        state = self._get_4h_state(row_4h)
        state_d = self.discretize_state(state)

        if state_d in self.q_table_4h:
            q_values = self.q_table_4h[state_d]
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            action = np.argmax(probs) if np.max(probs) > 0.3 else 2
        else:
            action = 2

        one_hot = np.zeros(4)
        one_hot[action] = 1
        return one_hot

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

        pred_4h = self._predict_4h_action(row['timestamp'])

        obs = np.concatenate([[support_pct, resistance_pct, volume_pct], pred_4h]).astype(np.float32)
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
        
        # action definition
        if action == 0: # call
            if self.position == 0:
                self.position = 1
                self.position_price = price
                # print("Call Enter Price:", price)
        elif action == 1:  # short
            if self.position == 0:
                self.position = -1
                self.position_price = price
                # print("Short Enter Price:", price)
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

        # calculate the reward for each step 
        '''
        We need to tune this part
        
        reward = realized_reward + unrealized_reward

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price
        }
        '''

        return obs, reward, done, info

    def _close_position(self, price):
        if self.position == 1:
            profit = price - self.position_price
        elif self.position == -1:
            profit = self.position_price - price
        else:
            profit = 0.0
        self.balance += profit
        self.position = 0
        self.position_price = 0.0
        return profit
