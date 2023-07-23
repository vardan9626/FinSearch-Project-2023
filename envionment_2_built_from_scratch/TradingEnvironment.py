import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# noinspection PyAttributeOutsideInit
class StockTrading:
    def __init__(self, df, window, frame_bound):
        # Initialize the stock trading environment with the provided DataFrame,
        # window size, and frame bounds.

        self.df = df
        if window is None:
            window = 1
        self.window_size = window
        self.frame_bound = frame_bound

        self.curr_position = 0
        self.begin = frame_bound[0]
        self.current_tick = self.begin
        self.end = frame_bound[1]
        self.total_ticks = self.end - self.begin + 1
        self.num_ticks = 1

        # Calculate the price difference ('diff') for each day and store it in the DataFrame
        self.df['diff'] = self.df['open'].diff()
        self.df['prev_high'] = self.df['high'].shift(1)
        self.df['prev_close'] = self.df['close'].shift(1)
        self.df['prev_low'] = self.df['low'].shift(1)
        self.df['SMA_5'] = self.df['close'].rolling(window=5).mean().shift(1)

        # Exponential Moving Average (EMA) for last 5 days
        self.df['EMA_5'] = self.df['close'].ewm(span=5, adjust=False).mean().shift(1)

        # Relative Strength Index (RSI)
        delta = self.df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        self.df['RSI'] = 100 - (100 / (1 + rs))

        self.df.fillna(0, inplace=True)
        assert self.begin > window, "Please start the frame_bound from a position greater than window_size"

    def get_observation_table(self):
        # Get the observation table containing the opening price and 'diff' for 'window_size' number of days

        start_idx = self.current_tick - self.window_size + 1
        end_idx = self.current_tick
        observation_table = self.df.loc[start_idx:end_idx, ['open', 'diff', 'prev_close', 'prev_low', 'prev_high', 'SMA_5', 'EMA_5', 'RSI']].values
        # Create a scaler object
        scaler = MinMaxScaler()

        # Fit and transform the data
        observation_table_normalized = scaler.fit_transform(observation_table)
        return observation_table_normalized

    def reset(self, cash, long, short):
        # Reset the environment to the initial state with the provided initial cash,
        # long holdings, and short holdings.

        self.current_tick = self.begin
        self.num_ticks = 1
        self.done = 0
        self.cash = cash
        self.long_holding = long
        self.short_holding = short
        self.initial_net_val = self.cash + self.long_holding * self.df.iloc[self.current_tick - 1]['open'] - self.short_holding * self.df.iloc[self.current_tick - 1]['open']
        self.curr_net_val = self.initial_net_val
        self.risk = []
        self.profit = []
        self.info = {"cash": cash, "Long Holdings": long, "Short Holdings": short, "Profit": 0, "Reward": 0,
                     "Open Price": self.df.iloc[self.current_tick - 1]['open'], "Risk": "NA"}
        return self.get_observation_table(), self.info

    def get_reward(self):
        # Calculate the relative profit
        relative_profit = (self.curr_net_val / self.initial_net_val - 1)

        # Reward for holding onto profitable assets
        hold_reward = 0
        if self.curr_net_val > self.initial_net_val:
            hold_reward = 0.01  # Small reward for each step holding a profitable asset

        reward = relative_profit + hold_reward

        return reward

    def step(self, action):
        # Perform a step in the environment based on the given action.

        open_today = self.df.iloc[self.current_tick - 1]['open']
        risk = 0

        if self.current_tick == self.end:
            self.done = 1

        self.current_tick += 1

        if action[0] > 0:
            # Buy long position
            self.cash += self.long_holding * open_today * action[0]
            self.long_holding *= (1 - action[0])
        else:
            # Sell long position
            self.long_holding += (self.cash * (-action[0]) / open_today)
            self.cash += self.cash * action[0]
            risk += (self.cash * (-action[0]) / open_today) * open_today * 0.05

        if action[1] > 0:
            # Buy short position
            self.cash += self.short_holding * open_today * action[1]
            self.short_holding *= (1 - action[1])
        else:
            # Sell short position
            self.short_holding += (self.cash * (-action[1]) / open_today)
            self.cash += self.cash * action[1]
            risk += (self.cash * (-action[1]) / open_today) * open_today * 0.05

        value = self.cash + self.long_holding * open_today - self.short_holding * open_today
        self.curr_net_val = value
        reward = self.get_reward()
        profit = value - self.initial_net_val
        self.profit.append(profit)
        self.risk.append(risk)
        self.curr_net_val = value

        self.info = {"cash": round(self.cash, 4), "Long Holdings": round(self.long_holding, 4),
                     "Short Holdings": round(self.short_holding, 4), "Profit": round(profit, 2),
                     "Reward": round(reward, 2), "Open Price": round(open_today, 2), "Risk": round(risk, 2)}

        return self.get_observation_table(), reward, self.done, self.info

    def random_action(self):
        # Generate a random action within the range [-1, 1]
        return np.random.uniform(-1, 1, size=(2,))

    def percentageReturn(self):
        # Calculate the percentage return of the current net value compared to the initial net value.
        return round((self.curr_net_val / self.initial_net_val - 1) * 100, 2)

    def render(self):
        # Create a 1x2 subplot grid and plot profit and risk over time.
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].plot(self.profit, label='Profit', color='green')
        axs[0].set_title('Profit Plot')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Profit')
        axs[0].legend()

        axs[1].plot(self.risk, label='Risk', color='red')
        axs[1].set_title('Risk Plot')
        axs[1].set_xlabel('Days')
        axs[1].set_ylabel('Risk')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
