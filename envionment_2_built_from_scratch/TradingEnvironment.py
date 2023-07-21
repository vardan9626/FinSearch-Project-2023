import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class StockTrading:
    def __init__(
        
        self,
        df,
        window,
        frame_bound,
   
    ):
        # Lets define some of the initial parameters 
        
        self.df=df
        if window==None:
            window=1
        self.window_size=window
        self.frame_bound = frame_bound
        
        

        self.curr_position=0
        self.begin=frame_bound[0]
        self.current_tick=self.begin
        self.end=frame_bound[1]
        self.total_ticks=self.end-self.begin+1
        self.num_ticks=1
        self.df['diff']=self.df['open'].diff()
        assert self.begin>window, "Please start the frame_bound from a position greater than window_size"
        
    def get_observation_table(self):
        # Get the observation table containing closing price and 'diff' for 'window_size' number of days
            start_idx = self.current_tick - self.window_size + 1
            end_idx = self.current_tick
            observation_table = self.df.loc[start_idx:end_idx, ['open', 'diff']].values
            return observation_table
    
    def reset(
        
        self,
        cash,
        long,
        short
        
        ):
        
        self.current_tick=self.begin
        self.num_ticks=1
        self.done=0
        self.cash=cash
        self.long_holding=long
        self.short_holding=short
        self.initial_net_val=self.cash+self.long_holding*self.df.iloc[self.current_tick-1]['open']-self.short_holding*self.df.iloc[self.current_tick-1]['open']
        self.curr_net_val=self.initial_net_val
        self.risk=[]
        self.profit=[]
        self.info = {"cash":cash,"Long Holdings":long,"Short Holdings":short,"Profit":0,"Reward":0,"Open Price":self.df.iloc[self.current_tick-1]['open'],"Risk":"NA"}
        return self.get_observation_table(),self.info
    
    def get_reward(self,value):
        return (value-self.curr_net_val)*0.1
        
    def step(self,action):
        open_today=self.df.iloc[self.current_tick-1]['open']

        risk=0
        if self.current_tick==self.end:
            self.done=1
        
        self.current_tick+=1
        
        if action[0]>0:
            self.cash+=(self.long_holding)*open_today*action[0]
            self.long_holding*=(1-action[0])
        else:
            self.long_holding+=(self.cash*(-action[0])/open_today)
            self.cash+=self.cash*action[0]
            risk+=(self.cash*(-action[0])/open_today)*(open_today)*0.05
        
        if action[1]>0:
            self.cash+=(self.short_holding)*open_today*action[1]
            self.short_holding*=(1-action[1])
        else:
            self.short_holding+=(self.cash*(-action[1])/open_today)
            self.cash+=self.cash*action[1]
            risk+=(self.cash*(-action[1])/open_today)*open_today*0.05
      
        value=self.cash+self.long_holding*open_today-self.short_holding*open_today
        
        reward=self.get_reward(value)+risk
        profit = value-self.initial_net_val
        self.profit.append(profit)
        self.risk.append(risk)
        self.curr_net_val=value
        
        self.info= {"cash":round(self.cash,4),"Long Holdings":round(self.long_holding,4),"Short Holdings":round(self.short_holding,4),"Profit":round(profit,2),"Reward":round(reward,2),"Open Price":round(open_today,2),"Risk":round(risk,2)}
        
        return self.get_observation_table(),reward,self.done,self.info
    
    def random_action(self):
        return np.random.uniform(-1, 1, size=(2,)) 
    
    def percentageReturn(self):
        return round((self.curr_net_val/self.initial_net_val-1)*100,2)

    def render(self):
        # Create a 1x2 subplot grid
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot profit
        axs[0].plot(self.profit, label='Profit', color='green')
        axs[0].set_title('Profit Plot')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Profit')
        axs[0].legend()

        # Plot loss
        axs[1].plot(self.risk, label='Risk', color='red')
        axs[1].set_title('Risk Plot')
        axs[1].set_xlabel('Days')
        axs[1].set_ylabel('Risk')
        axs[1].legend()

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()

