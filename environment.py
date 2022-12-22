import pandas as pd
import numpy as np
from datetime import datetime
import torch

action_space = [
    -1, #sell
    0,  # ignore
    1 # buy
]

class Environment:
    def __init__(self, data:pd.DataFrame, stock_name:str, initial_money:float):
        self.data = data[['Date', 'Weekday']+[col for col in data.columns if stock_name in col]]
        # immutable
        self.init_money = initial_money
        self.stock_name = stock_name
        self.action_space = [
        -1, #sell
        0,  # hold
        1] # buy
        self.day_count = self.data.shape[0]
        
        #mutable
        self.money = [initial_money]
        self.stocks = [0]
        
        
        
        
    def reset(self):
        self.money = [self.init_money]
        self.stocks = [0]
    
    def first_date(self):
        return self.data['Date'].min() 
    
    
    def observation_tensor(self, index):
        '''
        provides an observation of 10 days before the date provided
        return: tensor( prices as vector, money:float, stocks_num:int)
        '''
#         print(index)
        assert 10 <= index <= self.data.shape[0], f'index out of range, {index}'
        prices = torch.from_numpy(self.data[self.data.columns[1:]].iloc[index-10:index].values).float().flatten()
#         print(prices.shape[0])
        if index-10 < len(self.money):
            return torch.hstack([prices, torch.tensor(self.money[index-10]), torch.tensor(self.stocks[index-10])])
        else:
            return torch.hstack([prices, torch.tensor(self.money[-1]), torch.tensor(self.stocks[-1])])
    
    
    def observation(self, index):
        '''
        provides an observation of 10 days before the date provided
        return: dict, keys: prices:pd.DataFrame, money:float, stocks_num:int
        '''
        
        assert 10 <= index <= self.data.shape[0], f'index out of range, {index}'
        prices = torch.from_numpy(self.data.iloc[index-10:index].values).float().flatten()
        if index-10 < len(self.money):
            return torch.cat([prices, self.money[index-10], self.stocks[index-10]])
        else:
            return torch.cat([prices, self.money[index-1], self.stocks[-1]])
    
    
    def reward(self, index, action:int) -> float:
        curr_price = self.data.iloc[index][self.stock_name + '_close']
        next_price = self.data.iloc[index+1][self.stock_name + '_close']
        
        if (action == -1 and self.stocks[-1] == 0) or (action == 1 and self.money[-1] < curr_price):
            return -1000
        
        
        delta = next_price - curr_price
        return action*delta
    
    
    def get_possible_actions(self, index:int) -> list:
        possible_actions = []
        if index is None:
            index = self.data.index[self.data['Date'] == date]
        if self.data.iloc[index][self.stock_name + '_close'] <= self.money[index]:
            possible_actions.append(-1)
        possible_actions.append(0) # we can skip in any case
        if self.stocks[index] > 0:
            possible_actions.append(1)
        return possible_actions
        
    
    def transition(self, index:int, action) -> None:
        self.stocks.append(self.stocks[-1]+action)
        self.money.append(self.money[-1] - action*self.data.iloc[index][self.stock_name + '_close'])
        
        
        
        
        