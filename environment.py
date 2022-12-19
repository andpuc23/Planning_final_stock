import pandas as pd
import numpy as np

action_space = [
    -1, #sell
    0,  # ignore
    1 # buy
]

class Environment:
    def __init__(self, data:pd.DataFrame, stock_name:str, initial_money:float):
        self.data = data[['Date']+[col for col in data.columns if stock_name in col]]
        # immutable
        self.init_money = initial_money
        self.stock_name = stock_name
        #mutable
        self.money = [initial_money]
        self.stocks = [0]
    
    def observation(self, date):
        '''
        provides an observation of 10 days before the date provided
        return: dict, keys: prices:pd.DataFrame, money:float, stocks_num:int
        '''
        index = self.data.index[self.data['Date'] == date]
        return {'prices':self.data.iloc[index-10:index], 'money':self.money, 'stocks_num':self.stocks}
    
    
    def reward(self, date:Datetime, action:int) -> float:
        index = self.data.index[self.data['Date'] == date]
        curr_price = self.data.iloc[index][self.stock_name + '_close']
        next_price = self.data.iloc[index+1][self.stock_name + '_close']
        
        delta = next_price - curr_price
        return action*delta
    
    
    def get_possible_actions(self, date:Datetime=None, index:int=None) -> list:
        possible_actions = []
        if index is None:
            index = self.data.index[self.data['Date'] == date]
        if self.data.iloc[index][self.stock_name + '_close'] <= self.money[index]:
            possible_actions.append(-1)
        possible_actions.append(0) # we can skip in any case
        if self.stocks[index] > 0:
            possible_actions.append(1)
        return possible_actions
        
    
    def transition(self, action, date:Datetime=None, index:int=None) -> None:
        if index is None:
            index = self.data.index[self.data['Date'] == date]
        
        if action == 0:
            return
        
        self.stocks.append(self.stocks[-1]+action)
        self.money.append(self.money[-1] - action*self.data.iloc[index][self.stock_name + '_close'])
        
        
        
        
        