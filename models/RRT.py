import numpy as np
from tqdm import tqdm
import torch

def simple_correlation(a,b):
    return torch.nn.CosineSimilarity(dim=0)(a,b)

class Random_trees:
    def __init__(self):
        pass
    
    def fit(self, stocks_dict, cur_money_stocks, cur_history, cur_index, N_points = 100, tree_depth = 3):
        
        L = len(stocks_dict) #number of stocks
        possible_points = L*(cur_index+1-10)
        N_points = min(N_points, possible_points)
        print(f'N processed points: {N_points}')
        stocks_range = np.arange(L)
        time_range = np.arange(10,cur_index+1)
        pairs = np.dstack(np.meshgrid(stocks_range, time_range)).reshape(-1,2) 
        pairs[np.random.choice(np.arange(pairs.shape[0]), size=N_points, replace=False)]
        
        init_money, init_stocks = cur_money_stocks
        Points_statistics = {}  #{(a,t): (probs, correlation)}
        for (a,t) in tqdm(pairs):
            stock_env = stocks_dict[list(stocks_dict.keys())[a]] #history of action number a

            #reinitialize starting point
            stock_env.init_money = init_money
            stock_env.money = [init_money]
            stock_env.stocks = [init_stocks]
            
            point = stock_env.observation_tensor(t)

            #play a (gay)me for tree_depth days
            probs = self.little_tree_BFS(stock_env, point, t, stock_env.money, stock_env.stocks, tree_depth)
            
            cur_history = torch.nan_to_num(cur_history, nan = 0.0)
            point[:-2] = torch.nan_to_num(point[:-2], nan = 0.0)
            
            Points_statistics[(a,t)] = (tuple(probs), tuple(point[:-2]))
            self.Points_statistics = Points_statistics
        return Points_statistics
    
    def little_tree_BFS(self, env, root, global_start_ind, money_init, stock_init, tree_depth = 3):
        Q = []
        Parent_table = {}
        Visited_probs = {} #probabilities
        Goals = []
        reward_init = 0
        lvl_init = 0
        
        Q.append((money_init, stock_init, lvl_init, reward_init))
        Visited_probs[(tuple(money_init), tuple(stock_init), lvl_init, reward_init)] = (0, 0) #(prob, num_children)
        while Q:
            prev_money, prev_stock,  prev_state_lvl, prev_reward = Q[0] #root
            
            Q = Q[1:] #drop considered node from queue
            
            
            actions = env.get_possible_actions_custom(global_start_ind+prev_state_lvl, prev_state_lvl)
            
            for i,a in enumerate(actions):
                #do step: a.k.a reset money and stocks according to parent
                env.reset()
                env.money = prev_money.copy()
                env.stocks = prev_stock.copy()
                
                env.transition(prev_state_lvl, a) #resets env.money and env.stock
                next_reward = prev_reward+env.reward(global_start_ind+prev_state_lvl, a)
                
                next_state_lvl = prev_state_lvl+1 
                
                assert next_state_lvl+1 == len(env.money) == len(env.stocks)
                
                prev_prob = next_prob = 0 #until we got the end of the path we can't say the probability
                
                Parent_table[(tuple(env.money), tuple(env.stocks), next_state_lvl, next_reward)]=(prev_money, prev_stock, prev_state_lvl, prev_reward)
                Visited_probs[(tuple(env.money), tuple(env.stocks), next_state_lvl, next_reward)] = (0,0)
    
                if next_state_lvl< tree_depth:
                    Q.append((env.money.copy(), env.stocks.copy(), next_state_lvl, next_reward.copy()))
                else:
                    Goals.append((env.money.copy(), env.stocks.copy(), next_state_lvl, next_reward.copy()))
                    Visited_probs[(tuple(env.money), tuple(env.stocks), next_state_lvl, next_reward)]=(next_reward,1)
        
        #update and backprop probabilities:
        New_goals = Goals
        while New_goals:
            g = New_goals[0]
            g_hash =tuple(g[0]),tuple(g[1]),g[2], g[3]
            
            if g==(money_init.copy(), stock_init.copy(), lvl_init, reward_init):
                break
            par = Parent_table[g_hash]
            par_hash =tuple(par[0]),tuple(par[1]),par[2], par[3]
            #getting out of queue
            if Visited_probs[g_hash][1] ==0:
                 denom =1
            else: 
                denom =Visited_probs[g_hash][1]
            g_reward_avg = Visited_probs[g_hash][0]/denom
            
            #update parent prob
            P,N = Visited_probs[par_hash]
            Visited_probs[par_hash] = (P+g_reward_avg, N+1)
            New_goals = New_goals[1:] 
            if par not in New_goals:
                New_goals.append(par)
        #find probabilities for children of root
        probs = []
        for a in (1, 0, -1):
            #root, money_init, stock_init, lvl_init, reward_init
            env.money = money_init
            env.stocks = stock_init
            env.transition(global_start_ind, a) #resets env.money and env.stock
            next_reward = prev_reward+env.reward(global_start_ind+prev_state_lvl, a)
            next_lvl = 1
            
            if (tuple(env.money), tuple(env.stocks), next_lvl, next_reward) in Visited_probs:
                probs.append(Visited[(env.money, env.stocks, next_lvl, next_reward)][0]/Visited[(env.money, env.stocks, next_lvl, next_reward)][1])
            else:
                probs.append(0)
        probs = np.array(probs)
        sum_ = np.sum(probs)
        if sum_ == 0:
            sum_ = 1
        probs = np.exp(probs)/sum(np.exp(probs))
        assert np.sum(np.abs(probs))<=1 and all([i>=0 for i in probs])
        return probs

    def predict(self, data, ind, stocks_dict, corr_fn = simple_correlation):
        cur_money_stocks = data[-2:]
        cur_history = data[:-2]
        cur_index = ind
        
        statistics = self.Points_statistics 
        cum_probs = np.array([0,0,0])
        for i in range(len(statistics)):
            probs, hist = statistics[list(statistics.keys())[i]] 
            cur_history = torch.nan_to_num(cur_history, nan = 0.0)
            hist = torch.tensor(hist)
            hist = torch.nan_to_num(hist, nan = 0.0)
            coeff = corr_fn(hist, cur_history)
            
            cum_probs=cum_probs + np.array(coeff*np.array(probs))
        
        return cum_probs/np.sum(cum_probs)