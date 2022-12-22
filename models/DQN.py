import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
from tqdm import tqdm
import numpy as np
# from ..utils.utils import get_diff_action_ohe

from .base_model import Base_model

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_diff_action_ohe(action):
    maxes = torch.tensor(torch.max(action, dim=1).values).reshape(-1,1)
    maxes = torch.cat([maxes, maxes, maxes], dim=1)
    
    bin_action = torch.tensor(torch.eq(action, maxes).int(), requires_grad=True)
    
    bin_action = bin_action @ torch.tensor([-1, 0, 1], dtype=torch.int).reshape(-1, 1)
    
    print(bin_action.grad)
    return bin_action


class ReplayMemory:
    def __init__(self, env):
        self.memory = []
        env.reset()
        for i in range(10, env.day_count-1):
            state = env.observation_tensor(i)
            for action in env.action_space:
                reward = env.reward(i, action)
                env.transition(i, action)
                next_state = env.observation_tensor(i+1)
                self.memory.append(Transition(state, action, next_state, reward))
                env.reset()
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
        

class Network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )
            
    def forward(self, x):
        return self.layers(x)


class DQN(Base_model):
            
    def __init__(self, n_observations, n_actions, env,
                 gamma=.9,
                 lr=1e-4,
                 eps_start=.9,
                 eps_end=.05,
                 eps_decay=1000,
                 tau = 5*1e-3,
                 batch_size=16,
                 num_episodes=10000):
        self.policy_net = Network(n_observations, n_actions)
        self.target_net = Network(n_observations, n_actions)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_done = 0
        self.lr = lr
        self.es = eps_start
        self.ee = eps_end
        self.ed = eps_decay
        self.tau = tau
        self.num_episodes = num_episodes
        
        self.memory = ReplayMemory(env)
        
        random.seed(42)
    
    def __select_action(state):
        sample = np.random.random()
        eps_threshold = self.ee + (self.es - self.ee) * \
            math.exp(-1. * self.steps_done / self.ed)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    
    
    def optimise_model(self):
        n_actions = 3
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions)) # batch of transitions to transition of batches (matrix.transpose())
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
#         print([type(x) for x in batch.state[0]])
        state_batch = torch.stack(batch.state).float()
        next_state_batch = torch.stack(batch.next_state).float()
        action_batch = torch.tensor(np.asarray(batch.action))
        reward_batch = torch.tensor(np.asarray(batch.reward))
        
        
        #print(self.policy_net(state_batch))
        state_action_values = self.policy_net(state_batch)
       # print(state_action_values)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
    
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
        optimizer.zero_grad()
        loss.backward()
    # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer.step()
    
        
    def fit(self, policy_state_dict=None):
        
        if policy_state_dict:
            self.policy_net.load_state_dict(policy_state_dict)
        
        for i in tqdm(range(self.num_episodes)):
            self.optimise_model()
                
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
            
            
            
    def predict(self, state) -> int:
        return np.argmax(self.policy_net(state).detach().numpy())-1
            
            
    def state_dict(self):
        return self.policy_net.state_dict()
          
            
            
            
            
            
        