from typing import List, Optional, Iterator
from tqdm.notebook import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def td_transition(state, action, indexes, env):    
    action = (torch.argmax(action) - 1).numpy()
    new_state, reward = env.transition_batch(action, indexes)
    return new_state, reward


class ActorModel(nn.Module):
    """Actor nn model"""

    def __init__(self, 
                 actor_layer_size: int,
                 actor_in_size: int = 10, 
                 actor_out_size: int = 3, 
                ) -> None:
        """
        Constructor
        :param actor_layer_size: hidden layer's size
        :param constraints_linear: linear constraints
        :param constraints: angle constraints
        """
        super().__init__()
        self.actor = nn.Sequential(
            torch.nn.Linear(actor_in_size, actor_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(actor_layer_size, actor_out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param x: (batch_size, 6) state's tensor
        """
        x = self.actor(x)
        return x


class CriticModel(nn.Module):
    """Critic nn model"""

    def __init__(self, critic_layer_size: int,
                       critic_in_size: int, 
                       critic_out_size: int = 1, 
                       scale_factor: int = -5000
                ) -> None:
        """
        Constructor
        :param critic_layer_size: hidden critic's size
        :param scale_factor: scaling for model's output
        """
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(critic_in_size, critic_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_layer_size, critic_out_size)
        )
        self.act = torch.nn.Tanh()
#         self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param x: (batch_size, 6) state's tensor
        """
        x = self.critic(x)
        x = -(x ** 2)
        x = self.act(x)
        return x #* self.scale_factor
    
    
class CriticTD(nn.Module):
    """
    Critic Temporal Difference model
    """

    def __init__(self,
                 actor,
                 critic, #: CriticModel, 
                 env,      
                 transition=td_transition, #: CameraTransition,
                 satellite_discount: float = .98) -> None:
        """
        Init method
        :param actor: Actor nn model
        :param critic: Critic nn model
        :param transition: Camera Transition model
        :param satellite_discount: satellite discount for TD method
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.transition = transition
        self.env = env
        self.satellite_discount = satellite_discount
        self.loss = nn.MSELoss()

    def forward(self, state: torch.Tensor, indexes) -> torch.Tensor:
        """
        Main forward method
        :param state: (batch_size, 6) camera's state
        :return: TD loss
        """
        with torch.no_grad():
            action = self.actor(state)
            next_state, reward = self.transition(state, action, indexes, self.env)
            td_target = reward + self.satellite_discount * self.critic(next_state)
        value = self.critic(state)
        return self.loss(value, td_target)

    def parameters(self) -> Iterator[Parameter]:
        return self.critic.parameters()


class ActorImprovedValue(nn.Module):

    def __init__(self,
                 actor, #: ActorModel,
                 critic, #: CriticModel, 
                 env,      
                 transition=td_transition, #: CameraTransition,
                 satellite_discount: float = .98) -> None:
        """
        Init method
        :param actor: Actor nn model
        :param critic: Critic nn model
        :param transition: Camera Transition model
        :param satellite_discount: satellite discount for TD method
        """
        super().__init__()
        self.critic = critic
        self.actor = actor
        self.transition = transition
        self.env = env
        self.satellite_discount = satellite_discount

    def forward(self, state, index_):
        """
        Main forward method
        :param state: (batch_size, 6) camera's state
        :return: actor's improved value
        """
        action = self.actor(state)
        next_state, reward = self.transition(state, action, index_, self.env)
        improved_value = reward + self.satellite_discount * self.critic(next_state)
        return -improved_value.mean()

    def parameters(self) -> Iterator[Parameter]:
        return self.actor.parameters()
    

def get_random_state(batch_size: int,
                     env,
                     index_min: int = 10,
                     index_max: int = 11, 
                     ):
    
    rand_indexes = np.random.randint(index_min, index_max, batch_size)
    
    X = []
    
    for index in rand_indexes:
        state_dict = env.observation(index)
        state_array = np.hstack([state_dict['prices'],
                                      np.array(state_dict['money']),
                                      np.array(state_dict['stocks_num'])
                                     ])
        X.append(state_array)
        
    return torch.Tensor(X), torch.tensor(rand_indexes)


def critic_epoch(optimizer: torch.optim.Optimizer,
                 model: CriticTD, 
                 iterations: int,
                 env,
                 batch_size: int) -> List[float]:
    losses = []
    for iteration in tqdm(range(iterations), "Critic epoch"):
        X, indexes = get_random_state(batch_size, env, index_max=env.data.shape[0]-1)
        X, indexes = X.to(device), indexes.to(device)

        optimizer.zero_grad()
        
        loss = model(X, indexes)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    print(f"Critic mean loss: {np.mean(losses)}")
    return losses

def actor_epoch(optimizer: torch.optim.Optimizer,
                 model: CriticTD, 
                 iterations: int, 
                 env,
                 batch_size: int) -> List[float]:
    values = []
    for iteration in tqdm(range(iterations), "Actor epoch"):
        X, indexes = get_random_state(batch_size, env, index_max=env.data.shape[0]-1)
        X, indexes = X.to(device), indexes.to(device)

        optimizer.zero_grad()
        improved_value = model(X, indexes)
        improved_value.backward()
        optimizer.step()
        values.append(improved_value.detach().cpu().numpy())
    print(f"Actor mean value: {np.mean(values)}")
    
    return values


    
    
    
    