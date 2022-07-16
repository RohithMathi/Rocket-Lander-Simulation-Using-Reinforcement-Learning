import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

my_env = gym.make("LunarLander-v2")

state_dim=my_env.observation_space.shape[0]
action_dim=my_env.action_space.n
learning_rate=0.01

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 50)
        self.l2 = nn.Linear(50, 100)
        self.l3 = nn.Linear(100, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x


act=Actor(state_dim,action_dim,1)
cri=Critic(state_dim,action_dim)
act.eval()
cri.eval()

for i in range(10):
    my_env.reset()
    next_state, reward, done, info = my_env.step(0)

    while True:
        my_env.render()
        #action=act(next_state)
        action=0
        next_state,reward,done,info=my_env.step(action)
        print(next_state,reward,done,info,action)
        if done:
            break



my_env.close()