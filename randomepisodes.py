import gym
import numpy as np
my_env = gym.make("LunarLander-v2")

for i in range(10):
    my_env.reset()
    while True:
        my_env.render()
        action=my_env.action_space.sample()
        next_state,reward,done,info=my_env.step(action)

        print(next_state,reward,done,info,action)
        if done:
            break
my_env.close()