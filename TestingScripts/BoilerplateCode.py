import gym
#import neat
import numpy as np


env = gym.make('LunarLander-v2')
observation = env.reset() # The observation is just going to be a vector/matrix of values

print(observation)
print(env.action_space)

done = False

while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample())
    env.render()

    
