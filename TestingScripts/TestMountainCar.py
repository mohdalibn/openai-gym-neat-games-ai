
import os
import pickle
import neat
import gym
import numpy as np

# Load the winner
with open('Winners/winner-MountainCar', 'rb') as f:
    c = pickle.load(f)

print("Loaded genome:")
print(c)


local_dir = os.getcwd()  # This get the Current Working Directory
config_path = os.path.join(local_dir, 'ConfigFiles/config-MountainCar.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('MountainCar-v0')
observation = env.reset()

done = False

while not done:
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = env.step(action)

    env.render()
