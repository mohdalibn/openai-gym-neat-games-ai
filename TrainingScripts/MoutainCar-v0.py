
import multiprocessing
import os
import pickle
import neat
import gym
import numpy as np


runs_per_net = 2
# simulation_seconds = 60.0 # We don't need this cuz this simulation will kill on its own


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make('MountainCar-v0')
        observation = env.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        #frames = 0

        while not done:

            action = np.argmax(net.activate(observation))
            observation, reward, done, info = env.step(action)

            fitness += reward

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.max(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, 'ConfigFiles/config-MountainCar.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('Winners/winner-MountainCar', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__':
    run()
