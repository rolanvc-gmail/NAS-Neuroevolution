import random
from enum import Enum


class Initialization(Enum):
    RANDOM = 0


class CNNLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization


class PoolingLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization


class FCLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization


class CNNEntity(object):
    pass


class EvoCNN(object):
    def __init__(self, population_size, max_init_convs, max_init_fc):
        self.population = []
        self.population_size = population_size
        while len(self.population < self.population_size):
            cnn_entity = self.generate_random_individual(max_init_convs, max_init_fc)
            self.population.append(cnn_entity)

    @staticmethod
    def generate_random_individual(max_init_convs, max_init_fc):
        n_convs = random.randint(0, max_init_convs)
        layers = []
        n_fcs = random.randint(0, max_init_fc)
        for i in range(n_convs):
            flip = random.random()
            if flip <= 0.5:
                cnn_layer = CNNLayer(Initialization.RANDOM)
                layers.append(cnn_layer)
            else:
                pool_layer = PoolingLayer(Initialization.RANDOM)
                layers.append(pool_layer)
        for i in range(n_fcs):
            fc = FCLayer(Initialization.RANDOM)
            layers.append(fc)

        cnn_entity = CNNEntity(layers)

        return cnn_entity

    def build_initial_population(self):
        pass

    def do_evolution(self):
        pass

    def get_best(self):
        pass