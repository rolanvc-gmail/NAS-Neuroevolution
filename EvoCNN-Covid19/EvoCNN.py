import random
from CNNEntity import CNNEntity, Initialization
from CNNEntity import CNNLayer, PoolingLayer, FCLayer


class EvoCNN(object):
    def __init__(self, population_size, max_init_convs, max_init_fc):
        self.population = []
        self.population_size = population_size
        self.max_init_convs = max_init_convs
        self.max_init_fc = max_init_fc
        self.generations = 0

    @staticmethod
    def generate_random_individual(max_init_convs, max_init_fc):
        n_convs = random.randint(0, max_init_convs)
        layers = []
        n_fcs = random.randint(0, max_init_fc)
        cnn_layer = CNNLayer(Initialization.RANDOM)
        layers.append(cnn_layer)
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
        while len(self.population) <= self.population_size:
            cnn_entity = self.generate_random_individual(self.max_init_convs, self.max_init_fc)
            self.population.append(cnn_entity)

        print(self.population)

    def not_finished(self):
        """
        :return: True if evolution should continue; else False
        """
        self.generations += 1
        if self.generations < 10:
            return True
        else:
            return False

    def evaluate_individual_fitness(self):
        # Todo: finish this
        for s in self.population:
            pass

    def select_parents(self):
        # Todo: finish this
        pass

    def generate_offspring(self):
        pass
        # Todo: finish this

    def do_evolution(self):
        while self.not_finished():
            self.evaluate_individual_fitness()
            P = self.select_parents()
            Q = self.generate_offspring()

    def get_best(self):
        pass