from enum import Enum


class Initialization(Enum):
    RANDOM = 0


class CNNLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization

    def __repr__(self):
        return 'C-'


class PoolingLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization

    def __repr__(self):
        return 'P-'


class FCLayer(object):
    def __init__(self, initialization):
        self.initialization = initialization

    def __repr__(self):
        return 'FC-'


class CNNEntity(object):
    def __init__(self, layers):
        self.layers = layers

    def score_me(self, train_set, valid_set):
        pass

    def test_me(self, test_set):
        pass

    def __repr__(self):
        representation = ""
        for layer in self.layers:
            representation = representation + str(layer)

        return representation
