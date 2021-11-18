import random
from EvoCNN import EvoCNN


def main():
    random.seed(0)
    evo = EvoCNN(population_size=20, max_init_convs=5, max_init_fc=3)
    evo.build_initial_population()
    evo.do_evolution()
    best = evo.get_best()


if __name__ == "__main__":
    main()

