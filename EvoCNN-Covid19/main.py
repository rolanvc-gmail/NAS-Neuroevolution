import random
from EvoCNN import EvoCNN


def main():
    random.seed(0)
    evo = EvoCNN()
    evo.build_initial_population()
    evo.do_evolution()
    best = evo.get_best()


if __name__ == "__main__":
    main()