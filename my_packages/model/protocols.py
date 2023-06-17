from abc import ABC, abstractmethod

class EA_base(ABC):

    @abstractmethod
    def evaluate_fitness(self, *args, **kargs):
        pass

    @abstractmethod
    def select_pair(self, *args, **kargs):
        pass

    @abstractmethod
    def crossover(self, *args, **kargs):
        pass

    @abstractmethod
    def mutation(self, *agrs, **kargs):
        pass

    @abstractmethod
    def run_evolution(self, *args, **kargs):
        pass