import numpy as np
from .abstract import Generator
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

class DataIterator:
    def __init__(self, generating_class: Generator):
        """
        Generally best not to normalize the data directly in the iterator, but rather
        normalize the data before passing it to network. This is because you may want to use
        all the data properties in the future.
        
        """
        self.gen = generating_class

    def __iter__(self):
        return self

    def __next__(self):
        labeled_data = self.gen.generate_labeled_data()      
        return labeled_data

    def generate_N_data_samples(self, N):
        samples = []
        for _ in tqdm(range(N), desc="Generating data", unit="samples"):
            samples.append(next(self))
        
        f, t = zip(*samples)
        return np.asarray(f), np.asarray(t)