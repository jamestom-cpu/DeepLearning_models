import numpy as np
from .abstract import Generator
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


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
    
    def generate_and_save_data_samples(self, N, file_prefix, M=None, data_dir=".", start_index=0):
        if M is None:
            M = N
        temp_inputs = []
        temp_target = []
        all_datasets = []
        files_to_remove = []

        for i in tqdm(range(start_index, N), desc="Generating data", unit="samples"):
            f, t = next(self)
            temp_inputs.append(f)
            temp_target.append(t)
            
            if (i+1) % M == 0 or i+1 == N:
                # transorm the lists to numpy arrays for speed in converting to torch tensors
                temp_inputs = np.asarray(temp_inputs)
                temp_target = np.asarray(temp_target)

                # convert the accumulated samples to TensorDataset
                dataset = TensorDataset(torch.from_numpy(temp_inputs).float(), torch.from_numpy(temp_target).float())
                # store the TensorDataset in the list
                all_datasets.append(dataset)
                # save the TensorDataset
                fullpath = os.path.join(data_dir, f"{file_prefix}_{i // M + 1}.pt")
                torch.save(dataset, fullpath)
                # add the file name to the list of files to be removed
                files_to_remove.append(fullpath)
                # reset the lists
                temp_inputs = []
                temp_target = []

        # combine all datasets and save into a single file
        combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
        fullpath = os.path.join(data_dir, f"{file_prefix}.pt")
        torch.save(combined_dataset, fullpath)

        # remove all the individual saved files
        for file in files_to_remove:
            os.remove(file)

        return combined_dataset