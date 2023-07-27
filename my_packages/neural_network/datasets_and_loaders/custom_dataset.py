import os
import json

import torch
from torch.utils.data import Dataset


class LowMemDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.file_names = os.listdir(directory)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_names[idx])
        return torch.load(file_path)