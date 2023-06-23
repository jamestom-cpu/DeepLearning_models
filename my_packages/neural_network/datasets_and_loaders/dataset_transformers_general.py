from torch.utils.data import Dataset

class Single_Probe_Height_View_Dataset(Dataset):
    def __init__(self, parent_dataset, probe_height_index):
        self.parent_dataset = parent_dataset
        self.probe_height_index = probe_height_index

    def __getitem__(self, index):
        data, label = self.parent_dataset[index]
        return data[:, self.probe_height_index], label
    
    def __len__(self):
        return len(self.parent_dataset)