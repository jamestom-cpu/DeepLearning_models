import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class H_Components_Dataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.transformation_list = [
            transforms.Lambda(lambda x: x[-2:]),
        ]
    
    def scale_to_01(self):
        self.transformation_list.append(
            transforms.Lambda(
            lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
            ))
        return self
    
    def reset_transformations(self):
        self.transformation_list = [
            transforms.Lambda(lambda x: x[-2:]),
        ]
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        transform = transforms.Compose(self.transformation_list)
        if isinstance(index, slice):
            data_tensor = []
            label_tensor = []
            for i in range(*index.indices(len(self))):
                data, label = self.dataset[i]
                data_tensor.append(transform(data))
                label_tensor.append(label[-2:])
            return torch.stack(data_tensor), torch.stack(label_tensor)
        else:
            data, label = self.dataset[index]
            return transform(data), label[-2:]
    
