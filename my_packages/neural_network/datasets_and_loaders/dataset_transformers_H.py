import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .dataset_transformers_general import Single_Probe_Height_View_Dataset


class H_Components_Dataset(Dataset):
    def __init__(self, dataset: Dataset, probe_height_index=None):
        self.dataset = dataset
        self.transformation_list = [
            transforms.Lambda(lambda x: x[-2:]),
        ]
        self.label_transformation_list = []

        if probe_height_index is not None:
            self.dataset = Single_Probe_Height_View_Dataset(self.dataset, probe_height_index)


    @staticmethod
    def trim_outer_rings_from_mask(input_tensor, k=1):
        # Check if the input tensor is at least 2-dimensional
        if len(input_tensor.shape) < 2:
            raise ValueError("Input tensor should be at least 2-dimensional")
        
        # Check if k is less than half the size of the smallest dimension
        if k > min(input_tensor.shape[-2:]) // 2:
            raise ValueError("k should be less than half the size of the smallest dimension")

        # Trimming the outer "rings" from the last two dimensions
        trimmed_tensor = input_tensor[..., k:-k, k:-k]
        
        return trimmed_tensor


    def unpad_label(self, k=1):
        self.label_transformation_list.append(
            transforms.Lambda(
            lambda x: self.trim_outer_rings_from_mask(x, k)
            ))
        return self

    
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
        if len(self.label_transformation_list) != 0:
            transform_label = transforms.Compose(self.label_transformation_list)
        else:
            transform_label = lambda x: x

        if isinstance(index, slice):
            data_tensor = []
            label_tensor = []
            for i in range(*index.indices(len(self))):
                data, label = self.dataset[i]
                data_tensor.append(transform(data))
                label_tensor.append(transform_label(label[-2:]))
            return torch.stack(data_tensor), torch.stack(label_tensor)
        else:
            data, label = self.dataset[index]
            return transform(data), transform_label(label[-2:])
    
