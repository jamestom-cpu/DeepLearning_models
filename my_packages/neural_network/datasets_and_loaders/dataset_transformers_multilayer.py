import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .dataset_transformers_general import Single_Probe_Height_View_Dataset


class H_Components_Dataset_Multilayer(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.transformation_list = [
            transforms.Lambda(lambda x: x[-2:]),
        ]
        self.label_transformation_list = [
            transforms.Lambda(lambda x: x[:, -2:]),
        ]

        print("finished initializing")
    
    @staticmethod
    def rescaling_func_data(data_input):
        maximums = data_input.numpy().max(axis=(0, 2, 3))
        # maximums, _ = torch.max(data_input.view(input_data_shape[0], -1), dim=-1)
        normalized_data = data_input / torch.from_numpy(maximums).view(1, -1, 1, 1)
        return normalized_data
    
    @staticmethod
    def rescaling_label_data(label_input):
        binary, magnitude = label_input[0], label_input[1]

        normalized_magnitude = magnitude.div(torch.max(magnitude))        
        normalized_layers = torch.stack([binary, normalized_magnitude], dim=0)
        return torch.concatenate([normalized_layers, label_input[2:]], dim=0)
    
    @staticmethod
    def phase_to_sin_and_cos(label_input):
        binary, _, phase = label_input
        mask = binary.ne(0)

        sin_phase = phase.clone()
        cos_phase = phase.clone()

        sin_phase[mask] = torch.sin(phase[mask])
        cos_phase[mask] = torch.cos(phase[mask])
        return torch.stack([binary, label_input[1], sin_phase, cos_phase], dim=0)

    def rescale_probe_heights(self):
        self.transformation_list.append(
            transforms.Lambda(
            lambda x: self.rescaling_func_data(x)
            ))
        return self
    
    def rescale_labels(self):
        self.label_transformation_list.append(
            transforms.Lambda(
            lambda x: self.rescaling_label_data(x)
            ))
        return self
    
    def shift_phases(self):
        self.label_transformation_list.append(
            transforms.Lambda(
            lambda x: self.shift_phases_with_respect_to_dipole0(x)
            ))
        return self
    
    def expand_phase(self):
        self.label_transformation_list.append(
            transforms.Lambda(
            lambda x: self.phase_to_sin_and_cos(x)
            ))
        return self

    def add_transformation(self, func:callable):
        self.transformation_list.append(
            transforms.Lambda(func)
        )
        return self

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
                label_tensor.append(transform_label(label))
            return torch.stack(data_tensor), torch.stack(label_tensor)
        else:
            data, label = self.dataset[index]
            return transform(data), transform_label(label)