import numpy as np
# %% Define Model Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from my_packages.neural_network.model.model_base import Model_Base
from NN_model_architectures.NN_blocks import simple_conv_block, conv_block, linear_block

from torchsummary import summary
from torchviz import make_dot

 
class Convolutional_Base(nn.Module):
    def __init__(
            self, input_shape, output_shape):
        super(Convolutional_Base, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_layers = self.input_shape[0]
        self.out_size = np.prod(output_shape)

        self.conv1 = simple_conv_block(self.n_layers, 64)
        self.res1 = conv_block(64, n=2)

        self.conv2 = nn.Sequential(
            simple_conv_block(64, 128),
            nn.MaxPool2d(2)
            )
        
        self.res2 = conv_block(128, n=2)

        self.conv3 = nn.Sequential(
            simple_conv_block(128, 256),
            nn.MaxPool2d(2)
            )
        self.res3 = conv_block(256, n=2)
        self.global_pool = nn.Sequential(
            simple_conv_block(256, 256),
            nn.AdaptiveMaxPool2d((1,1))
            )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.conv3(x)
        x = self.res3(x) + x
        x = self.global_pool(x)
        return x.view(x.shape[0], -1)
    
class BinaryPredictionHead(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BinaryPredictionHead, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.output_size = np.prod(output_shape)

        self.dropout3 = nn.Dropout(0.05)
        self.fc1 = linear_block(256, 512)
        self.dropout4 = nn.Dropout(0.05)
        self.fc2 = linear_block(512, self.output_size, activation=None)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x.view(x.shape[0], *self.output_shape)
    
class DipoleMagnitudePredictionHead(nn.Module):
    def __init__(self, input_shape, output_shape, binary_output_shape):
        super(DipoleMagnitudePredictionHead, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.output_size = np.prod(output_shape)
        self.binary_output_size = np.prod(binary_output_shape)

        self.dropout3 = nn.Dropout(0.05)
        self.fc1 = linear_block(self.binary_output_size + 256, 512) # Modify here
        self.dropout4 = nn.Dropout(0.05)
        self.fc2 = linear_block(512, 512)
        self.dropout5 = nn.Dropout(0.05)
        self.fc3 = linear_block(512, self.output_size, activation=nn.ReLU())

    def forward(self, x, binary_label): # Accept binary_label as input
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        if len(binary_label.shape) > 2:
            binary_label = binary_label.view(binary_label.shape[0], -1)
        x = torch.cat((x, binary_label), dim=1) # Concatenate along the feature dimension
        x = self.dropout3(x)
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.dropout5(x)
        x = self.fc3(x)
        return x.view(x.shape[0], *self.output_shape)



