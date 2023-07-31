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

class SmallConv_Base(nn.Module):
    def __init__(
            self, 
            input_shape,
            output_shape,
            conv_size1=64,
            conv_size2=128,
    ):
        super(SmallConv_Base, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_layers = self.input_shape[0]
        self.out_size = np.prod(output_shape)

        self.conv1 = simple_conv_block(self.n_layers, conv_size1)
        self.res1 = conv_block(conv_size1, n=2)

        self.conv2 = nn.Sequential(
            simple_conv_block(conv_size1, conv_size2),
            nn.MaxPool2d(2)
            )
        
        self.res2 = conv_block(conv_size2, n=2)
        self.global_pool = nn.Sequential(
            simple_conv_block(conv_size2, output_shape),
            nn.AdaptiveMaxPool2d((1,1))
            )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.global_pool(x)
        return x.view(x.shape[0], -1)
 
class Convolutional_Base(nn.Module):
    def __init__(
            self, input_shape, output_shape,
            conv_size1=64, conv_size2=128, conv_size3=256
            ):
        super(Convolutional_Base, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3
        self.n_layers = self.input_shape[0]
        self.out_size = np.prod(output_shape)

        self.conv1 = simple_conv_block(self.n_layers, self.conv_size1)
        self.res1 = conv_block(self.conv_size1, n=2)

        self.conv2 = nn.Sequential(
            simple_conv_block(self.conv_size1, self.conv_size2),
            nn.MaxPool2d(2)
            )
        
        self.res2 = conv_block(self.conv_size2, n=2)

        self.conv3 = nn.Sequential(
            simple_conv_block(self.conv_size2, self.conv_size3),
            nn.MaxPool2d(2)
            )
        self.res3 = conv_block(self.conv_size3, n=2)
        self.global_pool = nn.Sequential(
            simple_conv_block(self.conv_size3, self.conv_size3),
            nn.AdaptiveMaxPool2d((1,1))
            )
        
    def forward(self, x, shape_print=False):
        x1 = self.conv1(x)
        x1 = self.res1(x1) + x1
        x2= self.conv2(x1)
        x2 = self.res2(x2) + x2
        x3 = self.conv3(x2)
        x3 = self.res3(x3) + x3
        out = self.global_pool(x3)
        if shape_print:
            print("x shape: ", x.shape)
            print("x1 shape: ", x1.shape)
            print("x2 shape: ", x2.shape)
            print("x3 shape: ", x3.shape)
            print("out shape: ", out.shape)
        return out.view(out.shape[0], -1)
    
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
        self.fc1 = linear_block(self.binary_output_size + input_shape, 512) # Modify here
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



## params
# conv_base
conv_size1 = 64
conv_size2 = 128
conv_size3 = 256

# small conv base
small_conv_size1 = 64
small_conv_size2 = 128



class ModelStructure(nn.Module):
    def __init__(
            self, input_shape, output_shape, binary_output_shape,
            ):
        super(ModelStructure, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_base = Convolutional_Base(input_shape, output_shape, conv_size1, conv_size2, conv_size3)
        self.conv_base_extra = SmallConv_Base(input_shape, 256, small_conv_size1, small_conv_size2)
        self.pred_head = BinaryPredictionHead(256, binary_output_shape)
        self.magntiude_head = DipoleMagnitudePredictionHead(512, output_shape, binary_output_shape)

    def forward(self, x, binary_labels=None):
        conv_output = self.conv_base(x)
        small_conv_output = self.conv_base_extra(x)
        binary_prediction = self.pred_head(conv_output)
        
        if binary_labels is None:
            # Only during inference
            binary_labels = torch.sigmoid(binary_prediction)
        
        combined_output = torch.cat((conv_output, small_conv_output), dim=1)
        magnitude_prediction = self.magntiude_head(combined_output, binary_labels)
        return binary_prediction, magnitude_prediction