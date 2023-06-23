import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchsummary import summary

from my_packages.neural_network.model.model_base import Model_Base



## Define the Model Structure

def conv_block(in_channels, out_channels, pool=False, batch_norm=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def deconv_block(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                           stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)



class Quasi_ResNet2(Model_Base):
    def __init__(
            self, in_shape, out_shape, conv_size1 = 64, conv_size2 = 128, conv_size3 = 256,
            fc_size = 512,
            loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3

        self.fc_size = fc_size 
        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(Quasi_ResNet4, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = nn.Sequential(conv_block(self.conv_size1, self.conv_size1), 
                                  conv_block(self.conv_size1, self.conv_size1))

        self.conv2 = conv_block(self.conv_size1, self.conv_size2, pool=True,) # output: conv_size2 x 15 x 15
        self.res2 = nn.Sequential(conv_block(self.conv_size2, self.conv_size2), 
                                  conv_block(self.conv_size2, self.conv_size2))

        self.conv3 = conv_block(self.conv_size2, self.conv_size3, pool=True) # output: conv_size3 x 7 x 7
        self.res3 = nn.Sequential(conv_block(self.conv_size3, self.conv_size3), conv_block(self.conv_size3, self.conv_size3))
        
        # global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        # fc_input_size = self.conv_size3 * 7 * 7
        fc_input_size = self.conv_size3

        # upsampling layers
        self.fc1 = nn.Linear(fc_input_size, self.fc_size) # 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, out_size) 

    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out

        out = self.global_max_pool(out)


        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)


class Quasi_ResNet4(Model_Base):
    def __init__(
            self, in_shape, out_shape, conv_size1 = 64, conv_size2 = 128, conv_size3 = 256,
            fc_size = 512,
            loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3

        self.fc_size = fc_size 
        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(Quasi_ResNet4, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = nn.Sequential(conv_block(self.conv_size1, self.conv_size1), 
                                  conv_block(self.conv_size1, self.conv_size1))

        self.conv2 = conv_block(self.conv_size1, self.conv_size2, pool=True, batch_norm=True) # output: conv_size2 x 15 x 15
        self.res2 = nn.Sequential(conv_block(self.conv_size2, self.conv_size2), 
                                  conv_block(self.conv_size2, self.conv_size2))

        self.conv3 = conv_block(self.conv_size2, self.conv_size3, pool=True) # output: conv_size3 x 7 x 7
        # self.res3 = nn.Sequential(conv_block(self.conv_size3, self.conv_size3), conv_block(self.conv_size3, self.conv_size3))

        # # global max pooling
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        fc_input_size = self.conv_size3 * 7 * 7

        # upsampling layers
        self.fc1 = nn.Linear(fc_input_size, self.fc_size) # 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, out_size) 

    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        # out = self.conv3(out)
        # out = self.res3(out) + out


        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)
    

class Quasi_ResNet6_1(Model_Base):
    def __init__(
            self, in_shape, out_shape, conv_size1 = 64, conv_size2 = 128, conv_size3 = 256,
            conv_size4 = 512, fc_size = 512,
            loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3
        self.conv_size4 = conv_size4

        self.fc_size = fc_size 
        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(Quasi_ResNet6_1, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = nn.Sequential(conv_block(self.conv_size1, self.conv_size1), 
                                  conv_block(self.conv_size1, self.conv_size1))

        self.conv2 = conv_block(self.conv_size1, self.conv_size2, pool=True,) # output: conv_size2 x 15 x 15
        self.res2 = nn.Sequential(conv_block(self.conv_size2, self.conv_size2), 
                                  conv_block(self.conv_size2, self.conv_size2))

        self.conv3 = conv_block(self.conv_size2, self.conv_size3, pool=True) # output: conv_size3 x 7 x 7
        self.res3 = nn.Sequential(conv_block(self.conv_size3, self.conv_size3), conv_block(self.conv_size3, self.conv_size3))

        self.conv4 = conv_block(self.conv_size3, self.conv_size4, pool=True) # output: conv_size4 x 3 x 3
        self.res4 = nn.Sequential(conv_block(self.conv_size4, self.conv_size4), conv_block(self.conv_size4, self.conv_size4))
        
        # global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        # fc_input_size = self.conv_size3 * 7 * 7
        fc_input_size = self.conv_size4

        # upsampling layers
        self.fc1 = nn.Linear(fc_input_size, self.fc_size) # 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, out_size) 

    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out

        out = self.global_max_pool(out)


        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)



class Quasi_ResNet9(Model_Base):
    def __init__(
            self, in_shape, out_shape, conv_size1 = 128, conv_size2 = 512, conv_size3=64, fc1_size = 1024,
            loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3

        self.fc1_size = fc1_size

        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(Quasi_ResNet9, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = nn.Sequential(conv_block(self.conv_size1, self.conv_size1), 
                                  conv_block(self.conv_size1, self.conv_size1))

        self.conv2 = conv_block(self.conv_size1, self.conv_size2, pool=True,) # output: conv_size2 x 15 x 15
        self.res2 = nn.Sequential(conv_block(self.conv_size2, self.conv_size2), 
                                  conv_block(self.conv_size2, self.conv_size2))

        
        self.conv3 = nn.Conv2d(self.conv_size2, self.conv_size3, kernel_size=3, padding=1)
        
        # global max pooling
        adaptive_sampling_size = (5,5)
        self.global_max_pool = nn.AdaptiveMaxPool2d(adaptive_sampling_size)

        # fc_input_size = self.conv_size3 * 7 * 7
        fc_input_size = self.conv_size3 * adaptive_sampling_size[0] * adaptive_sampling_size[1]

        # upsampling layers
        self.fc1 = nn.Linear(fc_input_size, self.fc1_size) # 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_size, out_size) 

    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out

        out = self.conv3(out)
        out = self.global_max_pool(out)


        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)