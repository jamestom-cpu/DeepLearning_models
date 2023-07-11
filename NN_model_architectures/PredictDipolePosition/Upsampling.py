import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

from my_packages.neural_network.model.model_base import Model_Base


def conv_block(channels, n=2, batch_norm=True):
    layers = [
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.ReLU()
        ]*n
    if batch_norm:
        layers.append(nn.BatchNorm2d(channels))
    return nn.Sequential(*layers)

def simple_conv_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
            )

def upsample_block(in_channels, out_channels, scale):
    layers = [nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)]
    layers += [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
        nn.ReLU()
        ]
    return nn.Sequential(*layers)
            
def linear_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())


class Quasi_ResNet(Model_Base):
    def __init__(
            self, input_shape, output_shape, 
            conv_size1 = 64, conv_size2 = 128, conv_size3 = 256, conv_size4 = 512,
            up_sample_size1 = 128, up_sample_size2 = 64,
            loss_fn=nn.BCEWithLogitsLoss()):
        
        

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3
        self.conv_size4 = conv_size4
        self.up_sample_size1 = up_sample_size1
        self.up_sample_size2 = up_sample_size2
        n_layers = self.input_shape[0]
        out_size = np.prod(output_shape)


        self.n_initial_upsample_layers = 256
        self.initial_upsample_dim = 3
        self.upscaled_shape= 5

        # initialize the base model
        super(Quasi_ResNet, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = simple_conv_block(n_layers, self.conv_size1)
        self.res1 = conv_block(self.conv_size1, n=2)

        self.conv2 = simple_conv_block(self.conv_size1, self.conv_size2)
        self.res2 = conv_block(self.conv_size2, n=2)
        self.conv3 = simple_conv_block(self.conv_size2, self.conv_size3)

        self.conv4 = simple_conv_block(self.conv_size3, self.conv_size4)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        # fc connection
        upsample_input_size = self.initial_upsample_dim**2*self.n_initial_upsample_layers
        self.fc_connection = linear_block(
            self.conv_size4,
            upsample_input_size
        )

        # upsample layers
        scale_factor = self.upscaled_shape/self.initial_upsample_dim
        self.up1 = upsample_block(self.n_initial_upsample_layers, self.up_sample_size1, scale=scale_factor)
        self.res3 = conv_block(self.up_sample_size1, n=2)
        
        # head
        flat_input_shape = self.up_sample_size1*self.upscaled_shape**2
        self.fc1 = linear_block(flat_input_shape, out_size)


    def forward(self, xb):
        # forward pass through the network
        x = self.conv1(xb)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = nn.MaxPool2d(2)(x)
        x = self.res2(x) + x
        x = self.conv3(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv4(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        # connection with upsample layers
        x = self.fc_connection(x)
        x = x.view(x.size(0), self.n_initial_upsample_layers, self.initial_upsample_dim, self.initial_upsample_dim)
        x = self.up1(x)
        x = self.res3(x) + x
        
        
        # head
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), *self.output_shape)
        return x
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.input_shape, device=device)
    
      
input_shape = (2, 30, 30)
output_shape = (2, 11, 11)
conv_size1 = 64
conv_size2 = 128
conv_size3 = 256
conv_size4 = 512
upsample_size1 = 256

# fc_size = 1024
loss_fn = nn.BCEWithLogitsLoss()

# training


# model = Quasi_ResNet(input_shape = input_shape, output_shape = output_shape,
#                      conv_size1=conv_size1, conv_size2=conv_size2, conv_size3=conv_size3, conv_size4=conv_size4,
#                      up_sample_size1= upsample_size1
#                      )
# summary(model, input_shape, device="cpu")

def get_model(input_shape, output_shape, loss_fn=nn.BCEWithLogitsLoss()):
    return Quasi_ResNet(input_shape = input_shape, output_shape = output_shape,
                     conv_size1=conv_size1, conv_size2=conv_size2, conv_size3=conv_size3, conv_size4=conv_size4,
                     up_sample_size1= upsample_size1
                     )