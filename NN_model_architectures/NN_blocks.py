
import torch
import torch.nn as nn
import torch.nn.functional as F


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