import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

if __name__ == "__main__":
    import sys
    PROJECT_CWD = r"/workspace/"
    sys.path.append(PROJECT_CWD)

from my_packages.neural_network.model.model_base import Model_Base
from ..NN_blocks import simple_conv_block, conv_block, linear_block



 
class Quasi_ResNet(Model_Base):
    def __init__(
            self, input_shape, output_shape, 
            conv_size1 = 64, conv_size2 = 128, conv_size3 = 256, 
            fc1_size = 512, dropout_p_fc=0.1, loss_fn=F.mse_loss):
        
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3

        self.dropout_p_fc = dropout_p_fc

        self.fc1_size = fc1_size 
        n_layers = self.input_shape[0]
        out_size = np.prod(output_shape)
        super(Quasi_ResNet, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = simple_conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = conv_block(self.conv_size1, n=2)
        

        self.conv2 = nn.Sequential(
            simple_conv_block(self.conv_size1, self.conv_size2),
            nn.MaxPool2d(2)
        )
        self.res2 = conv_block(self.conv_size2, n=2)
        
        self.conv3 = nn.Sequential(
            simple_conv_block(self.conv_size2, self.conv_size3),
            nn.MaxPool2d(2),
            simple_conv_block(self.conv_size3, self.conv_size3),
        )

        # global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        # fc_input_size = self.conv_size3 * 7 * 7
        fc_input_size = self.conv_size3

        # upsampling layers
        self.dropout3 = nn.Dropout(self.dropout_p_fc)
        self.fc1 = linear_block(fc_input_size, self.fc1_size) #
        #self.dropout4 = nn.Dropout(self.dropout_p_fc)
        self.fc2 = nn.Linear(self.fc1_size, out_size) 

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        # out = self.dropout1(out)
        out = self.conv2(out)
        out = self.res2(out) + out
        # out = self.dropout2(out)
        out = self.conv3(out)
        out = self.global_max_pool(out)


        out = out.view(out.size(0), -1)
        out = self.dropout3(out)
        out = self.fc1(out)
        #out = self.dropout4(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.output_shape[0], self.output_shape[1], self.output_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.input_shape, device=device)
    
## hyperparameters

# model structure
conv_layer1 = 64
conv_layer2 = 128
conv_layer3 = 256
fc_layer1 = 126

# training
loss_fn = nn.BCEWithLogitsLoss()

# regularization
dropout_p_fc = 0.05
dropout_p_conv = 0


## make model instance accessible
def get_model(input_shape, output_shape):
    model = Quasi_ResNet(
    input_shape,
    output_shape, 
    conv_size1=conv_layer1, 
    conv_size2= conv_layer2, 
    conv_size3=conv_layer3,
    fc1_size=fc_layer1,
    dropout_p_fc=dropout_p_fc,
    loss_fn=loss_fn)
    return model


if __name__ == "__main__":
    model = get_model((2, 30, 30), (2, 11, 11))
    print(model.print_summary(device="cpu"))