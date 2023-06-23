print("starting script")


import os, sys
import h5py
import numpy as np
import pandas as pd
import scipy
import math as m
import cmath
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint




PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base

# torch import 
import torch
from torch.utils.data import TensorDataset, DataLoader

print("cuda available: ", torch.cuda.is_available())
print("number of GPUs: ",torch.cuda.device_count())
print("I am currently using device number: ", torch.cuda.current_device())
print("the device object is: ", torch.cuda.device(0))
print("the device name is: ", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()


from my_packages.neural_network.model.early_stopping import EarlyStopping


# consider the GPU
from my_packages.neural_network.gpu_aux import get_default_device, to_device, DeviceDataLoader
from torchsummary import summary

from torch import nn
import torch.nn.functional as F
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_H import H_Components_Dataset
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_E import E_Components_Dataset



# data parameters
resolution=(11,11)
field_res = (30,30)
xbounds = [-1e-2, 1e-2]
ybounds = [-1e-2, 1e-2]
padding = None
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.4
dynamic_range = 2
probe_heights = [3e-3, 6e-3, 8e-3]
dipole_density_E = 0.1
dipole_density_H = 0.1
inclde_dipole_position_uncertainty = False
# data_dir = "/share/NN_data/high_res_with_noise"
data_dir = "/workspace/NN_data/11_res_uncertain_position"


rmg = MixedArrayGenerator(
    resolution=resolution,
    xbounds=xbounds,
    ybounds=ybounds,
    padding=padding,
    dipole_height=dipole_height,
    substrate_thickness=substrate_thickness,
    substrate_epsilon_r=substrate_epsilon_r,
    probe_height=probe_heights,
    dynamic_range=dynamic_range,
    f=[1e9],
    field_res=field_res,
    dipole_density_E=dipole_density_E,
    dipole_density_H=dipole_density_H,
    include_dipole_position_uncertainty=inclde_dipole_position_uncertainty,
    )

data_iterator = DataIterator(rmg)


fields,labels = data_iterator.generate_N_data_samples(10)

f, t = fields[0], labels[0]


ds = TensorDataset(torch.from_numpy(fields), torch.from_numpy(labels))
Eds = E_Components_Dataset(ds, probe_height_index=1).unpad_label().scale_to_01()
Hds = H_Components_Dataset(ds, probe_height_index=1).unpad_label().scale_to_01()

fH, lH = Hds[0]

rmg.plot_labeled_data(f, t, index=1)
rmg.plot_Hlabeled_data(fH, lH, mask_padding=1)  
plt.show()

print("f")

# N = 100000
# N_test = 1000


# # save the datasets
# save_dir = os.path.join(PROJECT_CWD, "NN_data", "mixed_array_data")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# fullpath_train = os.path.join(save_dir, "train_and_valid_dataset.pt")
# fullpath_test = os.path.join(save_dir, "test_dataset.pt")
fullpath_train = os.path.join(data_dir, "train_and_valid_dataset.pt")
fullpath_test = os.path.join(data_dir, "test_dataset.pt")


# load the data from the datasets
train_and_valid_dataset = torch.load(fullpath_train)
test_dataset = torch.load(fullpath_test)

# only consider the probe height of 6mm

# # create a new dataset 
# train_and_valid_dataset = train_and_valid_dataset[:, 0, ...][0], train_and_valid_dataset[:][0]
# train_and_valid_dataset = TensorDataset(*train_and_valid_dataset)

# test_dataset = test_dataset[:, 0, ...][0], test_dataset[:][0]
# test_dataset = TensorDataset(*test_dataset)


height_index = 0

Hds = H_Components_Dataset(train_and_valid_dataset, probe_height_index=height_index).scale_to_01()
Eds = E_Components_Dataset(train_and_valid_dataset, probe_height_index=height_index).unpad_label().scale_to_01()

Hds_test = H_Components_Dataset(test_dataset, probe_height_index=height_index).unpad_label().scale_to_01() 
Eds_test = E_Components_Dataset(test_dataset, probe_height_index=height_index).unpad_label().scale_to_01()




# split into training and validation sets
train_size = int(0.8 * len(Eds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds, [train_size, val_size])

print("train_dataset size: ", len(train_dataset))
print("val_dataset size: ", len(val_dataset))



# # inspect data
# n_examples = 5


# # plot the examples
# plt.switch_backend('TkAgg')
# fig, axs = plt.subplots(n_examples, 2, figsize=(15, 5))

# for i in range(n_examples):
#     H_examples, labels_examples = train_dataset[i]
#     rmg.plot_Hlabeled_data(H_examples, labels_examples, ax=axs[i])
# plt.show()


## Define the Model Structure

def conv_block(in_channels, out_channels, pool=False, batch_norm=True):
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


class Quasi_ResNet(Model_Base):
    def __init__(
            self, in_shape, out_shape, conv_size1 = 64, conv_size2 = 128, conv_size3 = 256,
            fc1_size = 512, dropout_p_fc = 0.15, dropout_p_conv=0.1,
            loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2
        self.conv_size3 = conv_size3

        self.dropout_p_fc = dropout_p_fc
        self.dropout_p_conv = dropout_p_conv

        self.fc1_size = fc1_size 
        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(Quasi_ResNet, self).__init__(loss_fn=loss_fn, apply_sigmoid_to_accuracy=True)

        # conv layers
        self.conv1 = conv_block(n_layers, self.conv_size1) # output: conv_size1 x 30 x 30
        self.res1 = nn.Sequential(conv_block(self.conv_size1, self.conv_size1), 
                                  conv_block(self.conv_size1, self.conv_size1))
        
        self.dropout1 = nn.Dropout(self.dropout_p_conv)

        self.conv2 = conv_block(self.conv_size1, self.conv_size2, pool=True,) # output: conv_size2 x 15 x 15
        self.res2 = nn.Sequential(conv_block(self.conv_size2, self.conv_size2), 
                                  conv_block(self.conv_size2, self.conv_size2))

        self.dropout2 = nn.Dropout(self.dropout_p_conv)

        self.conv3 = conv_block(self.conv_size2, self.conv_size3, pool=True) # output: conv_size3 x 7 x 7
        self.res3 = nn.Sequential(conv_block(self.conv_size3, self.conv_size3), conv_block(self.conv_size3, self.conv_size3))
        
        # global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

        # fc_input_size = self.conv_size3 * 7 * 7
        fc_input_size = self.conv_size3

        # upsampling layers
        self.dropout3 = nn.Dropout(self.dropout_p_fc)
        self.fc1 = nn.Linear(fc_input_size, self.fc1_size) # 
        self.relu1 = nn.ReLU()
        #self.dropout4 = nn.Dropout(self.dropout_p_fc)
        self.fc2 = nn.Linear(self.fc1_size, out_size) 

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.dropout2(out)
        out = self.conv3(out)
        out = self.res3(out) + out

        out = self.global_max_pool(out)


        out = out.view(out.size(0), -1)
        out = self.dropout3(out)
        out = self.fc1(out)
        out = self.relu1(out)
        #out = self.dropout4(out)
        out = self.fc2(out)
        out = out.view(out.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)

device = get_default_device()
print("device: ", device)


batch_size = 256   
conv_layer1 = 128
conv_layer2 = 256
conv_layer3 = 512
fc_layer1 = 1024
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.001
patience = 5
lr_dampling_factor = 0.5
lr_patience = 0
opt_func = torch.optim.Adam
n_iterations = 100

# regularization
dropout_p_fc = 0.05
dropout_p_conv = 0
weight_decay= 1e-6


# create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True ,num_workers = 4,  pin_memory=True)
test_dataloader = DataLoader(Hds_test, batch_size=batch_size, shuffle=True)
# move the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dataloader, device)
val_dl = DeviceDataLoader(val_dataloader, device)

input_shape =   (2, 30, 30)
output_shape =  (2, 11, 11)

model = Quasi_ResNet(
    input_shape,
    output_shape, 
    conv_size1=conv_layer1, 
    conv_size2= conv_layer2, 
    conv_size3=conv_layer3,
    fc1_size=fc_layer1,
    dropout_p_fc=dropout_p_fc,
    dropout_p_conv=dropout_p_conv,
    loss_fn=loss_fn)
print(model.print_summary(device="cpu"))


# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_electric")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
experiment_name = "t1_30x30 -> 11x11"
run_name = "ResNet_2reg"

trainer = Trainer(
    model, opt_func=opt_func,
    lr=lr, patience=patience, 
    optimizer_kwargs={"weight_decay":weight_decay},
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':lr_patience, 
                      'verbose':True}, 
    model_dir=model_dir, experiment_name=experiment_name, run_name=run_name,
    log_gradient=["conv1", "conv2", "fc1"], log_weights=[], parameters_of_interest={
        "conv_layer1": conv_layer1,
        "conv_layer2": conv_layer2,
        "fc_layer1": fc_layer1,
    }, print_every_n_epochs=1,
    log_mlflow=True, log_tensorboard=True
    )


model = to_device(model, device)
print("evaluation before training: ", model.evaluate(val_dl))


torch.cuda.empty_cache()


print("starting training")
history = trainer.fit(n_iterations, train_dl, val_dl)


# use the model to evaluate the test set
print("evaluation after training")
test_dl = DeviceDataLoader(test_dataloader, device)
print("evaluation on the test set: ", model.evaluate(test_dataloader))

# try clearing the cache
torch.save(model.state_dict(), os.path.join(model_dir, "temp.pt"))
