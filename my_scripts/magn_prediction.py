#%%
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
import json



PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"

from my_packages.classes.aux_classes import Grid

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.array_generator_mag_and_phase import ArrayGenerator_MagnitudesAndPhases
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainers.dipole_position_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.plotting_functions.datapoints_plotting import Plotter

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
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_multilayer import H_Components_Dataset_Multilayer
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_E import E_Components_Dataset

from singleton_python_objects.mixed_array_generator import get_mixed_array_generator
from NN_model_architectures.PredictDipolePosition.ResNet import get_model

#%% create the data generators


data_dir = "/ext_data/NN_data/11_res_noise_MP_labels/"

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)

properties["probe_height"] = [6e-3, 8e-3, 1e-2, 1.2e-2]

fullpath_train = os.path.join(data_dir, "training.pt")
fullpath_test = os.path.join(data_dir, "test.pt")


# load the data from the datasets
height_indices = [0, -1]
Hds = H_Components_Dataset_Multilayer(fullpath_train, height_indices=height_indices)
Hds = Hds.rescale_probe_heights().rescale_labels()
Hds_test = H_Components_Dataset_Multilayer(fullpath_test, height_indices=height_indices)
Hds_test = Hds_test.rescale_probe_heights().rescale_labels()


#%% inspect the dataset
from operator import itemgetter

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)



xaxis = np.linspace(*properties["xbounds"], properties["field_res"][0])
yaxis = np.linspace(*properties["ybounds"], properties["field_res"][1])
z = np.array(properties["probe_height"])

grid = Grid(np.meshgrid(xaxis, yaxis, z, indexing="ij"))
plotter = Plotter.initialize_from_res(
    grid, properties["resolution"], properties["dipole_height"], 1e9
)

input, target = Hds[6]
probe_heights = itemgetter(*height_indices)(properties["probe_height"])

print("input shape: ", input.shape)
print("target shape: ", target.shape)
print("probe heights: ", probe_heights)
plotter.plot_fields_and_magnitude(input, target, index=0)
plt.show()



# %% Define Model Type
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



class myModel(Model_Base):
    def __init__(self, input_shape, output_shape, binary_output_shape):
        super(myModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv_base = Convolutional_Base(input_shape, 256)
        self.pred_head = BinaryPredictionHead(256, binary_output_shape)
        self.magntiude_head = DipoleMagnitudePredictionHead(256, output_shape, binary_output_shape)

    def forward(self, x, binary_labels=None):
        conv_output = self.conv_base(x)
        binary_prediction = self.pred_head(conv_output)
        
        if binary_labels is None:
            # Only during inference
            binary_labels = torch.sigmoid(binary_prediction)
        
        magnitude_prediction = self.magntiude_head(conv_output, binary_labels)
        return binary_prediction, magnitude_prediction

    def training_step_dipole_position(self, batch):
        inputs, targets = batch
        binary_target = targets[0]
        binary_pred, _ = self(inputs)
        loss = self.loss_fn(binary_pred, binary_target)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        binary_target = targets[:, 0]
        binary_pred, _ = self(inputs)
        loss = self.loss_fn(binary_pred, binary_target)
        accuracy = self._accuracy(binary_pred, binary_target, apply_sigmoid=True)
        return {'val_loss': loss.detach(), 'val_acc': accuracy}    
    
    
input_shape = (4,30,30)
output_shape = (2,11,11)
binary_output_shape = (2,11,11)

model = myModel(input_shape, output_shape, binary_output_shape)
onnx_path = "onnx_models/myModel.onnx"
model.export_to_onnx(onnx_path)

to_device(model, torch.device('cuda'))
summary(model, input_shape)
#%% Define Training

from typing import Callable, Iterable, Tuple
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from my_packages.neural_network.model.early_stopping import EarlyStopping
from my_packages.neural_network.model.model_trainers.trainer_base import Trainer_Base

class Trainer(Trainer_Base):
    def __init__(
            self, 
            model: Model_Base, 
            opt_func=torch.optim.SGD, 
            lr=0.01, patience=7, 
            scheduler_kwargs={}, 
            optimizer_kwargs={},
            model_dir="models", 
            log_mlflow=True,
            log_tensorboard=True,
            experiment_name="My Experiment",
            run_name=None,
            print_every_n_epochs=3,
            log_gradient: Iterable[str]=[], 
            log_weights: Iterable[str]=[],
            parameters_of_interest: dict={},
            save_models_to_mlflow=True,
            _include_cleaning_of_mlflow_metrics=False,):
        
        self.lr = lr
        self.model = model
        self._init_optimizer(opt_func, lr, **optimizer_kwargs)
        self._init_optimizer_scheduler(**scheduler_kwargs)

        self.config = self._define_config_dict_of_interest(
            opt_func, patience, scheduler_kwargs, parameters_of_interest
            )
            
        super().__init__(model=model, patience=patience, model_dir=model_dir,
                         log_mlflow=log_mlflow, log_tensorboard=log_tensorboard,
                         experiment_name=experiment_name, run_name=run_name,
                         print_every_n_epochs=print_every_n_epochs,
                         log_gradient=log_gradient, log_weights=log_weights,
                         parameters_of_interest=parameters_of_interest,
                         save_models_to_mlflow=save_models_to_mlflow,
                         _include_cleaning_of_mlflow_metrics=_include_cleaning_of_mlflow_metrics
                         )
        
     

    def _define_config_dict_of_interest(
            self, opt_func, patience, scheduler_kwargs, 
            parameters_of_interest) -> dict:
        config = dict(
            lr=self.lr,
            opt_func=opt_func.__name__,
            patience=patience,
            damping_factor=scheduler_kwargs.get('factor', None),
        )
        config.update(parameters_of_interest)
        return config

    def _init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _init_optimizer(self, opt_func: Callable, lr, **optimizer_kwargs):
        self.optimizer = opt_func(self.model.parameters(), lr, **optimizer_kwargs)

    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
        loss.backward()
        self.optimizer.step()

        # log gradient statistics
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()
        # clear gradients
        self.optimizer.zero_grad() 
        return loss.detach()


    def fit(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = result['val_loss']
            val_acc = result['val_acc']

            self.scheduler.step(val_loss)
            
            self._log_history(train_loss, val_loss, val_acc)
            if self.log_mlflow:
                self._log_metrics_mlflow(epoch, **self.history[-1])   
            if self.log_tensorboard:
                self._log_history_tensorboard(train_loss, val_loss, val_acc, epoch)

            if (epoch) % self.print_every_n_epochs == 0:
                self.model.epoch_end(epoch, result)

            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(torch.load(self.early_stopping_checkpoint_path))
                break
        
        if self.log_mlflow:
            print("Close mlflow session")
            if self.save_models_to_mlflow:
                mlflow.pytorch.log_model(self.model, "models")
            mlflow.end_run()
        
        if self.log_tensorboard:
            print("Close tensorboard session")
            self.writer.close()
        
        print("Training finished")
        return self.history
    
#%% Define DataLoaders
# create the dataloaders
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds.view_with_shape(input_shape), [train_size, val_size])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

#%% Train the model
# params
loss_fn = nn.BCEWithLogitsLoss()
# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_electric")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
experiment_name = "MultiTask0_dipole_position"
run_name = "largeRN"

# training parameters
lr = 0.001
patience = 2
lr_dampling_factor = 0.5
lr_patience = 0
opt_func = torch.optim.Adam
n_iterations = 5
# regularization
weight_decay= 1e-6

trainer = Trainer(
    model, opt_func=opt_func,
    lr=lr, patience=patience,
    optimizer_kwargs={"weight_decay":weight_decay},
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':lr_patience,
                      'verbose':True},
    model_dir=model_dir, experiment_name=experiment_name, run_name=run_name,
    log_gradient=["conv1", "conv2", "fc1"], log_weights=[], parameters_of_interest={}, 
    print_every_n_epochs=1,
    log_mlflow=False, log_tensorboard=False
    )

device = get_default_device()
model = to_device(model, device)

# move loaders to device (GPU)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

print("evaluation before training: ", model.evaluate(val_dl))
