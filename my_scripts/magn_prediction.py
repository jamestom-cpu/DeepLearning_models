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

    def training_step_dipole_position(self, batch, loss_fn):
        inputs, targets = batch
        binary_target = targets[:, 0]
        binary_pred, _ = self(inputs)
        loss = loss_fn(binary_pred, binary_target)
        return loss
    
    def training_step_dipole_magnitude(self, batch, loss_fn):
        inputs, targets = batch
        magnitude_targets = targets[:, 1]
        _, magnitude_pred = self(inputs)
        loss = loss_fn(magnitude_pred, magnitude_targets)
        return loss
    
    def validation_step_binary(self, batch, loss_fn):
        inputs, targets = batch
        binary_target = targets[:, 0]
        binary_pred, _ = self(inputs)
        loss = loss_fn(binary_pred, binary_target)
        accuracy = self._accuracy(binary_pred, binary_target, apply_sigmoid=True)
        return {'val_loss': loss.detach(), 'val_acc': accuracy}  

    def validation_step_magnitude(self, batch, loss_fn):
        inputs, targets = batch
        magnitude_targets = targets[:, 1]
        _, magnitude_pred = self(inputs)
        loss = loss_fn(magnitude_pred, magnitude_targets)
        return {'val_loss': loss.detach(), 'val_acc': torch.tensor(0.0)}

    def evaluate_binary(self, val_loader, loss_fn):
        self.eval()
        outputs = [self.validation_step_binary(batch, loss_fn) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def evaluate_magnitude(self, val_loader, loss_fn):
        self.eval()
        outputs = [self.validation_step_magnitude(batch, loss_fn) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    
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
            model: myModel, 
            opt_func=torch.optim.SGD, 
            loss_fn_binary=nn.BCEWithLogitsLoss(),
            loss_fn_magnitude=nn.MSELoss(),
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
        
        # initialize the loss functions
        self.loss_fn_binary = loss_fn_binary
        self.loss_fn_magnitude = loss_fn_magnitude

        # initialize the optimizer
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

    def _train_on_batch_binary(self, batch):
        loss = self.model.training_step_dipole_position(batch, self.loss_fn_binary)
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
    
    def _train_on_batch_magnitude(self, batch):
        loss = self.model.training_step_dipole_magnitude(batch, self.loss_fn_magnitude)

        # clear gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()
        self.optimizer.step()

        # log gradient statistics
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()        
        return loss.detach()


    def fit_binary(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch_binary(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate_binary(val_loader, self.loss_fn_binary)
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
    
    def fit_magnitude(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch_magnitude(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate_magnitude(val_loader, self.loss_fn_magnitude)
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
    
    def unfreeze_all_model(self, reinitialize_early_stopping=True):
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = opt_func(self.model.parameters(), self.lr)
        if reinitialize_early_stopping:
            self._init_early_stopping(patience=self.patience)
    
    def switch_to_magnitude(self):
        self._init_early_stopping(patience=self.patience)
        # Freeze the convolutional base
        for param in self.model.conv_base.parameters():
            param.requires_grad = False
        # Unfreeze the magnitude head
        for param in self.model.magntiude_head.parameters():
            param.requires_grad = True

        # Reinitialize the optimizer
        self.optimizer = opt_func(self.model.magntiude_head.parameters(), self.lr)
    
    def fit(self, epochs, train_loader, val_loader):
        print("fitting func...")
    
#%% Define DataLoaders
# create the dataloaders
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds.view_with_shape(input_shape), [train_size, val_size])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

#%% Train the model
# params
loss_fn_binary = nn.BCEWithLogitsLoss()
loss_fn_magnitude = nn.MSELoss()
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
    loss_fn_binary=loss_fn_binary, loss_fn_magnitude=loss_fn_magnitude,
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

print("evaluation before training: ", model.evaluate_binary(val_dl, loss_fn_binary))
print("evaluation before training: ", model.evaluate_magnitude(val_dl, loss_fn_magnitude))

trainer.fit_binary(10, train_dl, val_dl)
trainer.switch_to_magnitude()
trainer.fit_magnitude(10, train_dl, val_dl)
trainer.lr = 0.0001
trainer.unfreeze_all_model()
trainer.fit_magnitude(10, train_dl, val_dl)

# save the model

model_dir = "models/multi_task"
model_name = "multi_task_model_1.pt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, model_name)
torch.save(model.state_dict(), model_path)


# #%% Evaluate the model

# class DipolePredictor:
#     def __init__(self, model):
#         self.model = model

#     def predict_binary(self, input_data):
#         # Reshaping the input data to (4, 30, 30)
#         input_data = input_data.reshape(-1, 4, 30, 30)
#         self.model.eval()
#         with torch.no_grad():
#             binary_prediction, _ = self.model(torch.Tensor(input_data).to(self.model.device))
#             binary_prediction = torch.sigmoid(binary_prediction)
#         return binary_prediction.cpu().numpy()

#     def predict_magnitude_with_label(self, input_data, binary_labels):
#         # Reshaping the input data to (4, 30, 30)
#         input_data = input_data.reshape(-1, 4, 30, 30)
#         self.model.eval()
#         with torch.no_grad():
#             _, magnitude_prediction = self.model(torch.Tensor(input_data).to(self.model.device), torch.Tensor(binary_labels).to(self.model.device))
#         return magnitude_prediction.cpu().numpy()

#     def predict_magnitude(self, input_data):
#         binary_predictions = self.predict_binary(input_data)
#         magnitude_predictions = self.predict_magnitude_with_label(input_data, binary_predictions)
#         return magnitude_predictions
    

# predictor = DipolePredictor(model)
# # Assume input_data is your input with shape (2, 2, 30, 30)
# binary_predictions = predictor.predict_binary(input_data)
# magnitude_predictions_with_label = predictor.predict_magnitude_with_label(input_data, binary_labels)
# magnitude_predictions = predictor.predict_magnitude(input_data)