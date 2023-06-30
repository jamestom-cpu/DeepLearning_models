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
from types import SimpleNamespace


plt.switch_backend('TkAgg')
PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)

# neuratl network imports
from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.predictor.predictor  import Predictor
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np

# hdf5 imports
from my_packages.hdf5.util_classes import Measurement_Handler

# torch import 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary


print("cuda available: ", torch.cuda.is_available())

## inspect the data

# import the data generator 
from singleton_python_objects import Quasi_ResNet





# load scanned data
class FieldLoader():
    def __init__(self, db_directory, db_filename, db_group_name, probe_height=6e-3):
        self.db = SimpleNamespace(directory=db_directory, filename=db_filename, savename=db_group_name)
        self.probe_height = probe_height
        self.run_scans_on_database_field()
    
    def run_scans_on_database_field(self):
         # hdf5 database properties
        fullpath = os.path.join(self.db.directory, self.db.filename)

        # load the database properties to mhandler
        self.m_handler = Measurement_Handler.from_h5_file(fullpath, self.db.savename)

        # create the target fields: Ez, Hx, Hy only magnitudes on a plane
        self.Ez = self.m_handler.E.run_scan("z", field_type="E", index = self.probe_height)
        self.Hx = self.m_handler.H.run_scan("x", field_type="H", index = self.probe_height)
        self.Hy = self.m_handler.H.run_scan("y", field_type="H", index = self.probe_height)

        assert self.Ez.f == self.Hx.f == self.Hy.f, "All fields must have the same frequency"
        assert np.allclose(self.Ez.grid, self.Hx.grid) and np.allclose(self.Ez.grid, self.Hy.grid), "All fields must have the same grid"
        self.f = [self.Ez.f] # assuming all fields have the same frequency
        self.scan_grid = np.expand_dims(self.Ez.grid, axis=-1)  # the scan grid is 2D, we need to add the third dimension


# load the data
db_directory = "/ext_data/simulated_fields"
db_name = "field_testing_for_susceptibility.hdf5"
savename = "RSTEST_2_LINEs.hdf5"

field_loader = FieldLoader(db_directory, db_name, savename, probe_height=6e-3)

Hx, Hy, Ez = field_loader.Hx, field_loader.Hy, field_loader.Ez

fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
Hx.plot_fieldmap(ax=ax[0,0])
Hy.plot_fieldmap(ax=ax[0,1])
Hx.plot(ax=ax[1,0], label="Hx")
Hy.plot(ax=ax[1,1], label="Hy")
plt.show()
# predictor 
def preprocessing(x):
    # get the H fields
    H = x[-2:]
    maxH = np.max(np.abs(H))
    minH = np.min(np.abs(H))
    H = (H - minH)/(maxH - minH)
    return H

def postprocessing(x):
    return 1/(1+np.exp(-x)) # sigmoid


input_shape =   (2, 30, 30)
output_shape =  (2, 11, 11)

model = Quasi_ResNet.get_model(input_shape=input_shape, output_shape=output_shape)
print(model.print_summary(device="cpu"))

## load mlflow model
import mlflow.pytorch
mlflow_model_path = r"/workspace/mlflow/829057909622999832/700803c8139d40aea15b0b6e809d0cda/artifacts/models"
mlflow_model = mlflow.pytorch.load_model(mlflow_model_path)


predictor = Predictor(
    preprocessing_func=preprocessing, 
    postprocessing_func=postprocessing,
    model=mlflow_model).consider_specific_height(index=0)
# predictor.load_model_and_weights(model_path, device="cuda")



prediction = predictor.predict(random_fields)
probability_map = predictor.prediction_probability_map(random_fields)



labelsH = l[-2:]
accuracy = predictor.accuracy(random_fields, labelsH, certainty_level=0.5)


fig, ax = plt.subplots(3,2, figsize=(15,4.5), constrained_layout=True)
rmg.plot_Hlabeled_data(random_fields, labelsH, ax=(ax[0,0], ax[0,1]))
predictor.plot(random_fields, certainty_level=0.5, ax= ax[1:])
plt.show()

print("finished")