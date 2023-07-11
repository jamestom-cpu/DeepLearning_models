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


plt.switch_backend('TkAgg')
PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)


from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainers.model_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.predictor.predictor  import Predictor
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np
# torch import 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary


print("cuda available: ", torch.cuda.is_available())

## inspect the data

# import the data generator 
# from singleton_python_objects.mixed_array_generator import get_mixed_array_generator
from singleton_python_objects import Quasi_ResNet
import json

data_dir = "/ext_data/NN_data/11_res_noise/"

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)

rmg = MixedArrayGenerator(**properties)



# generate the data
random_fields, l = rmg.generate_labeled_data()
# inspect the generated data
rmg.plot_Hlabeled_data(random_fields, l)
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
mlflow_model_path = r"/workspace/mlflow/378794452446859122/034225c1ea9f44b598cb1b57b9d16c31/artifacts/models"
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