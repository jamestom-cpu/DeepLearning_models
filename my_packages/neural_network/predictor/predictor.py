import torch
import numpy as np
import functools
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np
import matplotlib.pyplot as plt
from my_packages.neural_network.gpu_aux import to_device


class Predictor:
    def __init__(
            self, model=None, 
            preprocessing_func = lambda x: x, 
            postprocessing_func = lambda x: x,
            device="cpu"):    
        if model is not None:
            to_device(model, device)
        self.model = model
        self.device = device
        self.preprocessing_function = preprocessing_func
        self.postprocessing_function = postprocessing_func

    def load_model_and_weights(self, path:str, model=None, device="cpu"):
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("model is None, please provide a model")
        
        model = self._load_model_and_weights(model, path, device)
        self.model = model
        return model
    
    
    @staticmethod
    def _load_model_and_weights(model, path, device="cpu"):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    
    def preprocess(self, inputs: np.ndarray):
        # the preprocessing function should be a function that takes a numpy array and returns a numpy array
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        self.processed_inputs = self.preprocessing_function(inputs)
        return self.processed_inputs
    
    def postprocess(self, outputs: np.ndarray):
        # the postprocessing function should be a function that takes a numpy array and returns a numpy array
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        self.processed_outputs = self.postprocessing_function(outputs)
        return self.processed_outputs
    
    def model_wrapper(func):
        @functools.wraps(func)
        def wrapper(self, inputs, *args, **kwargs):
            inputs = self.preprocess(inputs)
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            
            inputs = inputs.to(self.device)
            inputs = inputs.float()
            
            # Check if the batch dimension is present
            if len(inputs.shape) == 3:
                # Add a batch dimension
                inputs.unsqueeze_(0)

            with torch.no_grad():
                torch_outputs = func(self, inputs, *args, **kwargs)
            
            # back to numpy array
            outputs = torch_outputs.cpu().numpy()
            prediction = self.postprocess(outputs)
            return prediction
        return wrapper

    
    
    @model_wrapper
    def prediction_probability_map(self, inputs):
        predictions = self.model(inputs)
        if inputs.shape[0] == 1:
            predictions = predictions.squeeze(0)
        return predictions
    
    def predict(self, inputs, certainty_level=0.5):
        prob_map = self.prediction_probability_map(inputs)
        prediction = np.where(prob_map>certainty_level, 1, 0)
        return prediction
    
    def accuracy(self, inputs, targets, certainty_level=0.5):
        prediction_prob_map = self.prediction_probability_map(inputs)
        if inputs.shape[0] == 1:
            prediction_prob_map = prediction_prob_map.unsqueeze(0)
        accuracy = f1_score_np(prediction_prob_map, targets, certainty_level)
        return accuracy

    def plot(self, inputs, certainty_level=0.5, ax=None):
        prediction = self.predict(inputs, certainty_level=certainty_level)
        n_layers = prediction.shape[0]
        probability_map = self.prediction_probability_map(inputs)
        if ax is None:
            fig, ax = plt.subplots(2, n_layers, figsize=(9,4.5), constrained_layout=True)
            fig.suptitle("Dipole Prediction")
        self.plot_predictions(prediction, inputs, ax=ax[0])
        self.plot_probability_maps(probability_map, prediction, ax=ax[1])
        

    @staticmethod
    def plot_inputs(inputs, ax=None):
        n_layers = inputs.shape[0]
        if ax is None:
            fig, ax = plt.subplots(1, n_layers, figsize=(9,4.5), constrained_layout=False)
            # plt.subplots_adjust(wspace=0.4)
            
        # Adjust the horizontal spacing
        #       
        for i in range(n_layers):
            data = inputs[i]
            p1 = ax[i].pcolor(data.T, cmap="jet")
            colorbar = plt.colorbar(p1, ax=ax[i])

            # Get minimum and maximum of the data
            vmin = np.min(data)
            vmax = np.max(data)

            # Create 5 evenly spaced ticks between vmin and vmax
            ticks = np.linspace(vmin, vmax, 5)

            # Set ticks on the colorbar
            colorbar.set_ticks(ticks)

            # Optional: Format the tick labels. You can change this to any format you want.
            colorbar.ax.set_yticklabels(['{:.2e}'.format(tick) for tick in ticks]) 
            ax[i].set_title("input layer {}".format(i))
        return ax
    
    @staticmethod
    def plot_predictions(prediction, inputs, ax=None):
        if prediction.shape[0] == 1:
            inputs = inputs[0]
            markers = ["s"]
            colors = ["w"]
        elif prediction.shape[0] == 2:
            inputs = inputs[-2:]
            markers = [">", "^"]
            colors = ["r", "r"]
        else: 
            inputs = inputs[-3:]
            markers = ["o", ">", "^"]
            colors = ["w", "r", "r"]


        n_layers = prediction.shape[0]
        # ax = self.plot_inputs(inputs, ax=ax)

        if ax is None:
            fig, ax = plt.subplots(n_layers, 1, figsize=(9,4.5), constrained_layout=True)

        Predictor.plot_inputs(inputs, ax=ax)

        for i in range(n_layers):   
            # Create a meshgrid for the original prediction coordinates
            pred_x, pred_y = np.asarray(np.where(prediction[i] == 1))+0.5
            
            scale_x = inputs.shape[1] / prediction.shape[1]
            scale_y = inputs.shape[2] / prediction.shape[2]
            
            # Get the scaled coordinates for your markers
            marker_x = pred_x * scale_x
            marker_y = pred_y * scale_y

            ax[i].scatter(marker_x, marker_y, marker=markers[i], color="r", edgecolors="k", s=75)
            ax[i].set_title("prediction layer {}".format(i))
        return ax


    @staticmethod
    def plot_probability_maps(probability_map, prediction=None, ax=None):
        n_layers = prediction.shape[0]
        if ax is None:
            fig, ax = plt.subplots(n_layers, 1, figsize=(9,4.5), constrained_layout=True)
        
        
        
        for i in range(n_layers):
            p1 = ax[i].pcolor(probability_map[i].T, cmap='plasma', vmin=0, vmax=1)
            plt.colorbar(p1, ax=ax[i])
            if prediction is not None:
                x, y = np.asarray(np.where(prediction[i]==1))+0.5
                ax[i].scatter(x, y, marker="x", color="k", s=10)
            ax[i].set_title("probability map")

    


    # def predict(self, inputs, preprocessing_function=None, postprocessing_function=None):
    #     inputs = self.preprocess(inputs, preprocessing_function)
    #     if isinstance(inputs, np.ndarray):
    #         inputs = torch.from_numpy(inputs)
        
    #     inputs = inputs.to(self.device)
    #     inputs = inputs.float()        
    #     with torch.no_grad():
    #         # add a batch dimension
    #         inputs.unsqueeze_(0)
    #         self.torch_outputs = self.model(inputs)            
    #     # back to numpy array
    #     outputs = self.torch_outputs.cpu().numpy()
    #     prediction = self.postprocess(outputs, postprocessing_function)
    #     return prediction
    
     

