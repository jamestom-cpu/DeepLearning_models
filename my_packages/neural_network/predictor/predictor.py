import torch
import numpy as np
import functools

class Predictor:
    def __init__(
            self, model=None, 
            preprocessing_func = lambda x: x, 
            postprocessing_func = lambda x: x,
            device="cpu"):
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
    def accuracy(self, inputs, targets, thresh=0.5):
        predictions = self.model(inputs)
        return self.model._accuracy(predictions, targets, thresh)
    
    @model_wrapper
    def predict(self, inputs):
        return self.model(inputs)


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
    
     

