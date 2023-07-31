
import torch

from NN_model_architectures.NN_blocks import simple_conv_block, conv_block, linear_block
from NN_model_architectures.PredictDipoleMoments.MultiTask1 import Convolutional_Base, BinaryPredictionHead, DipoleMagnitudePredictionHead
from NN_model_architectures.PredictDipoleMoments.MultiTask1 import SmallConv_Base
from .model_base import Model_Base
from NN_model_architectures.PredictDipoleMoments.MultiTask2 import ModelStructure

class MultiTargetModel(Model_Base):
    def __init__(self, input_shape, output_shape, binary_output_shape):
        super(MultiTargetModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.arch = ModelStructure(input_shape, output_shape, binary_output_shape)

    def forward(self, x, binary_labels=None):
        return self.arch(x, binary_labels)
    

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