import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
# from numba import cuda

from my_packages.neural_network.gpu_aux import to_device

class Model_Base(nn.Module):
    def __init__(
            self, 
            binary_loss_fn=nn.BCEWithLogitsLoss(),
            magnitude_loss_fn=nn.MSELoss(),
            phase_loss_fn=nn.MSELoss(),
            compounded_loss_fn: callable = lambda x,y,z: x+y+z,
            apply_sigmoid_to_accuracy=True, 
            *args, **kwargs):
        super(Model_Base, self).__init__()
        # initialize loss functions
        self.loss_fn_binary = binary_loss_fn
        self.loss_fn_magnitude = magnitude_loss_fn
        self.loss_fn_phase = phase_loss_fn

        # initialize compounded loss function
        self.compounded_loss_fn = compounded_loss_fn

        # applies sigmoid to accuracy for the binary loss
        self.apply_sigmoid_to_accuracy = apply_sigmoid_to_accuracy


    def training_step(self, batch):
        inputs, targets = batch
        binary_target, magnitude_target, phase_target = targets
        binary_pred, magnitude_pred, phase_pred = self(inputs)

        # calculate different losses
        binary_loss = self.loss_fn_binary(binary_pred, binary_target)
        magnitude_loss = self.loss_fn_magnitude(magnitude_pred, magnitude_target)
        phase_loss = self.loss_fn_phase(phase_pred, phase_target)
    
        return binary_loss, magnitude_loss, phase_loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        binary_target, magnitude_target, phase_target = targets
        binary_pred, magnitude_pred, phase_pred = self(inputs)

        # calculate different losses
        binary_loss = self.loss_fn_binary(binary_pred, binary_target)
        magnitude_loss = self.loss_fn_magnitude(magnitude_pred, magnitude_target)
        phase_loss = self.loss_fn_phase(phase_pred, phase_target)

        compounded_loss = self.compounded_loss_fn(binary_loss, magnitude_loss, phase_loss)
        accuracy = self._accuracy(binary_loss, binary_target, apply_sigmoid=self.apply_sigmoid_to_accuracy)
        
        return {
                'val_binary_loss': binary_loss.detach(),
                'val_magnitude_loss': magnitude_loss.detach(),
                'val_phase_loss': phase_loss.detach(),
                'val_loss': compounded_loss.detach(), # for early stopping
                'val_acc': accuracy
        }
    
    

    def validation_epoch_end(self, outputs):
        batch_binary_losses = [x['val_binary_loss'] for x in outputs]
        epoch_binary_loss = torch.stack(batch_binary_losses).mean()
        batch_magnitude_losses = [x['val_magnitude_loss'] for x in outputs]
        epoch_magnitude_loss = torch.stack(batch_magnitude_losses).mean()
        batch_phase_losses = [x['val_phase_loss'] for x in outputs]
        epoch_phase_loss = torch.stack(batch_phase_losses).mean()
        batch_compounded_losses = [x['val_loss'] for x in outputs]
        epoch_compounded_loss = torch.stack(batch_compounded_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {
                'val_binary_loss': epoch_binary_loss.item(), 
                'val_magnitude_loss': epoch_magnitude_loss.item(), 
                'val_phase_loss': epoch_phase_loss.item(), 
                'val_loss': epoch_compounded_loss.item(), 
                'val_acc': epoch_acc.item()
                }
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result['train_loss'],
            result['val_loss'],
            result['val_acc']))

        
    def evaluate(self, val_loader):
        self.eval() # set to evaluation mode
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def print_summary(self, in_shape, device = "cpu"):
        return summary(self, input_size=in_shape, device=device)

    
    @staticmethod
    def _accuracy(out, targets, thresh=0.5, apply_sigmoid=False):
        with torch.no_grad():
            if apply_sigmoid:
                out = torch.sigmoid(out)
            # Convert output probabilities to binary values (0 or 1)
            out_binary = (out > thresh).float()

            # Calculate true positives, false positives and false negatives
            true_positives = (out_binary * targets).sum().item()
            false_positives = (out_binary * (1 - targets)).sum().item()
            false_negatives = ((1 - out_binary) * targets).sum().item()

            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)

            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            return torch.tensor(f1_score)