import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
# from numba import cuda

from my_packages.neural_network.gpu_aux import to_device

class Model_Base(nn.Module):
    def __init__(self, loss_fn=F.mse_loss, apply_sigmoid_to_accuracy=True, *args, **kwargs):
        super(Model_Base, self).__init__()
        self.loss_fn = loss_fn
        self.apply_sigmoid_to_accuracy = apply_sigmoid_to_accuracy

    @property
    def device(self):
        # Return the device of the first parameter of the model
        return next(self.parameters()).device

    def training_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        accuracy = self._accuracy(out, targets, apply_sigmoid=self.apply_sigmoid_to_accuracy)
        return {'val_loss': loss.detach(), 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
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
    
    def export_to_onnx(self, path):
        dummy_input = torch.randn(1, *self.input_shape)
        torch.onnx.export(self, dummy_input, path)
    
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