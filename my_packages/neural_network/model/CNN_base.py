import torch
from torch import nn, optim
import torch.nn.functional as F

class CNN_Base(nn.Module):
    def __init__(self, loss_fn=F.mse_loss, *args, **kwargs):
        super(CNN_Base, self).__init__()
        self.loss_fn = loss_fn


    def training_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        accuracy = self._accuracy(out, targets)
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
    
    @staticmethod
    def _accuracy(out, targets, thresh=0.5):
        with torch.no_grad():
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