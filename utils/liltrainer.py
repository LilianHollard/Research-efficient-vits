import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from utils import utils
from utils.losses import DistillationLoss

import os


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # init process group
    init_process_group('nccl', rank=rank, world_size=world_size)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class LilTrainer(nn.Module):

    def __init__(self, model_):
        super().__init__()
        self.model = model_

    def training_step(self, batch, device, criterion):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self.model(images)  # Generate predictions

        if isinstance(criterion, DistillationLoss):
            loss = criterion(images, out, labels)
        else:
            loss = criterion(out, labels)  # Calculate loss

        return loss

    def validation_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = self.model(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    @torch.no_grad()
    def evaluate(self, test_loader, device):
        self.model.eval()
        outputs = [self.validation_step(batch, device) for batch in test_loader]
        return self.validation_epoch_end(outputs)

    def fit_one_cycle(self, epochs, max_lr, train_loader, test_loader, teacher_model=None
                      , weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, device="cpu", ddp=False):
        torch.cuda.empty_cache()
        history = []

        # Set up cutom optimizer with weight decay
        optimizer = opt_func(self.model.parameters(), max_lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
        # steps_per_epoch=len(train_loader))

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30000,
        )
        # sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
        #                                        steps_per_epoch=len(train_loader))

        # grad_scaler = torch.cuda.amp.GradScaler()

        criterion = nn.CrossEntropyLoss()  # LeViT code is not very clear on which loss is used

        if teacher_model is not None:
            criterion = DistillationLoss(criterion, teacher_model, "soft", 0.5, 1.0)

        best = 0.0
        for epoch in range(epochs):
            # Training Phase
            if ddp:
                self.model.train_data.sampler.set_epoch(epoch)

            self.model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = self.training_step(batch, device, criterion)
                train_losses.append(loss)
                # loss.backward()

                # Gradient clipping
                # if grad_clip:
                #    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.synchronize()
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))

                sched.step()

            # Validation phase
            result = self.evaluate(test_loader, device)
            if best < results['val_acc']:
                best = results['val_acc']
                utils.save_on_master({
                    'model': model

                })

            result['train_loss'] = torch.stack(train_losses).mean().item()
            # result['lrs'] = lrs
            self.epoch_end(epoch, result)
            history.append(result)
        return history
