import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class InvLR(_LRScheduler):
    """Decays the learning rate accroding to inv lr schedule
    """

    def __init__(self, optimizer, gamma=0.0001, power=0.75, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        super(InvLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (
            (1 + self.gamma * self.last_epoch)
            / (1 + self.gamma * (self.last_epoch - 1))
        ) ** (-self.power)
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            base_lr * ((1 + self.gamma * self.last_epoch) ** (-self.power))
            for base_lr in self.base_lrs
        ]
