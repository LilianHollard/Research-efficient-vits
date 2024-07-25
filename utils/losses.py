"""
Inspired by LeViT DistillationLoss code

@authors: HOLLARD Lilian
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau):
        super().__init__()
        assert distillation_type in ['none', 'soft', 'hard']

        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    """
    @forward

    inputs  : original inputs that are feed to the teacher model
    outputs : Ouputs of the trained model double head 
                (either a Tensor, or a Tuple[Tensor, Tensor] with both results from original head and distillation head)
    labels: the labels for the base criterion

    """

    def forward(self, inputs, outputs, labels):

        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):  # If Tuple[T,T]
            outputs, outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)  # NO BACKPROP THROUGHT THE TEACHER MODEL

        if self.distillation_type == 'soft':
            T = self.tau

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            )

            distillation_loss *= (T * T) / outputs_kd.numel()

        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
