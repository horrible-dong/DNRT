# Copyright (c) QIU Tian. All rights reserved.

from .DNRT import ARRLoss
from .cross_entropy import CrossEntropy


def build_criterion(args):
    criterion_name = args.criterion.lower()

    if criterion_name == 'ce':
        losses = ['labels']
        weight_dict = {'loss_ce': 1}
        return CrossEntropy(losses=losses, weight_dict=weight_dict)

    if criterion_name == 'arr':
        losses = ['labels', 'arr']
        weight_dict = {'loss_ce': 1, 'loss_arr': args.arr_loss_coef}
        return ARRLoss(losses=losses, weight_dict=weight_dict)

    raise ValueError(f"Criterion '{criterion_name}' is not found.")
