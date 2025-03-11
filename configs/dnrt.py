# Copyright (c) QIU Tian. All rights reserved.

import os

from _base_ import *
from qtcls.modules.DNRT import RAA

model = 'vit_tiny_patch4_32'
dataset = 'cifar100'

image_size = 32
batch_size = 256
lr = 0.0005 * (batch_size / 512)
output_dir = f'{output_root}/{dataset}/{model}/{os.path.splitext(os.path.basename(__file__))[0]}'
model_kwargs = dict(act_layer=RAA, drop_path_rate=0.1)

# Comment out the following lines when using the '--resume/-r' and '--eval' options for evaluation purposes only because
# the response memory is not saved in the checkpoint and is empty under this circumstance which will cause an error.
# Usually, you don't need to use '--eval' as the training process would already display and record the evaluation
# results of each epoch on the terminal and in the log file.
need_targets = True
criterion = 'arr'
arr_loss_coef = 20  # 10 for cifar10
