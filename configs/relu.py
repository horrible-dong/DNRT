# Copyright (c) QIU Tian. All rights reserved.

import os

from _base_ import *
from qtcls.modules.DNRT import ReLU

model = 'vit_tiny_patch4_32'
dataset = 'cifar100'

image_size = 32
batch_size = 256
lr = 0.0005 * (batch_size / 512)
output_dir = f'{output_root}/{dataset}/{model}/{os.path.splitext(os.path.basename(__file__))[0]}'
model_kwargs = dict(act_layer=ReLU, drop_path_rate=0.1)
