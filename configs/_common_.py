# Copyright (c) QIU Tian. All rights reserved.

# runtime
device = 'cuda'
seed = 42
epochs = 300
clip_max_norm = 1.0
eval_interval = 1
num_workers = 8
pin_memory = True
sync_bn = True
find_unused_params = False
dist_url = 'env://'
print_freq = 50
need_targets = False
amp = True

# dataset
data_root = './data'
dataset = ...

# model
model_lib = 'default'
model = ...

# criterion
criterion = 'ce'

# optimizer
optimizer = 'adamw'
weight_decay = 5e-2

# lr_scheduler
scheduler = 'cosine'
warmup_epochs = 20
warmup_lr = 1e-06
min_lr = 1e-05

# evaluator
evaluator = 'default'

# loading
no_pretrain = True

# saving
output_root = "./runs"
save_interval = 5
