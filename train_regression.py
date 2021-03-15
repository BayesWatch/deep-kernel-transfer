import logging
import os

import numpy as np
import torch

import backbone
from io_utils import parse_args_regression
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer
from configs import Config

params = parse_args_regression('train_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
config = Config(params)

params.checkpoint_dir = '%scheckpoints/%s/' % (config.save_dir, params.dataset)
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (config.save_dir, params.dataset, params.model, params.method)

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device: {}'.format(device))

if params.dataset == "sines":
    bb = backbone.MLP(input_dim=1, output_dim=params.output_dim).to(device)
else:
    bb = backbone.Conv3().to(device)



if params.method == 'DKT':
    model = DKT(bb, device, num_tasks=params.output_dim, config=config)
elif params.method == 'transfer':
    model = FeatureTransfer(bb, device)
else:
    ValueError('Unrecognised method')

optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                              {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

for epoch in range(params.stop_epoch):
    model.train_loop(epoch, optimizer, params)

model.save_checkpoint(params.checkpoint_dir)
