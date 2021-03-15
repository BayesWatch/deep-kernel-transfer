import logging

import numpy as np
import torch
import torch.optim as optim

import backbone
import configs
from io_utils import parse_args_regression
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer
from configs import Config

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device: {}'.format(device))

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config(params)

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (config.save_dir, params.dataset, params.model, params.method)


if params.dataset == "sines":
    bb = backbone.MLP(input_dim=1, output_dim=params.output_dim).to(device)
else:
    bb = backbone.Conv3().to(device)



if params.method == 'DKT':
    model = DKT(bb, device, num_tasks=params.output_dim,  config=config)
    optimizer = None
elif params.method == 'transfer':
    model = FeatureTransfer(bb, device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

mse_list = []
for epoch in range(params.n_test_epochs):
    mse = float(model.test_loop(params.n_support, optimizer, params).cpu().detach().numpy())
    mse_list.append(mse)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")
