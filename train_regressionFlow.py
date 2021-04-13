import logging
import os

import numpy as np
import torch

import backbone
from io_utils import parse_args_regressionFlow
from methods.DKT_regressionFlow import DKT
from methods.feature_transfer_regression import FeatureTransfer
from configs import Config

from train_misc import set_cnf_options
from train_misc import add_spectral_norm
from train_misc import create_regularization_fns
from train_misc import build_model_tabular, build_conditional_cnf


params = parse_args_regressionFlow('train_regression')
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

if params.use_conditional:
    cnf = build_conditional_cnf(params, 1, params.context_dim).to(device)
else:
    regularization_fns, regularization_coeffs = create_regularization_fns(params)
    cnf = build_model_tabular(params, 1, regularization_fns).to(device)
if params.spectral_norm: add_spectral_norm(cnf)
set_cnf_options(params, cnf)

if params.method == 'DKT':
    model = DKT(bb, cnf, device, use_conditional=params.use_conditional,
                num_tasks=params.output_dim, config=config, dataset=params.dataset)
else:
    ValueError('Unrecognised method')

optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                              {'params': model.cnf.parameters(), 'lr': 0.001},
                              {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

for epoch in range(params.stop_epoch):
    model.train_loop(epoch, optimizer, params)


model.save_checkpoint(params.checkpoint_dir)
