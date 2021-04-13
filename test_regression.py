import logging

import numpy as np
import torch
import torch.optim as optim
import os
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
    model = DKT(bb, device, num_tasks=params.output_dim,  config=config, dataset=params.dataset)
    optimizer = None
elif params.method == 'transfer':
    model = FeatureTransfer(bb, device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

mse_list = []
mean_list = []
lower_list = []
upper_list = []
x_list = []
y_list = []
lists = (mse_list, mean_list, lower_list, upper_list, x_list, y_list)
for epoch in range(params.n_test_epochs):
    if params.dataset != "sines":
        mse = float(model.test_loop(params.n_support, params).cpu().detach().numpy())
        mse_list.append(mse)
    else:
        res = model.test_loop(params.n_support, params)
        for i, r in enumerate(res):
            lists[i].append(r.cpu().detach().numpy())

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")



if params.dataset == "sines":
    save_dict = {"mse": mse_list, "mean": mean_list, "lower": lower_list, "upper": upper_list, "x_list": x_list, "y_list":y_list}
    save_dir = '%scheckpoints/%s' % (config.save_dir, params.dataset)
    np.save(os.path.join(save_dir, "results_sines.npy"), save_dict)




