import torch
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.DKT import DKT
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, get_resume_file, parse_args, get_best_file , get_assigned_file

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        if(verbose): print("[INFO] Setting SEED: " + str(seed))   
    else:
        if(verbose): print("[INFO] Setting SEED: None")

class ECELoss(nn.Module):
    """ Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin.
    Adapted from: https://github.com/gpleiss/temperature_scaling
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calibrate(self, logits, labels, iterations=50, lr=0.01):
        temperature_raw = torch.ones(1, requires_grad=True, device="cuda")
        nll_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.LBFGS([temperature_raw], lr=lr, max_iter=iterations)
        softplus = nn.Softplus() #temperature must be > zero, Softplus could be used
        def closure():
            if torch.is_grad_enabled(): optimizer.zero_grad()
            #loss = nll_criterion(logits / softplus(temperature_raw.expand_as(logits)), labels)
            loss = nll_criterion(logits / temperature_raw.expand_as(logits), labels)
            if loss.requires_grad: loss.backward()
            return loss
        optimizer.step(closure)
        return temperature_raw

    def forward(self, logits, labels, temperature=1.0, onevsrest=False):
        logits_scaled = logits / temperature
        if(onevsrest):
            softmaxes = torch.sigmoid(logits_scaled) / torch.sum(torch.sigmoid(logits_scaled), dim=1, keepdim=True)
        else:
            softmaxes = torch.softmax(logits_scaled, dim=1)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def get_logits_targets(params):
    acc_all = []
    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'DKT':
        model           = DKT(model_dict[params.model], **few_shot_params)
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    if params.method in ['maml', 'maml_approx', 'DKT']: #maml do not support testing with feature
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84 
        else:
            image_size = 224

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)
        
        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
            else:
                loadfile   = configs.data_dir['CUB'] + split +'.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
            else:
                loadfile  = configs.data_dir['emnist'] + split +'.json' 
        else: 
            loadfile    = configs.data_dir[params.dataset] + split + '.json'

        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()

        logits_list = list()
        targets_list = list()    
        for i, (x,_) in enumerate(novel_loader):
            logits = model.get_logits(x).detach()
            targets = torch.tensor(np.repeat(range(params.test_n_way), model.n_query)).cuda()
            logits_list.append(logits) #.cpu().detach().numpy())
            targets_list.append(targets) #.cpu().detach().numpy())
    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5")
        cl_data_file = feat_loader.init_loader(novel_file)
        logits_list = list()
        targets_list = list()
        n_query = 15
        n_way = few_shot_params['n_way']
        n_support = few_shot_params['n_support']
        class_list = cl_data_file.keys()
        for i in range(iter_num):
            #----------------------
            select_class = random.sample(class_list,n_way)
            z_all  = []
            for cl in select_class:
                img_feat = cl_data_file[cl]
                perm_ids = np.random.permutation(len(img_feat)).tolist()
                z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch
            z_all = torch.from_numpy(np.array(z_all))
            model.n_query = n_query
            logits  = model.set_forward(z_all, is_feature = True).detach()
            targets = torch.tensor(np.repeat(range(n_way), n_query)).cuda()
            logits_list.append(logits)
            targets_list.append(targets)
            #----------------------
    return torch.cat(logits_list, 0), torch.cat(targets_list, 0)


def main():        
    params = parse_args('test')
    seed = params.seed
    repeat = params.repeat

    # 1. Find the value of temperature (calibration)    
    print("Calibration: finding temperature hyperparameter...")
    ece_module = ECELoss()
    temperature_list = list()
    for _ in range(repeat):#repeat):
        _set_seed(0) # random seed
        logits, targets = get_logits_targets(parse_args('test'))
        temperature = ece_module.calibrate(logits, targets, iterations=300, lr=0.01).item()
        if(temperature>0): temperature_list.append(temperature)
        print("Calibration: temperature", temperature, "; mean temperature", np.mean(temperature_list))
    # Filtering invalid temperatures (e.g. temp<0)
    if(len(temperature_list)>0):temperature = np.mean(temperature_list) 
    else: temperature = 1.0

    # 2. Use the temperature to record the ECE
    # repeat the test N times changing the seed in range [seed, seed+repeat]
    ece_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        logits, targets = get_logits_targets(parse_args('test'))
        #ece = ece_module.forward(logits, targets, temperature, onevsrest=params.method=='DKT').item()
        ece = ece_module.forward(logits, targets, temperature, onevsrest=False).item()
        ece_list.append(ece)
        print("ECE:", np.mean(ece_list), "+-", np.std(ece_list))

    # 3. Print the final ECE (averaged over all seeds)
    print("-----------------------------")
    print('Seeds = %d | Overall ECE = %4.4f +- %4.4f' %(repeat, np.mean(ece_list), np.std(ece_list)))
    print("-----------------------------")        
if __name__ == '__main__':
    main()
