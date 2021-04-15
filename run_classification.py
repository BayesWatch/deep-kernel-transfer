import os
import random
import time

import numpy as np
import torch
import torch.optim
import torch.utils.data.sampler

import data.feature_loader as feat_loader
from data.datamgr import SimpleDataManager, SetDataManager
from methods.DKT import DKT
from methods.baselinefinetune import BaselineFinetune
from methods.baselinetrain import BaselineTrain
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
from models import backbone
from reporting.loggers import ResultsLogger
from training import configs
from training.io_utils import model_dict, parse_args, get_best_file, get_assigned_file, get_resume_file


def _set_seed(seed, verbose=True):
    if (seed != 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if (verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if (verbose): print("[INFO] Setting SEED: None")


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, results_logger):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        # model.eval()
        # acc = model.test_loop(val_loader)
        model.train()
        model.train_loop(epoch, base_loader, optimizer,
                         results_logger)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


def main_train(params):
    _set_seed(params)

    results_logger = ResultsLogger(params)

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'
    optimization = 'Adam'
    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600  # default
    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['DKT', 'protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml',
                           'maml_approx']:
        n_query = max(1, int(
            16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if (params.method == 'DKT'):
            model = DKT(model_dict[params.model], **train_few_shot_params)
            model.init_summary()
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')
    model = model.cuda()
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update
    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, results_logger)
    results_logger.save()


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


def single_test(params, results_logger):
    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method == 'baseline':
        model = BaselineFinetune(model_dict[params.model], **few_shot_params)
    elif params.method == 'baseline++':
        model = BaselineFinetune(model_dict[params.model], loss_type='dist', **few_shot_params)
    elif params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    elif params.method == 'DKT':
        model = DKT(model_dict[params.model], **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **few_shot_params)
        if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    # modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++']:
        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    if params.method in ['maml', 'maml_approx', 'DKT']:  # maml do not support testing with feature
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)

        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                loadfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
            else:
                loadfile = configs.data_dir['emnist'] + split + '.json'
        else:
            loadfile = configs.data_dir[params.dataset] + split + '.json'

        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        if params.adaptation:
            model.task_update_num = 100  # We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)

    else:
        novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                                  split_str + ".hdf5")  # defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query=15, adaptation=params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    with open('record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++']:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' % (
            params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way)
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
            params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way,
            params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))
        results_logger.log("single_test_acc", acc_mean)
        results_logger.log("single_test_acc_std", 1.96 * acc_std / np.sqrt(iter_num))
        results_logger.log("time", timestamp)
        results_logger.log("exp_setting", exp_setting)
        results_logger.log("acc_str", acc_str)
    return acc_mean


def main_test(params):
    results_logger = ResultsLogger(params)

    seed = params.seed
    repeat = params.repeat
    # repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_list = list()
    for i in range(seed, seed + repeat):
        if (seed != 0):
            _set_seed(i)
        else:
            _set_seed(0)
        accuracy_list.append(single_test(params, results_logger))
    print("-----------------------------")
    print(
        'Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' % (repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------")

    results_logger.log("test_acc", np.mean(accuracy_list))
    results_logger.log("test_acc_std", np.std(accuracy_list))

    results_logger.save()


if __name__ == '__main__':
    params = parse_args()
    if params.test:
        main_test(params)
    else:
        main_train(params)
