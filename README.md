<!---
<p align="center">
<img src="etc/images/plot_head_trajectories.png" width="800">
</p>
-->

This repository contains the official pytorch implementation of the paper: 

*"Deep Kernel Transfer in Gaussian Processes for Few-shot Learning" (2019) Patacchiola, Turner, Crowley, and Storkey* [[download paper]](https://arxiv.org/abs/1910.05199)

**Overview.** We introduce a Bayesian method based on [Gaussian Processes (GPs)](https://en.wikipedia.org/wiki/Gaussian_process) that can learn efficiently from a limited amount of data and generalize across new tasks and domains. We frame few-shot learning as a model selection problem by learning a *deep kernel* across tasks, and then using this kernel as a covariance function in a GP prior for Bayesian inference. This probabilistic treatment allows for cross-domain flexibility, and uncertainty quantification. We provide substantial experimental evidence, showing that the proposed method is better than several state-of-the-art algorithms in few-shot regression and cross-domain classification.

Cite this paper if you use the method or code in this repository as part of a published research project:

```
@article{patacchiola2019deep,
  title={Deep Kernel Transfer in Gaussian Processes for Few-shot Learning},
  author={Patacchiola, Massimiliano and Turner, Jack and Crowley, Elliot J. and Storkey, Amos},
  journal={arXiv preprint arXiv:1910.05199},
  year={2019}
}
```

Requirements
-------------

1. Python >= 3.x
2. Numpy >= 1.17
3. [pyTorch](https://pytorch.org/) >= 1.2.0
4. [GPyTorch](https://gpytorch.ai/) >= 0.3.5
5. (optional) [TensorboardX](https://pypi.org/project/tensorboardX/) 


GPNet: code of our method
--------------------------

**Regression.** The implementation of our method is based on the [gpyTorch](https://gpytorch.ai/) library. The code for the regression case is available in [gpnet_regression.py](./methods/gpnet_regression.py).

**Classification.** The code for the classification case is accessible in [gpnet.py](./methods/gpnet.py), with most of the important pieces contained in the `train_loop()` method (training), and in the `correct()` method (testing). 

Note: there is the possibility of using the [scikit](https://scikit-learn.org/stable/modules/gaussian_process.html) Laplace approximation at test time (classification only), setting `laplace=True` in `correct()`. However, this has not been investigated enough and it is not the method used in the paper.


Experiments
============

These are the instructions to train and test the methods reported in the paper in the various conditions.

**Download and prepare a dataset.** This is an example of how to download and prepare a dataset for training/testing. Here we assume the current directory is the project root folder:

```
cd filelists/DATASET_NAME/
sh download_DATASET_NAME.sh
```

Replace `DATASET_NAME` with one of the following: `omniglot`, `CUB`, `miniImagenet`, `emnist`, `QMUL`. Notice that mini-ImageNet is a large dataset that requires substantial storage, therefore you can save the dataset in another location and then change the entry in `configs.py` in accordance.

**Methods.** There are a few available methods that you can use: `gpnet`, `maml`, `maml_approx`, `protonet`, `relationnet`, `matchingnet`, `baseline`, `baseline++`. You must use those exact strings at training and test time when you call the script (see below). Note that our method is `gpnet`, and that `baseline` corresponds to feature transfer in our paper. By default GPNet has a linear kernel, to change this please edit the entry in `configs.py`.

**Backbone.** The script allows training and testing on different backbone networks. By default the script will use the same backbone used in our experiments (`Conv4`). Check the file `backbone.py` for the available architectures, and use the parameter `--model=BACKBONE_STRING` where `BACKBONE_STRING` is one of the following: `Conv4`, `Conv6`, `ResNet10|18|34|50|101`.

Regression
-----------

TODO


Classification
---------------

**Train classification.** The various methods can be trained using the following syntax:

```
python train.py --dataset="miniImagenet" --method="gpnet" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug
```

This will train GPNet 5-way 1-shot on the mini-ImageNet dataset with seed 1. The `dataset` string can be one of the following: `CUB`, `miniImagenet`. At training time the best model is evaluated on the validation set and stored as `best_model.tar` in the folder `./save/checkpoints/DATASET_NAME`. The parameter `--train_aug` enables data augmentation. The parameter `seed` set the seed for pytorch, numpy, and random. Set `--seed=0` or remove the parameter for a random seed. Additional parameters are reported in the file `io_utils.py`.

**Test classification.** For testing `gpnet`, `maml` and `maml_approx` it is enough to repeat the train command replacing the call to `train.py` with the call to `test.py` as follows:

```
python test.py --dataset="miniImagenet" --method="gpnet" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug
```

Other methods require to store the features (for efficiency) before testing, this can be done running the script `save_features.py` before calling `test.py`. For instance, if you trained a `protonet`, you should call:

```
python save_features.py --dataset="miniImagenet" --method="protonet" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug
python test.py --dataset="miniImagenet" --method="protonet" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --repeat=5 --train_aug
```

We noticed that the [original code](https://github.com/wyharveychen/CloserLookFewShot) has a large variance on test tasks. To reduce this variance we add the parameter `repeat=N`. It iterates N times with different seeds and take an average over the N tests, we used `N=5` (3000 tasks) in our experiments.


Cross-domain classification
---------------------------

For the cross-domain classification experiments the procedure is the same described previously. The only difference is that the available datasets are: `cross_char`, and `cross`. The former being `omniglot -> EMNIST`, and the latter `miniImagenet -> CUB`. Here an example of training procedure:

```
python train.py --dataset="cross_char" --method="gpnet" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1
```

Note that the parameter `--train_aug` (data augmentation) is not used for `cross_char` but only for `cross`.

Acknowledgements
---------------

This repository is a fork of: [https://github.com/wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)
