Official pytorch implementation of the paper: 

*"Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels"* (2020) Patacchiola, M., Turner, J., Crowley, E. J., O'Boyle, M., & Storkey, A., *Advances in Neural Information Processing (NeurIPS, Spotlight)* [[arXiv]](https://arxiv.org/abs/1910.05199)

```bibtex
@inproceedings{patacchiola2020bayesian,
  title={Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels},
  author={Patacchiola, Massimiliano and Turner, Jack and Crowley, Elliot J. and Storkey, Amos},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

**Overview.** We introduce a Bayesian meta-learning method based on [Gaussian Processes (GPs)](https://en.wikipedia.org/wiki/Gaussian_process) to tackle the problem of few-shot learning. We propose a simple, yet effective variant of deep kernel learning in which the kernel is transferred across tasks, which we call *Deep Kernel Transfer (DKT)*. This approach is straightforward to implement, provides uncertainty quantification, and does not require estimation of task-specific parameters. We empirically demonstrate that DKT outperforms several state-of-the-art algorithms in few-shot regression, classification, and cross-domain adaptation.

**NOTE**: previous pre-prints of this paper have used the names "GPNet" and "GPShot". In the published version we are using the name "DKT".

Requirements
-------------

1. Python >= 3.x
2. Numpy >= 1.17
3. [pyTorch](https://pytorch.org/) >= 1.2.0
4. [GPyTorch](https://gpytorch.ai/) >= 0.3.5
5. (optional) [TensorboardX](https://pypi.org/project/tensorboardX/) 
 
**WARNING:** some users have experienced issues in running some of the scripts due to the error: *"Matrix not positive definite"*. This is likely caused by the latest versions of GPyTorch. The configuration that is working on our system uses: `gpytorch 1.0.1, python 3.6.9, torch 1.8.1`. We suggest to replicate this configuration in a conda environment in case you experience the same issue.


Installation
-------------

```
pip install numpy torch torchvision gpytorch h5py pillow
```

We confirm that the following configuration worked for us: numpy 1.18.1, torch 1.4.0, torchvision 0.5.0, gpytorch 1.0.1, h5py 5.10.0, pillow 7.0.0

DKT: code of our method
--------------------------

**Regression.** The implementation of our method is based on the [gpyTorch](https://gpytorch.ai/) library. The code for the regression case is available in [DKT_regression.py](./methods/DKT_regression.py).

**Classification.** The code for the classification case is accessible in [DKT.py](./methods/DKT.py), with most of the important pieces contained in the `train_loop()` method (training), and in the `correct()` method (testing). 

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

**Methods.** There are a few available methods that you can use: `DKT`, `maml`, `maml_approx`, `protonet`, `relationnet`, `matchingnet`, `baseline`, `baseline++`. You must use those exact strings at training and test time when you call the script (see below). Note that our method is `DKT`, and that `baseline` corresponds to feature transfer in our paper. By default DKT has a `BNCosSim` kernel, to change this please edit the entry in `configs.py`.

**Backbone.** The script allows training and testing on different backbone networks. By default the script will use the same backbone used in our experiments (`Conv4`). Check the file `backbone.py` for the available architectures, and use the parameter `--model=BACKBONE_STRING` where `BACKBONE_STRING` is one of the following: `Conv4`, `Conv6`, `ResNet10|18|34|50|101`.

Regression
-----------

**QMUL Head Pose Trajectory Regression.** In order to run this experiment you first have to download and setup the QMUL dataset, this can be done automatically running the file `download_QMUL.sh` from the folder `filelists/QMUL`. Moreover, you have to change the kernel type, editing the entry in `configs.py` (default `BNCosSim`) to `rbf` or `spectral`. Please note that other kernels are not supported for this experiment and their use will raise an error. The methods that can be used for regression are `DKT` and `transfer` (feature transfer). In order to train these methods, use:

```
python train_regression.py --method="DKT" --seed=1
```

The number of training epochs can be set with `--stop_epoch`. The above command will  save a checkpoint to `save/checkpoints/QMUL/Conv3_DKT`, which you can test on the test set with:

```
python test_regression.py --method="DKT" --seed=1
```

You can additionally specify the size of the support set with `--n_support` (which defaults to 5), and the number of test epochs with `--n_test_epochs` (which defaults to 10). 


**Periodic functions.** The code for the periodic functions experiments is available in the [sines](./sines) folder. This needs some adjustment of the parameters at the code level to reproduce the in-range and out-of-range conditions (see the associated [README](./sines/README.md)).


Classification
---------------

**Train classification.** The various methods can be trained using the following syntax:

```
python train.py --dataset="miniImagenet" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug
```

This will train DKT 5-way 1-shot on the mini-ImageNet dataset with seed 1. The `dataset` string can be one of the following: `CUB`, `miniImagenet`. At training time the best model is evaluated on the validation set and stored as `best_model.tar` in the folder `./save/checkpoints/DATASET_NAME`. The parameter `--train_aug` enables data augmentation. The parameter `seed` set the seed for pytorch, numpy, and random. Set `--seed=0` or remove the parameter for a random seed. Additional parameters are reported in the file `io_utils.py`.

**Test classification.** For testing `DKT`, `maml` and `maml_approx` it is enough to repeat the train command replacing the call to `train.py` with the call to `test.py` as follows:

```
python test.py --dataset="miniImagenet" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug
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
python train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1
```

Note that the parameter `--train_aug` (data augmentation) is not used for `cross_char` but only for `cross`.


Acknowledgements
---------------

This repository is a fork of: [https://github.com/wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)
