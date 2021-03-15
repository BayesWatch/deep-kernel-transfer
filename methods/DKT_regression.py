## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn


from data.qmul_loader import get_batch, train_people, test_people
from data.data_generator import SinusoidalDataGenerator


from kernels import NNKernel, MultiNNKernel

class DKT(nn.Module):
    def __init__(self, backbone, device, num_tasks=1,  config=None):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = device
        self.num_tasks=num_tasks
        self.config = config
        self.get_model_likelihood_mll()# Init model, likelihood, and mll


     
    def get_model_likelihood_mll(self, train_x=None, train_y=None):

#        if (train_x is None): train_x = torch.ones(19, 2916).to(self.device)
#        if (train_y is None): train_y = torch.ones(19).to(self.device)
        if self.num_tasks == 1:
            if (train_x is None): train_x = torch.ones(10, 1).to(self.device)
            if (train_y is None): train_y = torch.ones(10).to(self.device)
        else:
            if (train_x is None): train_x = torch.ones(10, self.num_tasks).to(self.device)
            if (train_y is None): train_y = torch.ones(10, self.num_tasks).to(self.device)


        if self.num_tasks==1:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPLayer(config=self.config, train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type)
        else:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
            model = MultitaskExactGPLayer(config=self.config, train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type, num_tasks=self.num_tasks)

        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        self.mse = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass


    def train_loop(self, epoch, optimizer, params):
        #print("NUM KERNEL PARAMS {}".format(sum([p.numel() for p in self.model.parameters() if p.requires_grad])))
        #print("NUM TRANSFORM PARAMS {}".format(sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad])))
        if params.dataset != "sines":
            batch, batch_labels = get_batch(train_people)
        else:
            batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.output_dim,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase).generate()

            if self.num_tasks == 1:
                batch = torch.from_numpy(batch)
                batch_labels = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
            else:
                batch = torch.from_numpy(batch)
                batch_labels = torch.from_numpy(batch_labels)
    
        batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)
        #print(batch.shape, batch_labels.shape)
        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)

            self.model.set_train_data(inputs=z, targets=labels)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch % 10 == 0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    
    def test_loop(self, n_support, optimizer=None, params=None):
        if params is None or params.dataset != "sines":
            return self.test_loop_qmul(n_support, optimizer)
        elif params.dataset == "sines":
            return self.test_loop_sines(n_support, params, optimizer)
        else:
            raise ValueError("unknown dataset")
    
    def test_loop_qmul(self, n_support, optimizer=None):  # no optimizer needed for GP
        inputs, targets = get_batch(test_people)

        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind = [i for i in range(19) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :]
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, len(test_people) - 1)

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            pred = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

    def test_loop_sines(self, n_support, params, optimizer=None):  # no optimizer needed for GP
        batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.output_dim,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase).generate()

        if self.num_tasks == 1:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        else:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels)


        support_ind = list(np.random.choice(list(range(10)), replace=False, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :]
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, x_support.shape[0])

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            pred = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse    
    
    
    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net': nn_state_dict}, checkpoint)


    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, config, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ## RBF kernel
        if (kernel == 'rbf' or kernel == 'RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif (kernel == 'spectral'):
            #self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=1)
        elif(kernel ==  "nn"):
            kernel = NNKernel(input_dim = config.nn_config["input_dim"],
                                        output_dim = config.nn_config["output_dim"],
                                        num_layers = config.nn_config["num_layers"],
                                        hidden_dim = config.nn_config["hidden_dim"])
            self.covar_module = kernel
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, config, train_x, train_y, likelihood, kernel='nn', num_tasks=2):
        super(MultitaskExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        if(kernel == "nn"):
            kernels = []
            for i in range(num_tasks):
                kernels.append(NNKernel(input_dim = config.nn_config["input_dim"],
                                        output_dim = config.nn_config["output_dim"],
                                        num_layers = config.nn_config["num_layers"],
                                        hidden_dim = config.nn_config["hidden_dim"]))
            self.covar_module = MultiNNKernel(num_tasks, kernels)
        elif kernel == "rbf":
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
            )
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for multi-regression, use 'nn'.")            
        


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
