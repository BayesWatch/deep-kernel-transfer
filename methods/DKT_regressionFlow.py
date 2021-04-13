## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn


from data.qmul_loader import get_batch, train_people, test_people
from data.data_generator import SinusoidalDataGenerator


from kernels import NNKernel, MultiNNKernel
from utils import normal_logprob


def get_transforms(model, use_context):
    if use_context:
        def sample_fn(z, context=None, logpz=None):
            if logpz is not None:
                return model(z, context, logpz, reverse=True)
            else:
                return model(z, context, reverse=True)

        def density_fn(x, context=None, logpx=None):
            if logpx is not None:
                return model(x, context, logpx, reverse=False)
            else:
                return model(x, context, reverse=False)
    else:
        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

    return sample_fn, density_fn


class DKT(nn.Module):
    def __init__(self, backbone, cnf, device, use_conditional, num_tasks=1,  config=None, dataset='QMUL'):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = device
        self.num_tasks = num_tasks
        self.config = config
        self.dataset = dataset
        self.get_model_likelihood_mll()# Init model, likelihood, and mll
        self.cnf = cnf
        self.use_conditional = use_conditional

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if self.dataset == 'QMUL':
            if (train_x is None): train_x = torch.ones(19, 2916).to(self.device)
            if (train_y is None): train_y = torch.ones(19).to(self.device)
        else:
            if self.num_tasks == 1:
                if (train_x is None): train_x = torch.ones(10, 1).to(self.device)
                if (train_y is None): train_y = torch.ones(10).to(self.device)
            else:
                if (train_x is None): train_x = torch.ones(10, self.num_tasks).to(self.device)
                if (train_y is None): train_y = torch.ones(10, self.num_tasks).to(self.device)

        if self.num_tasks == 1:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPLayer(config=self.config, train_x=train_x,
                                 train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type)
        else:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
            model = MultitaskExactGPLayer(config=self.config, train_x=train_x, train_y=train_y, likelihood=likelihood,
                                          kernel=self.config.kernel_type, num_tasks=self.num_tasks)
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
        if self.dataset != "sines":
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
            labels = labels.unsqueeze(1)
            if self.use_conditional:
                y, delta_log_py = self.cnf(labels, self.model.kernel.model(z),
                                           torch.zeros(labels.size(0), 1).to(labels))
            else:
                y, delta_log_py = self.cnf(labels, torch.zeros(labels.size(0), 1).to(labels))
            delta_log_py = delta_log_py.view(y.size(0), y.size(1), 1).sum(1)
            y = y.squeeze()
            self.model.set_train_data(inputs=z, targets=y)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets) + torch.mean(delta_log_py)
            loss.backward()
            optimizer.step()
            sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
            if self.use_conditional:
                new_means = sample_fn(predictions.mean.unsqueeze(1), self.model.kernel.model(z))
            else:
                new_means = sample_fn(predictions.mean.unsqueeze(1))
            mse = self.mse(new_means, labels)

            if (epoch % 10 == 0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    
    def test_loop(self, n_support, params=None):
        if params is None or self.dataset != "sines":
            return self.test_loop_qmul(n_support)
        elif self.dataset == "sines":
            return self.test_loop_sines(n_support, params)
        else:
            raise ValueError("unknown dataset")
    
    def test_loop_qmul(self, n_support):  # no optimizer needed for GP
        inputs, targets = get_batch(test_people)
        sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
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
            if self.use_conditional:
                y, delta_log_py = self.cnf(y_all[n].unsqueeze(1), self.model.kernel.model(z_query),
                                           torch.zeros(y_all[n].size(0), 1).to(y_all[n].unsqueeze(1)))
                new_means = sample_fn(pred.mean.unsqueeze(1), self.model.kernel.model(z_query))
            else:
                y, delta_log_py = self.cnf(y_all[n].unsqueeze(1),
                                           torch.zeros(y_all[n].size(0), 1).to(y_all[n].unsqueeze(1)))
                new_means = sample_fn(pred.mean.unsqueeze(1))

            log_py = normal_logprob(y.squeeze(), pred.mean, pred.stddev)

            NLL = -1.0 * torch.mean(log_py - delta_log_py.squeeze())
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(new_means, y_all[n])

        return mse, NLL


    def test_loop_sines(self, n_support, params):  # no optimizer needed for GP
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
            mean = pred.mean
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse, mean, lower, upper, x_all[n], y_all[n]    

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        cnf_dict = self.cnf.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict,
                    'net': nn_state_dict, 'cnf': cnf_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
        self.cnf.load_state_dict(ckpt['cnf'])


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
            self.kernel = NNKernel(input_dim = config.nn_config["input_dim"],
                                        output_dim = config.nn_config["output_dim"],
                                        num_layers = config.nn_config["num_layers"],
                                        hidden_dim = config.nn_config["hidden_dim"])
            self.covar_module = self.kernel
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
