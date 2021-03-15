import torch
import torch.nn as nn
import gpytorch



class NNKernel(gpytorch.kernels.Kernel):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, flatten=False, **kwargs):
        super(NNKernel, self).__init__(**kwargs)
    
      
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.flatten = flatten
        self.model = self.create_model()
    
    
    
    def create_model(self):

        assert self.num_layers>=1, "Number of hidden layers must be at least 1"
        modules = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        if self.flatten:
            modules = [nn.Flatten()]+modules
        for i in range(self.num_layers-1):
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        model = nn.Sequential(*modules)
        return model
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, full_covar=True, **params):
            r"""
            Computes the covariance between x1 and x2.
            This method should be imlemented by all Kernel subclasses.

            Args:
                :attr:`x1` (Tensor `n x d` or `b x n x d`):
                    First set of data
                :attr:`x2` (Tensor `m x d` or `b x m x d`):
                    Second set of data
                :attr:`diag` (bool):
                    Should the Kernel compute the whole kernel, or just the diag?
                :attr:`last_dim_is_batch` (tuple, optional):
                    If this is true, it treats the last dimension of the data as another batch dimension.
                    (Useful for additive structure over the dimensions). Default: False

            Returns:
                :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                    The exact size depends on the kernel's evaluation mode:

                    * `full_covar`: `n x m` or `b x n x m`
                    * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                    * `diag`: `n` or `b x n`
                    * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
            """
            if last_dim_is_batch:
                raise NotImplementedError()
            else:
                
                z1 = self.model(x1)
                z2 = self.model(x2)
                
                out = torch.matmul(z1, z2.T)
      
                
                if diag:
                    return torch.diag(out)
                else:
                    return out 
                

                
class MultiNNKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_tasks, kernels, **kwargs):
        super(MultiNNKernel, self).__init__(**kwargs)
        assert isinstance(kernels, list), "kernels must be a list of kernels"
        self.num_tasks = num_tasks
        self.kernels = nn.ModuleList(kernels)
        

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks        
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, full_covar=True, **params):
            r"""
            Computes the covariance between x1 and x2.
            This method should be imlemented by all Kernel subclasses.

            Args:
                :attr:`x1` (Tensor `n x d` or `b x n x d`):
                    First set of data
                :attr:`x2` (Tensor `m x d` or `b x m x d`):
                    Second set of data
                :attr:`diag` (bool):
                    Should the Kernel compute the whole kernel, or just the diag?
                :attr:`last_dim_is_batch` (tuple, optional):
                    If this is true, it treats the last dimension of the data as another batch dimension.
                    (Useful for additive structure over the dimensions). Default: False

            Returns:
                :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                    The exact size depends on the kernel's evaluation mode:

                    * `full_covar`: `n x m` or `b x n x m`
                    * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                    * `diag`: `n` or `b x n`
                    * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
            """
            if last_dim_is_batch:
                raise NotImplementedError()
            else:
                n = x1.shape[0]
                m = x2.shape[0]
                out = torch.zeros((n*self.num_tasks, m*self.num_tasks), device=x1.get_device())
                for i in range(self.num_tasks):
                    for j in range(self.num_tasks):
                        #print(x1.shape, x2.shape)
                        z1 = self.kernels[i].model(x1)
                        z2 = self.kernels[j].model(x2)

                        #print(z1.shape, z2.shape)
                        #print((i*n), n, (j*m), m)
                        out[(i*n):(i+1)*n, (j*m):(j+1)*m] = torch.matmul(z1, z2.T)       

                if diag:
                    return torch.diag(out)
                else:
                    return out     
