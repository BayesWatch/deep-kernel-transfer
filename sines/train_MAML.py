# Code adapted from: 
# https://github.com/vmikulik/maml-pytorch
# https://github.com/cbfinn/maml

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax

    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        return self.amplitude * np.sin(self.phase + x)

    def sample_data(self, size=1, noise=0.0, sort=False, gpu=False):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        x = np.random.uniform(self.xmin, self.xmax, size)
        if(sort): x = np.sort(x)
        y = self.true_function(x)
        if(noise>0): y += np.random.normal(loc=0.0, scale=noise, size=y.shape)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float)
        if(gpu): return x.cuda(), y.cuda()
        else: return x, y
        
        
class Cosine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax

    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        return self.amplitude * np.cos(self.phase + x)

    def sample_data(self, size=1, noise=0.0, sort=False, gpu=False):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        x = np.random.uniform(self.xmin, self.xmax, size)
        if(sort): x = np.sort(x)
        y = self.true_function(x)
        if(noise>0): y += np.random.normal(loc=0.0, scale=noise, size=y.shape)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float)
        if(gpu): return x.cuda(), y.cuda()
        else: return x, y

class Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """

    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max, family="sine"):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        self.family = family

    def sample_task(self):
        """
        Sample from the task distribution.

        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        if(self.family=="sine"):
            return Sine_Task(amplitude, phase, self.x_min, self.x_max)
        elif(self.family=="cosine"):
            return Cosine_Task(amplitude, phase, self.x_min, self.x_max)
        else:
            return None


class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1,40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40,1))
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x
        
class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        self.tasks = tasks
        self.model = model
        self.weights = list(model.parameters()) # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
        
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        self.tasks_per_meta_batch = tasks_per_meta_batch 
        
        # metrics
        self.plot_every = 10
        self.print_every = 100
        self.meta_losses = []
    
    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        
        # perform training on data sampled from task
        X, y = task.sample_data(self.K, noise=0.1)
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y[:,None]) / self.K
            
            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        
        # sample new data for meta-update and compute loss
        X, y = task.sample_data(self.K, noise=0.1)
        loss = self.criterion(self.model.parameterised(X, temp_weights), y[:,None]) / self.K
        
        return loss
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task = self.tasks.sample_task()
                meta_loss += self.inner_loop(task)
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0


def loss_on_random_task(initial_model, K, num_steps, tasks, optim=torch.optim.SGD):
    """
    trains the model on a random sine task and measures the loss curve.
    
    for each n in num_steps_measured, records the model function after n gradient updates.
    """
    
    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1,40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40,40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40,1))
    ]))
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), 0.01)

    # train model on a random task
    task = tasks.sample_task()
    X, y = task.sample_data(200, noise=0.1, sort=True)    
    indices = np.arange(200)
    np.random.shuffle(indices)
    support_indices = np.sort(indices[0:K])
    query_indices = np.sort(indices[K:])
    X_support = X[support_indices]
    y_support = y[support_indices]
    X_query = X[query_indices]
    y_query = y[query_indices]
        
    for step in range(1, num_steps+1):
        loss = criterion(model(X_support), y_support[:,None]) / K
        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    #Evaluate on query set
    loss = criterion(model(X_query), y_query[:,None])       
    return loss               

def average_losses(initial_model, n_samples, tasks, K=10, n_steps=10, optim=torch.optim.SGD):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    #x = np.linspace(-5, 5, 2) # dummy input for test_on_new_task
    avg_losses = list()
    for i in range(n_samples):
        loss = loss_on_random_task(initial_model, K, n_steps, tasks, optim)
        avg_losses.append(loss.item())    
    return avg_losses

def model_functions_at_training(initial_model, X, y, sampled_steps, x_axis, optim=torch.optim.SGD, lr=0.01):
    """
    trains the model on X, y and measures the loss curve.
    
    for each n in sampled_steps, records model(x_axis) after n gradient updates.
    """
    
    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1,40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40,40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40,1))
    ]))
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), lr)

    # train model on a random task
    num_steps = max(sampled_steps)
    K = X.shape[0]
    
    losses = []
    outputs = {}
    for step in range(1, num_steps+1):
        loss = criterion(model(X), y[:,None]) / K
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

        # plot the model function
        if step in sampled_steps:
            outputs[step] = model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1)).detach().numpy()
            
    outputs['initial'] = initial_model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1)).detach().numpy()
    
    return outputs, losses
    
def plot_sampled_performance(initial_model, model_name, task, X, y, test_range, train_range, name, optim=torch.optim.SGD, lr=0.01):    
    x_axis = np.linspace(test_range[0], test_range[1], 1000)
    sampled_steps=[10] #[1,10]
    outputs, losses = model_functions_at_training(initial_model, 
                                                  X, y, 
                                                  sampled_steps=sampled_steps, 
                                                  x_axis=x_axis, 
                                                  optim=optim, lr=lr)    
    fig, ax = plt.subplots()
    true_curve = np.linspace(train_range[0], train_range[1], 1000)
    true_curve = [task.true_function(x) for x in true_curve]
    ax.plot(np.linspace(train_range[0], train_range[1], 1000), true_curve, color='blue', linewidth=2.0)
    #ax.plot(x_axis, task.true_function(x_axis), color='blue', linewidth=2.0, label='true function')
    ax.scatter(X, y, color='darkblue', marker='*', s=50, zorder=10, label='data')
    if(train_range[1]<test_range[1]):
        dotted_curve = np.linspace(train_range[1], test_range[1], 1000)
        dotted_curve = [task.true_function(x) for x in dotted_curve]
        ax.plot(np.linspace(train_range[1], test_range[1], 1000), dotted_curve, color='blue', linestyle="--", linewidth=2.0)
    
    step=sampled_steps[0]
    plt.plot(x_axis, outputs[step], color='red', linewidth=2.0,
                 label='model after {} steps'.format(sampled_steps))
    # plot losses
    plt.ylim(-6.0, 6.0)
    plt.xlim(test_range[0], test_range[1])
    plt.savefig('plot_regression_maml' + str(name) + '.png', dpi=300)



def main():     
    ## Simulation Parameters
    train_iterations = 10000
    inner_steps = 1   
    train_range=(-5.0, 5.0)
    test_range=(-5.0, 5.0)  # This must be (-5, +10) for the out-of-range condition
    
    ## Train phase               
    tasks = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                          phase_min=0.0, phase_max=np.pi, 
                          x_min=train_range[0], x_max=train_range[1], 
                          family="sine")
    maml = MAML(MAMLModel(), tasks, inner_lr=0.01, meta_lr=0.001)
    maml.main_loop(num_iterations=train_iterations)

    ## Test phase
    K = 5
    tasks = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                          phase_min=0.0, phase_max=np.pi, 
                          x_min=test_range[0], x_max=test_range[1], 
                          family="sine")
    print("Test, please wait...")
    mse_list = average_losses(maml.model.model, n_samples=500, tasks=tasks, K=5, n_steps=inner_steps, optim=torch.optim.Adam)
    print("-------------------")
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")
    
    for i in range(5):
        task = tasks.sample_task()
        X, y = task.sample_data(K, noise=0.1)
        plot_sampled_performance(maml.model.model, 'MAML', task, X, y, test_range, train_range, name="_seed"+str(i))


if __name__ == '__main__':
    main()       
