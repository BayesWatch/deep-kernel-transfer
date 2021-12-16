import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

    def sample_data(self, size=1, noise=0.0, sort=False):
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
        return x, y
        
        
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

    def sample_data(self, size=1, noise=0.0, sort=False):
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
        return x, y

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

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.layer1 = nn.Linear(1, 40)
        self.layer2 = nn.Linear(40,40)
        self.layer3 = nn.Linear(40,1)

    def return_clones(self):
        layer1_w = self.layer1.weight.data.clone().detach()
        layer1_b = self.layer1.bias.data.clone().detach()
        layer2_w = self.layer2.weight.data.clone().detach()
        layer2_b = self.layer2.bias.data.clone().detach()
        layer3_w = self.layer3.weight.data.clone().detach()
        layer3_b = self.layer3.bias.data.clone().detach()
        return [layer1_w,layer1_b,layer2_w,layer2_b,layer3_w,layer3_b]
        
    def assign_clones(self, weights_list):
        self.layer1.weight.data.copy_(weights_list[0])
        self.layer1.bias.data.copy_(weights_list[1])
        self.layer2.weight.data.copy_(weights_list[2])
        self.layer2.bias.data.copy_(weights_list[3])
        self.layer3.weight.data.copy_(weights_list[4])
        self.layer3.bias.data.copy_(weights_list[5])        

                     
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out


def main():
    #Hyperparameters to change
    n_shot_train = 10
    n_shot_test = 5
    tot_fine_tune_iterations=100
    train_range=(-5.0, 5.0)
    test_range=(-5.0, 5.0) # This must be (-5, +10) for the out-of-range condition
    tasks     = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                  phase_min=0.0, phase_max=np.pi, 
                                  x_min=train_range[0], x_max=train_range[1], 
                                  family="sine")
                              
    ## Defining model
    net = Feature()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 1e-3}])

    ## Training
    net.train()

    tot_iterations=50000
    for epoch in range(tot_iterations):
        optimizer.zero_grad()
        inputs, labels = tasks.sample_task().sample_data(n_shot_train, noise=0.1)
        output = net(inputs)
        loss = criterion(output, labels[:,None])
        loss.backward()
        optimizer.step()
        #---- print some stuff ----
        if(epoch%100==0):
            print('[%d] - Loss: %.3f' % (
                epoch, loss.item()
            ))

    ## Test phase on a new sine/cosine wave
    tasks_test = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                   phase_min=0.0, phase_max=np.pi, 
                                   x_min=test_range[0], x_max=test_range[1], 
                                   family="sine")
    print("Test, please wait...")
   
    net.eval()
    tot_iterations=5 #500
    mse_list = list()
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 1e-2}])
    weights_list = net.return_clones()
    for epoch in range(tot_iterations):
        sample_task = tasks_test.sample_task()
        sample_size = 200
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        support_indices = np.sort(indices[0:n_shot_test])

        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]

        #fine-tune on support set
        net.assign_clones(weights_list) #set to initial weights    
        for _ in range(tot_fine_tune_iterations):
            optimizer.zero_grad()       
            output_support = net(x_support)
            loss = criterion(output_support, y_support[:,None])
            loss.backward()
            optimizer.step()        

        #Evaluation on query set
        output_query = net(x_query).detach()
        mse = criterion(output_query, y_query[:,None])
        mse_list.append(mse.item())

    print("-------------------")
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")

    for i in range(10):
        sample_task = tasks_test.sample_task()
        sample_size = 200
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        support_indices = np.sort(indices[0:n_shot_test])
        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]
        #fine-tune on support set
        net.assign_clones(weights_list) #set to initial weights    
        for _ in range(tot_fine_tune_iterations):
            optimizer.zero_grad()       
            output_support = net(x_support)
            loss = criterion(output_support, y_support[:,None])
            loss.backward()
            optimizer.step() 
        #Evaluation on all data
        output_all = net(x_all).detach()
        #Plot
        fig, ax = plt.subplots()
        #true-curve
        true_curve = np.linspace(train_range[0], train_range[1], 1000)
        true_curve = [sample_task.true_function(x) for x in true_curve]
        ax.plot(np.linspace(train_range[0], train_range[1], 1000), true_curve, color='blue', linewidth=2.0)
        if(train_range[1]<test_range[1]):
            dotted_curve = np.linspace(train_range[1], test_range[1], 1000)
            dotted_curve = [sample_task.true_function(x) for x in dotted_curve]
            ax.plot(np.linspace(train_range[1], test_range[1], 1000), dotted_curve, color='blue', linestyle="--", linewidth=2.0)
        ax.plot(np.squeeze(x_all), output_all.detach().numpy(), color='red', linewidth=2.0)
        #support points
        ax.scatter(x_support, y_support, color='darkblue', marker='*', s=50, zorder=10)
                    
        #all points
        #ax.scatter(x_all.numpy(), y_all.numpy())
        #plt.show()
        plt.ylim(-6.0, 6.0)
        plt.xlim(test_range[0], test_range[1])
        plt.savefig('plot_FT_' + str(i) + '.png', dpi=300)


if __name__ == "__main__":
    main()
