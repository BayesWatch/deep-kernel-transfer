import numpy as np
import torch
import torch.nn as nn

from data.data_generator import SinusoidalDataGenerator, Nasdaq100padding
from data.qmul_loader import get_batch, train_people, test_people


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.layer4 = nn.Linear(2916, 1)

    def return_clones(self):
        layer4_w = self.layer4.weight.data.clone().detach()
        layer4_b = self.layer4.bias.data.clone().detach()

    def assign_clones(self, weights_list):
        self.layer4.weight.data.copy_(weights_list[0])
        self.layer4.weight.data.copy_(weights_list[1])

    def forward(self, x):
        out = self.layer4(x)
        return out


class FeatureTransfer(nn.Module):
    def __init__(self, backbone, device):
        super(FeatureTransfer, self).__init__()
        regressor = Regressor()
        self.feature_extractor = backbone
        self.model = Regressor()
        self.criterion = nn.MSELoss()
        self.device = device

    def train_loop(self, epoch, optimizer, params, results_logger):
        if params.dataset == "sines":
            batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.num_tasks,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase,
                                                                      params.noise).generate()
            batch = torch.from_numpy(batch)
            batch_labels = torch.from_numpy(batch_labels)
        elif params.dataset == "nasdaq":
            nasdaq100padding = Nasdaq100padding(directory=self.config.data_dir['nasdaq'], normalize=True,
                                                partition="train", window=params.update_batch_size * 2,
                                                time_to_predict=params.meta_batch_size * 2)
            data_loader = torch.utils.data.DataLoader(nasdaq100padding, batch_size=params.update_batch_size * 2,
                                                      shuffle=True)
            batch, batch_labels = next(iter(data_loader))
            batch = batch.reshape(params.update_batch_size * 2, params.meta_batch_size * 2, 1)
            batch_labels = batch_labels[:, :, 0]
        else:
            batch, batch_labels = get_batch(train_people)

        batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)

        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            output = self.model(self.feature_extractor(inputs))
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step()

            if (epoch % 10 == 0):
                print('[%d] - Loss: %.3f' % (
                    epoch, loss.item()
                ))
                results_logger.log("epoch", epoch)
                results_logger.log("loss", loss.item())

    def test_loop(self, n_support, optimizer, params):  # we need optimizer to take one gradient step
        if params.dataset == "sines":
            batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.num_tasks,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase,
                                                                      params.noise).generate()

            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels)
        elif params.dataset == "nasdaq":
            nasdaq100padding = Nasdaq100padding(directory=self.config.data_dir['nasdaq'], normalize=True,
                                                partition="test", window=params.update_batch_size * 2,
                                                time_to_predict=params.meta_batch_size * 2)
            data_loader = torch.utils.data.DataLoader(nasdaq100padding, batch_size=params.update_batch_size * 2,
                                                      shuffle=True)
            batch, batch_labels = next(iter(data_loader))
            batch = batch.reshape(params.update_batch_size * 2, params.meta_batch_size * 2, 1)
            batch_labels = batch_labels[:, :, 0]

            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels)
        else:
            inputs, targets = get_batch(train_people)

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind = [i for i in range(19) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :].to(self.device)
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, len(test_people) - 1)

        optimizer.zero_grad()
        z_support = self.feature_extractor(x_support[n]).detach()
        output_support = self.model(z_support).squeeze()
        loss = self.criterion(output_support, y_support[n])
        loss.backward()
        optimizer.step()

        self.feature_extractor.eval()
        self.model.eval()
        z_all = self.feature_extractor(x_all[n]).detach()
        output_all = self.model(z_all).squeeze()
        return self.criterion(output_all, y_all[n])

    def save_checkpoint(self, checkpoint):
        torch.save({'feature_extractor': self.feature_extractor.state_dict(), 'model': self.model.state_dict()},
                   checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.feature_extractor.load_state_dict(ckpt['feature_extractor'])
        self.model.load_state_dict(ckpt['model'])
