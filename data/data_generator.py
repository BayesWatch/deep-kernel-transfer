""" Code for loading data. """

from multiprocessing.dummy import freeze_support

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

INPUT_DIM = 1


class SinusoidalDataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid data.
    A "class" is considered a particular sinusoid function.
    """

    def __init__(self, num_samples_per_class, batch_size, output_dim=1, multidimensional_amp=False,
                 multidimensional_phase=True, gaussian_noise=True):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.generate = self.generate_sinusoid_batch
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.dim_input = INPUT_DIM
        self.dim_output = output_dim
        self.multidimensional_amp = multidimensional_amp
        self.multidimensional_phase = multidimensional_phase
        self.gaussian_noise = gaussian_noise

    def generate_sinusoid_batch(self, input_idx=None):
        # input_idx is used during qualitative testing --the number of examples used for the grad update

        if self.multidimensional_amp:
            # y_1 = A_1*sinus(x_1+phi)
            # y_2 = A_2*sinus(x_2+phi)
            # ...
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size, self.dim_output])
        else:
            # y_1 = A*sinus(x_1+phi)
            # y_2 = A*sinus(x_2+phi)
            # ...
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])

        if self.multidimensional_phase:
            # y_1 = A*sinus(x_1+phi_1)
            # y_2 = A*sinus(x_2+phi_2)
            # ...
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size, self.dim_output])
        else:
            # y_1 = A*sinus(x_1+phi)
            # y_2 = A*sinus(x_2+phi)
            # ...
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        if self.gaussian_noise is True:
            noise = np.random.normal(0, 1, [self.batch_size, self.num_samples_per_class, self.dim_output])
        else:
            noise = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])

        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  [self.num_samples_per_class, self.dim_input])
            if input_idx is not None:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1],
                                                            num=self.num_samples_per_class - input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func]) + noise[func]
        return init_inputs.astype(np.float32), outputs.astype(np.float32), amp.astype(np.float32), phase.astype(
            np.float32)


class Nasdaq100padding(Dataset):
    """Nasdaq100padding dataset."""

    def __init__(self, directory="../filelists/Nasdaq_100/nasdaq100_padding.csv", normalize=None, partition="train",
                 window=10,
                 time_to_predict=10):
        self.df = pd.read_csv(directory)
        self.partition = partition
        self.window = window
        self.time_to_predict = time_to_predict

        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(self.df)
            self.df = pd.DataFrame(x_scaled, columns=self.df.columns)
        x_train, x_test = train_test_split(self.df, test_size=0.2, random_state=42, shuffle=False)
        self.df_test = pd.DataFrame(x_test, columns=self.df.columns).reset_index(drop=True)
        self.df_train = pd.DataFrame(x_train, columns=self.df.columns).reset_index(drop=True)

    def __len__(self):
        if self.partition == "train":
            return len(self.df_train) - self.window - self.time_to_predict
        if self.partition == "test":
            return len(self.df_test) - self.window - self.time_to_predict
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        begin = idx
        end_of_x = idx + self.window
        if self.partition == "train":
            return torch.FloatTensor(list(range(begin, end_of_x))), self.df_train.loc[begin:end_of_x - 1].values
        if self.partition == "test":
            return torch.FloatTensor(list(range(begin, end_of_x))), self.df_test.loc[begin:end_of_x - 1].values
        else:
            raise NotImplementedError


# example
if __name__ == '__main__':
    freeze_support()
    nasdaq100padding = Nasdaq100padding(normalize=False, partition="test", window=10, time_to_predict=10)
    dataset_loader = torch.utils.data.DataLoader(nasdaq100padding,
                                                 batch_size=4, shuffle=True)

    x, y = next(iter(dataset_loader))
    print(x.reshape(4, 10, 1))
    print(y[:, :, 0])
