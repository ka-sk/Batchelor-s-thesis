import torch
import nn_model
from gnlse_dataset import GnlseDataset
from torchvision import transforms
import os
import numpy as np


def loss_fn(pred, acc, features):
    res = ((pred - acc) ** 2).detach().numpy()
    return {features[i]: res[i] for i in len(features)}


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

results_path = 'Results'

# transform function
transform_fun = transforms.Compose([])

gamma = {'save_dir': 'Gamma_model',
         'data_dir': 'gamma_data'}

beta = {'save_dir': 'Beta_model',
        'data_dir': 'beta_data'}

pp = {'save_dir': 'Peak_power_model',
      'data_dir': 'power_data'}

time = {'save_dir': 'Duration_model',
        'data_dir': 'time_data'}

whole_model = {'save_dir': 'Whole_model',
               'data_dir': 'new_data'}

final_boss_of_my_thesis = [gamma, beta, pp, time, whole_model]

# iteracja po kazdym modelu
for elem in final_boss_of_my_thesis:
    train_path = os.path.join(elem['data_dir'], 'train')
    test_path = os.path.join(elem['data_dir'], 'test')

    save_path = os.path.join(results_path, elem['save_dir'])

    # inicjalizacja modelu i wczytanie go
    model = nn_model.CnnModel(input_shape=1,
                              hidden_units=80,
                              output_shape=1)
    model = torch.load(os.path.join(save_path, 'state_dict.pth'), map_location=device, weights_only=True)

    test_loss = {}

    for data in [GnlseDataset(train_path, transform_fun), GnlseDataset(test_path, transform_fun)]:

        # Put model in eval mode
        model.eval()

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for (X, y) in enumerate(data):
                # Send data to target device
                X, y = X.to(device).double(), y.to(device).double()

                # 1. Forward pass
                y_pred = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(y_pred, y)
                test_loss = {i: np.concatenate(test_loss[i], loss[i]) for i in loss.keys()}


# porównanie ich z właściwymi wartościami
