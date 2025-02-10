import torch
from torchvision import transforms
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt


def loss_fn(pred, acc, features):
    res = ((pred - acc) ** 2).detach().numpy()
    return {features[i]: res[i] for i in len(features)}


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

results_path = 'Results'

# transform function
transform_fun = transforms.Compose([])

w_model = {'save_dir': 'Whole_model',
               'data_dir': 'new_data',
               'name': 'Cały model'}

gamma = {'save_dir': 'Gamma_model',
         'data_dir': 'gamma_data',
               'name': 'Współczynnik nieliniowości'}

beta = {'save_dir': 'Beta_model',
        'data_dir': 'beta_data',
               'name': 'Współczynnik dyspersji'}

pp = {'save_dir': 'Peak_power_model',
      'data_dir': 'power_data',
               'name': 'Moc wiązki lasera'}

time = {'save_dir': 'Duration_model',
        'data_dir': 'time_data',
               'name': 'Szerokość połówkowa pulsu w domienie czasowej'}

whole_model = [w_model, gamma, beta, pp, time]
idx = 0
# iteracja po kazdym modelu
for model in whole_model:
    train_path = os.path.join(model['data_dir'], 'train')
    test_path = os.path.join(model['data_dir'], 'test')

    save_path = os.path.join(results_path, model['save_dir'])

    results = loadmat(os.path.join(save_path, 'results.mat'))
    test_acc = results['test_acc'][3:, 0]
    train_acc = results['train_acc'][3:, 0]

    plt.figure(idx)
    idx+=1
    plt.scatter(range(len(train_acc)), train_acc, label='Straty trenowania', s=6, c='saddlebrown')
    plt.scatter(range(len(test_acc)), test_acc, label='Straty testowania', s=6, c='olive')
    plt.title(model['name'])
    plt.xlabel('Epoka trenowania')
    plt.ylabel('Wartość funkcji strat')
    plt.legend()
    plt.savefig(os.path.join(save_path, model['data_dir'] + '.eps'))
    plt.savefig(os.path.join(save_path, model['data_dir'] + '.png'))
plt.show()

