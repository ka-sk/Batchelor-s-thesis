import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.io import loadmat
import numpy as np
from matplotlib import colors
from matplotlib import gridspec

results_path = 'Results'

BATCH_SIZE = 32 

model_ = {'save_dir': 'Whole_model',
               'data_dir': 'new_data'}

train_path = os.path.join(model_['data_dir'], 'train')
test_path = os.path.join(model_['data_dir'], 'test')

save_path = os.path.join(results_path, model_['save_dir'])

##########################################################################

data_dict = loadmat(os.path.join(save_path + '/save_dict.mat'))

b = data_dict['b'][0, :]
g = data_dict['g'][0, :]
pp = data_dict['pp'][0, :]
t = data_dict['t'][0, :]
loss = data_dict['loss'][0, :]

b_ = list(set(b))
b_.sort()
g_ = list(set(g))
g_.sort()
pp_ = list(set(pp))
pp_.sort()
t_ = list(set(t))
t_.sort()

loss_array = np.zeros((len(b_), len(g_), len(pp_), len(t_)))

for i in range(len(loss)):
    b_i = b_.index(b[i])
    g_i = g_.index(g[i])
    pp_i = pp_.index(pp[i])
    t_i = t_.index(t[i])
    
    loss_array[b_i, g_i, pp_i, t_i] = loss[i]
    
##########################################################################
                
b_i = [4, 13, 16]
g_i = [4, 11, 15]

fig_nr = 0

norm = colors.Normalize(vmin=np.min(loss_array), vmax=np.max(loss_array))

fig = plt.figure(figsize=(10, 10))
spec = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.6, hspace=0.6)

ax = []
for b_idx, b_slice in enumerate(b_i):
    row_ax = []
    for g_idx, g_slice in enumerate(g_i):
        ax_idx = fig.add_subplot(spec[b_idx, g_idx])
        array_slice = loss_array[b_slice, g_slice, :, :]
        im = ax_idx.pcolormesh(t_, pp_, array_slice, cmap='OrRd', norm=norm)
        fig_nr += 1

        ax_idx.set_xlabel('t [ps]')
        ax_idx.set_ylabel('P [W]')
        row_ax.append(ax_idx)

        print(f'{array_slice}\n\n\n\n')
    ax.append(row_ax)

# Pasek kolorów w ostatniej kolumnie
cbar_ax = fig.add_subplot(spec[:, 3])
cbar = fig.colorbar(im, cax=cbar_ax, location='right')
cbar.set_label('Wartości strat', loc='center')

# Dodanie opisów wierszy (b_i) i kolumn (g_i) za pomocą fig.text
for b_idx in range(len(b_i)):
    fig.text(0.04, 0.75 - b_idx * 0.25, f'β_2 = {round(b_[b_i[b_idx]], 4)} [ps^2/m]',
             ha='center', va='center', fontsize=12, rotation=90)

for g_idx in range(len(g_i)):
    fig.text(0.22 + g_idx * 0.25, 0.94, f'γ = {round(g_[g_i[g_idx]], 4)} [1/W/m]',
             ha='center', va='center', fontsize=12)

plt.savefig(os.path.join(save_path, '9_plots_pp_t.eps'))
plt.savefig(os.path.join(save_path, '9_plots_pp_t.png'))

plt.show()

########################################################################

