import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.io import loadmat
import numpy as np
from matplotlib import colors
from matplotlib import gridspec

results_path = 'Results'

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

array_slice = np.mean(loss_array, (0, 1))

norm = colors.Normalize(vmin=np.min(loss_array), vmax=np.max(loss_array))

fig, ax = plt.subplots()

fig.set_figheight(7)
fig.set_figwidth(5.7)

im = ax.pcolormesh(t_, pp_, array_slice, cmap='OrRd', norm=norm) #, interpolation='bilinear'

#ax.set_yticks(np.arange(len(pp_)), labels=[round(i, 2) for i in pp_])
#ax.set_xticks(np.arange(0, len(t_), 3), labels=[round(i, 2) for i in t_][::3])

ax.set_ylabel('P [W]')
ax.set_xlabel('t [ps]')

cbar = fig.colorbar(im, orientation='vertical')
cbar.set_label('Wartości strat', loc='center')  # Opcjonalnie, etykieta dla paska kolorów

plt.savefig(os.path.join(save_path, 'mean_plot_pp_t.eps'))
plt.savefig(os.path.join(save_path, 'mean_plot_pp_t.png'))

plt.show()