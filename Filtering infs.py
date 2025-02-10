import os
from gnlse_dataset import GnlseDataset
import torch
from tqdm.auto import tqdm

# Data path in string format
data_path = 'gamma_data/'
data_path = 'beta_data/'
data_path = 'power_data/'
data_path = 'time_data/'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')


test_data = GnlseDataset(data_dir=data_path)
aaa=[]

for i in tqdm(range(len(test_data))):
    if torch.isnan(test_data[i][0]).any() or torch.isinf(test_data[i][0]).any():
        aaa.append(test_data.file_names[i])
    if torch.isnan(test_data[i][1]).any() or torch.isinf(test_data[i][1]).any():
        print("NaN lub inf w etykietach")
print(len(aaa))
print('Deleting')
for i in tqdm(aaa):
    os.remove(os.path.join(data_path, i))
