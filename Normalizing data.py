import os
from gnlse_dataset import GnlseDataset
from tqdm.auto import tqdm
import gnlse

# Data path in string format
data_path = ['gamma_data/','beta_data/','power_data/','time_data/', 'new_data']


max_ = 0

for dir_ in data_path:
    for folder in ['train', 'test']:
        path = os.path.join(dir_, folder)
        data = GnlseDataset(data_dir=path)
    
        print(folder)
        for i in tqdm(range(len(data))):
            max_ = data[i][0][0].max() if data[i][0][0].max() > max_ else max_

 
max_ = float(max_)
print(f'Max: {max_}')

 
for dir_ in data_path:
    for folder in ['train', 'test']:
        path = os.path.join(dir_, folder)
        data = GnlseDataset(data_dir=path)
    
        print(path)
        for i in tqdm(range(len(data))):
            temp = data[i][0][0]
            half = int(len(temp)/2)
            sol_dict = {
                'time_domain': temp[half:].numpy(),
                'frequency_domain': temp[:half].numpy() }
            sol_dict['frequency_domain'] = sol_dict['frequency_domain'] * 100 / max_
        
            gnlse.write_mat(sol_dict, os.path.join(path, data.file_names[i]))
        
        
