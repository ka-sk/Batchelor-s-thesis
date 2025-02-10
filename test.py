import os
from gnlse_dataset import GnlseDataset

data_path = 'Sample data/'
path = os.path.join(data_path, 'Gamma')

a = GnlseDataset(path)

print(len(a[0][0][0]))

