import os
import gnlse
from torch.utils.data import Dataset
import functions as fun
import numpy as np
import torch
from functions import label_to_data


class GnlseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        '''
        :param data_dir: sciezka folderu plikow danych
        :param transform: funkcje transformujące #TODO jakiego typu one maja byc
        '''

        # Sprawdzenie czy sciezka isnieje, w przeciwnym razie wywołać FileNotFoundError
        if os.path.isdir(data_dir):
            # Inicjalizacja zmiennej file_names jako listy wszystkich plików w folderze
            self.file_names = os.listdir(data_dir)

            # Usuniecie z listy elementow: train, test oraz pliku siatki 'grid.mat'
            if 'grid.mat' in self.file_names: self.file_names.remove('grid.mat')
            if 'train' in self.file_names: self.file_names.remove('train')
            if 'test' in self.file_names: self.file_names.remove('test')

            # wywoanie ValueError jezeli zbior jest pusty
            if len(self.file_names) == 0:
                raise ValueError('Dataset is empty. Check filters and directory')

            # stworzenie listy zawierającej oznaczenia przewidywanych wielkosci
            self.labels = list(label_to_data(self.file_names[1]).keys())

            # przypisanie obiektowi klasy sciezki plikow
            self.data_dir = data_dir

            # przypisanie funkcji transformacji
            self.transform = transform
        else:
            # wywolaj blad jezeli sciezka dostepu nie istnieje
            raise FileNotFoundError(f'Directory {data_dir} not found')

    def __len__(self):
        # liczba plikow w folderze
        return len(self.file_names)

    def __getitem__(self, idx):
        # przygotowanie sceizki otwarcia
        data_path = os.path.join(self.data_dir, self.file_names[idx])

        # otworzenie pliku .mat o podanej nazwie
        data = gnlse.read_mat(data_path)

        # polaczenie obu domen: czestotliwosciowej i czasowej i zamiana na tensor o odpowiednich wymiarach
        data = torch.from_numpy(np.concatenate((data['frequency_domain'], data['time_domain']), axis=0))
        data = torch.unsqueeze(data, 0)
        data = torch.transpose(data, -2, 0)

        # wartości zmiennych wyciagniete z funkcji label_to_data
        quantities = [value for key, value in fun.label_to_data(self.file_names[idx]).items() if key in self.labels]

        # zamiana na tensor
        quantities = torch.tensor(quantities)

        # zwrocenie tensorow
        if self.transform:
            return self.transform(data), quantities
        else:
            return data, quantities
