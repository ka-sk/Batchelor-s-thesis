import gnlse
import numpy as np
import os
from shutil import move
from random import sample
from time import time
from math import pi
#import functions_visualization as vfun


def generate_one_pulse(path=None,
                       loss=0,
                       nonlinearity=0.11,
                       betas=[0],
                       peak_power=100,
                       duration=0.1,
                       resolution=2 ** 14,
                       time_window=15,
                       z_saves=200,
                       wavelength=835,
                       wavelength_boundary=400,
                       fiber_length=1,
                       ):

    # Stworzenie nazwy pliku
    file_name_ = file_name(b=betas, g=nonlinearity, pp=peak_power, t=duration)

    # Sprawdzenie czy plik już istnieje, jeżeli tak to należy ominąć obliczanie
    if file_name_ in os.listdir(path):
        print(f'File {file_name_} already exists.')
        return

    setup = gnlse.GNLSESetup()

    # Ustawienie parametrów
    setup.resolution = resolution
    setup.time_window = time_window  # ps
    setup.z_saves = z_saves
    setup.wavelength = wavelength  # nm
    setup.fiber_length = fiber_length  # m
    setup.nonlinearity = nonlinearity  # 1/W/m
    setup.raman_model = gnlse.raman_blowwood
    setup.self_steepening = True

    # Ustawienie modelu dyspersji
    setup.dispersion_model = gnlse.DispersionFiberFromTaylor(loss, betas)

    # Model wiązki wejściowej lasera
    pulse_model = gnlse.GaussianEnvelope(peak_power, duration)

    print('%s...' % pulse_model.name)

    # Użycie biblioteki gnlse do rozwiązania równania
    setup.pulse_model = pulse_model
    solver = gnlse.GNLSE(setup)
    solution = solver.run()

    # Wyodrębnienie kluczowej części wyniku
    sol_dict = {
        'time_domain': abs(solution.At[-1, :])**2,
        'frequency_domain': abs(solution.AW[-1, :])**2}

    # prędkość światła
    c = 299792458e-3  # nm/ps

    # zamiana częstotliwości na długość fali
    sol_dict['frequency_domain'] = np.array([2 * pi * c / i for i in sol_dict['frequency_domain']])

    # wyodrębnienie siatki pliku z zamianą częstotliwości na długość fali
    grid_dict = dict(time=solution.t, frequency=np.array([2*pi*c/i for i in solution.W]))

    # określenie indexów granicznych
    min_idx, max_idx = min_max_idx(max(wavelength - wavelength_boundary, 1), wavelength + wavelength_boundary, grid_dict['frequency'])
    min_idx, max_idx = min(min_idx, max_idx), max(min_idx, max_idx)
    
    # flitrowanie domeny czasowej: obliczenie indexu środkowego
    time_idx = int((max_idx - min_idx) / 2)

    # zamiana połów prawej i lewej ze względu na metody obliczeniowe
    half_idx = int(len(sol_dict['frequency_domain'])/2)
    sol_dict['frequency_domain'] = np.concatenate((sol_dict['frequency_domain'][half_idx:],
                                                   sol_dict['frequency_domain'][:half_idx]))
    # filtrowanie domeny czasowej i częstotliwościowej
    sol_dict['frequency_domain'] = sol_dict['frequency_domain'][min_idx: max_idx]
    sol_dict['time_domain'] = sol_dict['time_domain'][half_idx - time_idx: half_idx + time_idx]

    # jeżeli ścieżka została podana to plik zostanie zapisany
    if path is not None:
        # zapis pliku
        try:
            gnlse.write_mat(sol_dict, path + file_name_)
        except Exception:
            # wyjątek pojawia się kiedy plik już jest otwarty
            print(Exception)
            print(f'Is file already saved?: {file_name_ in os.listdir(path)}')

        # sprawdzanie czy plik z siatką 'grid.mat' istnieje
        if not os.path.exists(f'{path}/grid.mat'):

            # filtrowanie siatki
            grid_dict['time'] = grid_dict['time'][half_idx - time_idx: half_idx + time_idx]
            grid_dict['frequency'] = grid_dict['frequency'][min_idx: max_idx]

            # zapis siatki
            gnlse.write_mat(grid_dict, path + 'grid.mat')

    return sol_dict


def multi_data_generation(path='',
                          nonlinearity=(0.11, 0.11, 1),
                          beta2=(20e3, 20e3, 1),
                          peak_power=(100, 100, 1),
                          duration=(0.1, 0.1, 1),
                          resolution=2 ** 14,
                          time_window=12.5,
                          z_saves=100,
                          wavelength=835,
                          wavelength_boundary=400,
                          fiber_length=1):
    # sprawdzanie istnienia podanej ścieżki
    path = check_folder(path)

    # start stopera
    start = time()

    # zmienna zawierająca liczbę wszystkich iteracji
    nb_of_all = nonlinearity[2] * beta2[2] * peak_power[2] * duration[2]
    done = 0

    # iteracja po wszystkich punktach
    for nonlinearity_ in np.linspace(nonlinearity[0], nonlinearity[1], nonlinearity[2]):
        for beta2_ in np.linspace(beta2[0], beta2[1], beta2[2]):
            for peak_power_ in np.linspace(peak_power[0], peak_power[1], peak_power[2]):
                for duration_ in np.linspace(duration[0], duration[1], duration[2]):

                    # wywołanie funkcji do obliczanie rozwiązania
                    generate_one_pulse(nonlinearity=nonlinearity_,
                                       betas=[beta2_],
                                       peak_power=peak_power_,
                                       duration=duration_,
                                       resolution=resolution,
                                       time_window=time_window,
                                       z_saves=z_saves,
                                       wavelength=wavelength,
                                       wavelength_boundary=wavelength_boundary,
                                       fiber_length=fiber_length,
                                       path=path
                                       )
                    done += 1

                    # wyświetlanie postępu co 50 iteracji
                    if done % 50 == 0:
                        end = time()
                        print(f'Done {done}/{nb_of_all}')
                        print(f'Time: {end - start}')

    print(done)


def clear_dir(path=''):
    '''
    Functions that clears given directory except train and test directories
    Created to easily overwrite wrongly generated data
    :param path: folder path is which files are to be deleted
    :return: None
    '''
    # if path has no '/' at the end add it
    if path[-1] != '/':
        path += '/'

    # create list of all file in the directory
    file_list = os.listdir(path)

    # directories 'train' and 'test' are to be spared
    if 'test' in file_list:
        file_list.remove('test')
    if 'train' in file_list:
        file_list.remove('train')

    # remove all files in the list
    for file_name in file_list:
        os.remove(f'{path}{file_name}')
    print(f'Path: "{path}" cleared')


def train_test_move(path='', train_size=None, test_size=None):
    # Jeżeli żaden z warunków nie jest podany używana jest wartość test_size = 0.25
    if test_size is None and train_size is None:
        test_size = 0.25

    # Jeżeli tylko rozmiar zbioru testowego jest podany program może kontynuować obliczenia bez modyfikacji wartości
    elif train_size is None and test_size is not None:
        pass
     # Jeżeli podany jest tylko train_size obliczamy test_size
    elif train_size is not None and test_size is None:
        test_size = 1 - train_size

    # Jeżeli obie wartości są podane jednak ich suma jest 1 to program jest kontynuowany
    elif train_size + test_size == 1:
        pass

    # W przeciwnym przypadku należy przerwać obliczenia
    else:
        raise ValueError()
        
    print('Preparing data')

    # sprawdzenie folderu docelowego
    check_folder(path)

    # zebranie listy wszystkich plików w folderze docelowym
    file_list = os.listdir(path)

    # usunięcie z listy pliku z siatką 'grid.mat' oraz nazw folderów docelowych
    [file_list.remove(i) for i in ['train', 'test', 'grid.mat'] if i in file_list]

    # obliczenie liczby elementów zbioru testowego
    test_size = int(len(file_list) * test_size)

    # zebranie odpowiedniej liczby losowych elementów listy do zbioru testowego
    test_list = sample(file_list, test_size)
    # zebranie wszystkich pozostałych elementów do zbioru treningowego
    train_list = [i for i in file_list if i not in test_list]

    # Iteracja po liście treningowej
    for file_name in train_list:
        move(path + file_name, path + 'train/' + file_name)

    print(f'Moved {len(train_list)} files into train directory')

    # Iteracja po liście testowej
    for file_name in test_list:
        move(path + file_name, path + 'test/' + file_name)

    print(f'Moved {test_size} files into test directory')


def check_folder(path: str):
    '''
    Checks if given path exists and has 'test' and 'train' directories in it
    :param path: directory path
    :return: path (the same or with added '/')
    '''
    # check if path has '/' at the end
    if path[-1] != '/' and path != '':
        path += '/'

    # create directory if it doesn't exist
    if path != '' and not os.path.isdir(path):
        os.mkdir(path)

    if path != '' and not os.path.isdir(path + '/test'):
        os.mkdir(path + '/test')

    if path != '' and not os.path.isdir(path + '/train'):
        os.mkdir(path + '/train')

    return path


def file_name(mat=True, **kwargs):
    '''
    Creates file name that contains all the given information in it.
    There are a couple of rules that it follows:
    - every next variable is split with '__'
    - every value must be int, float or list
    - every key must be a string
    :param mat: if True adds '.mat' in the end of the string
    :param kwargs: parameters to be included in file name
    :return: string of a file name described above in .mat format
    '''

    text = ''
    for key, value in kwargs.items():
        # if value is iterable
        try:
            for i in value:
                # round every value
                i = round(i, 5)
                # replacing every '.' to '_' in every value
                i = str(i).replace('.', '_')

                # adding every key and i to file name
                text += f'{key}{i}__'
        except TypeError:
            # round every value
            value = round(value, 5)
            # replace every '.' to '_' in gamma
            value = str(value).replace('.', '_')

            # including gamma in name divided by __
            text += f'{key}{value}__'

    # deleting last "__" sequence
    text = text.rstrip('__')

    if mat:
        # adding .mat file format
        text += '.mat'
    return text


def min_max_idx(min_val: float, max_val: float, values: list):
    min_idx = 0
    max_idx = len(values) - 1

    # finding the closes value to min_val and max_val
    for idx in range(1, len(values)):
        if values[idx] * values[idx-1] < 0:
            min_idx = idx
        elif values[idx] > 0:
            # checking if min and max are found
            if values[idx] < min_val <= values[idx-1]:
                min_idx = idx - 1
            elif values[idx] < max_val <= values[idx - 1]:
                max_idx = idx
        if values[-1] > min_val:
            min_idx = len(values)-1
    return min_idx, max_idx


def check_files(path='',
                nonlinearity=(0.11, 0.11, 1),
                beta2=(20e3, 20e3, 1),
                peak_power=(100, 100, 1),
                duration=(0.1, 0.1, 1),
                create='ask',
                resolution=2 ** 14,
                time_window=12.5,
                z_saves=100,
                wavelength=835,
                wavelength_boundary=400,
                fiber_length=1):

    directory_list = os.listdir(path)
    lacking_list = []

    def create_data_loop(list):
        done = 0
        for beta2_, nonlinearity_, peak_power_, duration_ in lacking_list:
            generate_one_pulse(nonlinearity=nonlinearity_,
                               betas=[beta2_],
                               peak_power=peak_power_,
                               duration=duration_,
                               resolution=resolution,
                               time_window=time_window,
                               z_saves=z_saves,
                               wavelength=wavelength,
                               wavelength_boundary=wavelength_boundary,
                               fiber_length=fiber_length,
                               path=path
                               )
            done += 1
            if done % 50 == 0:
                print(f'Done: {done}/{len(lacking_list)}')
        print(f'Created all {len(lacking_list)} files')
        return None

    for nonlinearity_ in np.linspace(nonlinearity[0], nonlinearity[1], nonlinearity[2]):
        for beta2_ in np.linspace(beta2[0], beta2[1], beta2[2]):
            for peak_power_ in np.linspace(peak_power[0], peak_power[1], peak_power[2]):
                for duration_ in np.linspace(duration[0], duration[1], duration[2]):
                    # create file name to check if it already exists
                    file_name_ = file_name(b=beta2_,
                                           g=nonlinearity_,
                                           pp=peak_power_,
                                           t=duration_)

                    # if file doesn't exist add it to the list
                    if file_name_ not in directory_list and file_name_ not in lacking_list:
                        lacking_list.append([beta2_, nonlinearity_, peak_power_, duration_])

    if len(lacking_list) == 0:
        print(f'All files exist at path {path}')
        return []
    else:
        print(f'###\nLacking {len(lacking_list)} files\n###')
    if create == 'ask':
        answer = input('Do you want to compute lacking files? (Y/N)')
        if answer == 'N':
            return lacking_list
        elif answer == 'Y':
            create_data_loop(lacking_list)
            return []
    elif create == "N":
        return lacking_list
    elif create == "Y":
        create_data_loop(lacking_list)
        return []
