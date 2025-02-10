import functions_generate_data as gdfun
import os


# wartości graniczne i liczby punktów dla danych wielkości fizycznych
PATH = 'time_data/'
GAMMA = (0.055, 0.055, 1)  # 1/W/m
BETA2 = (10e-3, 10e-3, 1) #
POWER = (500, 500, 1)  # W
TIME = (0.05, 1, 5000)  # ps CZAS POŁÓWKOWY
LENGTH = 1  # m


# czyszczenie folderu docelowego dla uniknięcia obecności błędnych plików
#gdfun.clear_dir(PATH)
#gdfun.clear_dir(os.path.join(PATH, 'test'))
#gdfun.clear_dir(os.path.join(PATH, 'train'))

# wywołanie funkcji do generacji danych
gdfun.multi_data_generation(nonlinearity=GAMMA,
                            beta2=BETA2,
                            path=PATH,
                            duration=TIME,
                            peak_power=POWER,
                            wavelength_boundary=400
                            )


# sprawdzenie czy wszystkie pliki zostały zapisane
gdfun.check_files(nonlinearity=GAMMA, 
                  beta2=BETA2,
                  path=PATH,
                  duration=TIME,
                  peak_power=POWER,
                  wavelength_boundary=400,
                  create='ask',
                  delete='N'
                  )

# przeniesienie plików do folderów "train" i "test'
gdfun.train_test_move(PATH)


"""
# wartości graniczne i liczby punktów dla danych wielkości fizycznych
PATH = 'new_data/'
GAMMA = (0, 0.11, 20)  # 1/W/m
BETA2 = (-20e-3, 20e-3, 20) #
POWER = (0, 1000, 20)  # W
TIME = (0.05, 1, 10)  # ps
LENGTH = 1  # m

# wartości graniczne i liczby punktów dla danych wielkości fizycznych
PATH = 'beta_data/'
GAMMA = (0.055, 0.055, 1)  # 1/W/m
BETA2 = (-20e-3, 20e-3, 5000) #
POWER = (500, 500, 1)  # W
TIME = (0.1, 0.1, 1)  # ps
LENGTH = 1  # m

(0.01, 0.01, 1) beta2
(0, 0.11, 5000) gamma
1
gamma_data/
(500, 500, 1)  power
(0.1, 0.1, 1) time

# wartości graniczne i liczby punktów dla danych wielkości fizycznych
PATH = 'power_data/'
GAMMA = (0.055, 0.055, 1)  # 1/W/m
BETA2 = (10e-3, 10e-3, 1) #
POWER = (0.1, 1000, 5000)  # W
TIME = (0.1, 0.1, 1)  # ps
LENGTH = 1  # m

# wartości graniczne i liczby punktów dla danych wielkości fizycznych
PATH = 'time_data/'
GAMMA = (0.055, 0.055, 1)  # 1/W/m
BETA2 = (10e-3, 10e-3, 1) #
POWER = (500, 500, 1)  # W
TIME = (0.05, 1, 5000)  # ps
LENGTH = 1  # m
"""
