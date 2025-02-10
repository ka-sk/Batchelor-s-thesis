import functions_generate_data as gdfun

#DONE
# ranges and number of points that are generated
PATH = 'new_data/gen7'
GAMMA = (0, 0.055, 10)  # 1/W/m
BETA2 = (0, 20e-3, 10) #
POWER = (500, 1000, 10)  # W
TIME = (0.05, 1, 10)  # ps
LENGTH = 1  # m

# clearing path in case of old files
#gdfun.clear_dir(PATH)
#gdfun.clear_dir(os.path.join(PATH, 'test'))
#gdfun.clear_dir(os.path.join(PATH, 'train'))

# creating data
gdfun.multi_data_generation(nonlinearity=GAMMA,
                            beta2=BETA2,
                            path=PATH,
                            duration=TIME,
                            peak_power=POWER,
                            wavelength_boundary=400
                            )
# moving data to train and test files in random order
#gdfun.train_test_move(PATH)

'''
profil czasowy to kwadrat modułu At


okno czasowe to samo jak w test dudley
z_saves mniej
długość 1m
gamma 0 do 0.11
beta2 -20e3 do 20e3
beta3
P0 0 do 1000 W 
T0 1ps 0.05 ps max
lambda 835 nm
'''

"""
# ranges and number of points that are generated
PATH = 'sample_data/'
GAMMA = (0, 0.11, 101)  # 1/W/m
BETA2 = (-20e-3, 20e-3, 100) #
POWER = (0, 1000, 100)  # W
TIME = (0.05, 1, 100)  # ps
LENGTH = 1  # m
"""
