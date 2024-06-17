from run import CDCNN
from data import DataProcess

data_hadler = DataProcess('/content/gdrive/MyDrive/Causalidad_data/bd_prueba_2.xlsx',
                          '/content/gdrive/MyDrive/Causalidad_data/eval_prueba_2.csv')
data, eval_dict = data_hadler.prepare_data()

dilatation_c = 2
valp = 0.2
kernel_size = 4  
levels = 1
nrepochs = 500
valepoc = 200
learningrate = 0.001
optimizername = "Adam"
dilation_c = 1
loginterval = 500
seed=1234
cuda=True
significance=0.99
plot=True
lags=5
panel=False

sig_2 = 0.7
sig_3 = 0.58
sig_4 = 0.

signi_2 = 0.6
val_p = 0.025

main_model = CDCNN(signi_2, sig_2, sig_3, sig_4, valepoc, levels, dilatation_c, nrepochs, 
                 kernel_size, loginterval, learningrate, valp, lags,
                 significance, seed, optimizername, cuda, plot)

main_model.main(data, evaluation=True, test=eval_dict)

# Evaluate
rial = {
        1: {(0, 4): [4], (0, 1): [1], (0, 2): [2, 6], (1, 4): [1], (1, 2): [1], (1, 0): [3], 
            (1, 3): [2], (2, 2): [1], (2, 3): [1], (2, 0): [5], (2, 1): [6], (2, 4): [9], 
            (3, 3): [1], (3, 0): [2], (3, 2): [2], (3, 1): [3], (3, 4): [6], (4, 4): [1], 
            (4, 0): [1], (4, 1): [1], (4, 2): [2], (4, 3): [3], (5, 4): [4], (5, 3): [3], 
            (5, 1): [3], (5, 0): [6, 5, 5], (5, 2): [4, 5, 6]}, 
        2: {(0, 4): [1], (0, 1): [2], (2, 0): [1], (2, 1): [3], (2, 4): [2],
            (3, 2): [5], (3, 0): [6]}, 3: {(2, 0): [2]}
    }
readgt = {
        1: {0: [4, 1], 1: [4, 2, 0], 2: [2, 3, 0], 3: [3, 0, 2], 4: [4, 0, 1, 2], 5: [4, 3, 1]}, 
        2: {0: [4, 1], 1: [], 2: [0], 3: [2], 4: [], 5: []}, 3: {0: [], 1: [], 2: [0], 3: [], 4: [], 5: []}
    }
pred = {
        1: {0: [4, 1, 2], 
            1: [4, 2, 0, 3], 
            2: [2, 3, 0, 1, 4], 
            3: [3, 0, 2, 1, 4], 
            4: [4, 0, 1, 2, 3], 
            5: [4, 3, 1, 0, 2]}, 
        2: {0: [4, 1], 
            1: [], 
            2: [0, 1, 4], 
            3: [2, 0], 
            4: [], 
            5: []}, 
        3: {0: [], 
            1: [], 
            2: [0], 
            3: [], 
            4: [], 
            5: []}
    }

