'''
File name: train_baseline.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
from data import get_data
from utils.utils import test
import sys

mtypes = {
    'baseline': 'c',
    'duration': 'cd',
    'melody': 'cm',
    'all': 'cmd'
}

if __name__ == '__main__':

    for arg in sys.argv:

        if arg in mtypes:

            # set base parameters
            vocab_size = 157
            model_path = 'models/' + arg + '.pth'
            
            dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size, mtypes[arg])
            test(model_path, dataloader_test)




