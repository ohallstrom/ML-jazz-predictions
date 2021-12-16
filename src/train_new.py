'''
File name: train.py
Author: Anmol, Oskar, Nikunj
Date created: 16/12/2021
Date last modified: 16/12/2021
Python Version: 3.8
'''
import logging
import os
import sys
from datetime import datetime
from models import ChordSequenceModel
from data import get_data
from utils.utils_new import train
import numpy as np

input_sizes = {
    'baseline': 24,
    'duration': 25,
    'melody': 36,
    'all': 37
}
mtypes = {
    'baseline': 'c',
    'duration': 'cd',
    'melody': 'cm',
    'all': 'cmd'
}

def grid_search(dataloader_train, dataloader_val, save_pth, lr_l,weight_decay_l,hidden_size_l, input_size, vocab_size):
  """ Does the grid search over the params learning rate (lr), weirght decay and hidden size """
  accuracy = np.zeros((len(lr_l)),len(weight_decay_l),len(hidden_size_l))
  for ind_lr, lr in enumerate(lr_l):
    for ind_weight_decay, weight_decay in enumerate(weight_decay_l):
        for ind_hidden_size, hidden_size in enumerate(hidden_size_l):
          model = ChordSequenceModel(input_size, vocab_size, hidden_size)
          _, accuracies = train(model, dataloader_train, dataloader_val, save_pth, lr, weight_decay)

          accuracy[ind_lr,ind_weight_decay,ind_hidden_size] = accuracies[-1]

    return accuracy


if __name__ == '__main__':

    for arg in sys.argv:

        if arg in input_sizes:
            # set logging path and settings
            log_dir = 'logs' + '/' + arg 
            os.makedirs(os.path.join(log_dir), exist_ok=True)
            log_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
            numeric_level = getattr(logging, "INFO", None)
            logging.basicConfig(
                level = numeric_level,
                format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers = [logging.FileHandler(log_path),logging.StreamHandler()]
                )

            # set base parameters
            input_size = input_sizes[arg]
            hidden_size_l = np.array([300,350])
            lr_l = np.array([1e-6,1e-4])
            weight_decay_l = np.array([0,1])
            vocab_size = 157

            model_path = 'models/' + arg + '.pth'
            
            dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size, mtypes[arg])
            
            accuracy_grid = grid_search(dataloader_train, dataloader_val, model_path, lr_l, weight_decay_l, hidden_size_l, input_size, vocab_size)
            
            #model = ChordSequenceModel(input_size, vocab_size, hidden_size)
            #train(model, dataloader_train, dataloader_val, model_path)



