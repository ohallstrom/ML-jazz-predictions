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
import torch
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

def grid_search(dataloader_train, dataloader_val, save_pth, lr_l,weight_decay_l,hidden_size_l,drop_l, input_size, vocab_size):

  accuracy = np.zeros(((len(lr_l)),len(weight_decay_l),len(hidden_size_l),len(drop_l)))
  accuracy_train1 = np.zeros(((len(lr_l)),len(weight_decay_l),len(hidden_size_l),len(drop_l)))
  current_max_val = 0
  for ind_lr, lr in enumerate(lr_l):
    for ind_weight_decay, weight_decay in enumerate(weight_decay_l):
        for ind_hidden_size, hidden_size in enumerate(hidden_size_l):
            for ind_drop, drop in enumerate(drop_l):
                logging.info("Current hyperparams idxs: " + "lr: " + str(ind_lr) + " weight_decay : " + str(ind_weight_decay)+ " hidden_size: " + str(ind_hidden_size) + " drop_out: " + str(ind_drop))
                model = ChordSequenceModel(input_size, vocab_size, hidden_size, drop)
                _, accuracies, _, accuracy_train, max_val, max_val_model = train(model, dataloader_train, dataloader_val, save_pth, lr, weight_decay)
                if max_val > current_max_val:
                    logging.info("Current hyperparams were found optimal")
                    current_max_val = max_val
                    torch.save(max_val_model, save_pth)
                accuracy[ind_lr,ind_weight_decay,ind_hidden_size,ind_drop] = accuracies[-1]
                accuracy_train1[ind_lr,ind_weight_decay,ind_hidden_size,ind_drop] = accuracy_train[-1]

  return accuracy, accuracy_train1


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
            hidden_size_l = [300,350,400]
            lr_l = [1e-2]
            weight_decay_l = [1e-5,1e-4,1e-3]
            drop_l = [0.1,0.2,0.3]
            vocab_size = 157

            model_path = 'models/' + arg + '.pth'
            
            dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size, mtypes[arg])
            
            accuracy_grid_val, accuracy_grid_train = grid_search(dataloader_train, dataloader_val, model_path, lr_l, weight_decay_l, hidden_size_l, drop_l, input_size, vocab_size)
            
            print(accuracy_grid_val)
            print(accuracy_grid_train)
            #model = ChordSequenceModel(input_size, vocab_size, hidden_size)
            #train(model, dataloader_train, dataloader_val, model_path)


