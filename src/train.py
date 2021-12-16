'''
File name: train_baseline.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import logging
import os
import sys
from datetime import datetime
from models import ChordSequenceModel
from data import get_data
from utils.utils import train

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
            hidden_size = 300
            vocab_size = 157

            model_path = 'models/' + arg + '.pth'
            
            dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size, mtypes[arg])
            model = ChordSequenceModel(input_size, vocab_size, hidden_size)
            train(model, dataloader_train, dataloader_val, model_path)




