'''
File name: train.py
Author: Anmol, Oskar
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
from utils.utils import train
import matplotlib.pyplot as plt

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
            hidden_size = 350
            vocab_size = 157

            model_path = 'models/' + arg + '.pth'
            
            dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size, mtypes[arg])
            model = ChordSequenceModel(input_size, vocab_size, hidden_size, 0.2)
            losses, accuracies, losses_val, accuracies_val = train(model, dataloader_train, dataloader_val, model_path)
            
            plt.plot(losses,label = "train loss")
            plt.plot(losses_val,label = "validation loss")
            plt.title("train/validation loss versus number of epochs") # set the plot title
            plt.ylabel("losses") # set the name of the y axis
            plt.xlabel("epochs")
            plt.legend()
            plt.save()
            plt.show()
            
            plt.plot(accuracies,label = "train accuracy")
            plt.plot(accuracies_val,label = "validation accuracy")
            plt.title("train/validation accuracy versus number of epochs") # set the plot title
            plt.ylabel("accuracies") # set the name of the y axis
            plt.xlabel("epochs")
            plt.legend()
            plt.save()
            plt.show()




