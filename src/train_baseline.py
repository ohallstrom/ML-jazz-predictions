'''
File name: train_baseline.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import logging
import os
from datetime import datetime
from torch import optim
from utils.utils import get_loss, get_accuracy
from models import BaselineChordSequenceModel
from data import get_baseline_dataloader

def train(model, dataloader):
    '''
    Trains the baseline model
    during 100 epochs, the results
    are saved into a log file.
    :param model: LSTM-model to train
    :param dataloader: dataloader containing training data
    '''
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience=5)

    for epoch in range(100):
        accuracy=0
        avg_loss=0
        count=0
        for batch_idx, batch in enumerate(dataloader):
            count+=1
            inputs = batch["input"].float()
            lengths = batch["length"]
            targets = batch["target"][:, :max(lengths)]  # Pad_packed cuts off at max_length

            optimizer.zero_grad()
            outputs = model(inputs, lengths)

            loss = get_loss(outputs, targets)
            lossv = loss.item()
            acc = get_accuracy(outputs, targets)
            accv = acc.item()

            accuracy+=accv
            avg_loss+=lossv

            loss.backward()
            optimizer.step()
            

        avg_loss/=count
        accuracy/=count  
        scheduler.step(accv)

        logging.info("EPOCH " + str(epoch) + " Loss: " + str(avg_loss) + " Acc: " + str(accuracy))

if __name__ == '__main__':
    # set logging path and settings
    os.makedirs(os.path.join("baseline_training_logs"), exist_ok=True)
    log_path = os.path.join("baseline_training_logs", datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    numeric_level = getattr(logging, "INFO", None)
    logging.basicConfig(
        level = numeric_level,
        format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers = [logging.FileHandler(log_path),logging.StreamHandler()]
        )

    # set base parameters
    vocab_size = 24
    lstm_hidden_size = 48
    
    dataloader, classes_size = get_baseline_dataloader(vocab_size)

    model = BaselineChordSequenceModel(vocab_size, lstm_hidden_size, classes_size)

    train(model, dataloader)


