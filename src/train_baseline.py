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
from models import ChordSequenceModel
from data import get_data
import torch
import numpy as np
from sklearn.metrics import classification_report

def train(model, dataloader_train, dataloader_val, save_pth):
    '''
    Trains the baseline model
    during 30 epochs, the results
    are saved into a log file.
    :param model: LSTM-model to train
    :param dataloader: dataloader containing training data
    '''
    model=model.to('cuda')
    losses = []
    accuracies = []
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience=10,verbose=True)

    max_val=0
    for epoch in range(30):
        accuracy=0
        avg_loss=0
        accuracy_val=0
        avg_loss_val=0
        count=0
        for batch_idx, batch in enumerate(dataloader_train):
            count+=1
            inputs = batch["input"].float().to('cuda')
            lengths = batch["length"]
            # print(lengths)
            targets = batch["target"][:, :max(lengths)].to('cuda')  # Pad_packed cuts off at max_length

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            # print(inputs.shape,targets.shape, outputs.shape)
            loss = get_loss(outputs, targets)
            lossv=loss.item()
            acc = get_accuracy(outputs, targets)
            accv = acc.item()

            accuracy+=accv
            avg_loss+=lossv

            loss.backward()
            optimizer.step()
        

        avg_loss/=count
        accuracy/=count  

        count=0

        for batch_idx, batch in enumerate(dataloader_val):
            count+=1
            inputs = batch["input"].float().to('cuda')
            lengths = batch["length"]
            # print(lengths)
            targets = batch["target"][:, :max(lengths)].to('cuda')  # Pad_packed cuts off at max_length

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            # print(inputs.shape,targets.shape, outputs.shape)
            loss = get_loss(outputs, targets)
            lossv=loss.item()
            acc = get_accuracy(outputs, targets)
            accv = acc.item()

            accuracy_val+=accv
            avg_loss_val+=lossv
        

        avg_loss_val/=count
        accuracy_val/=count  


        scheduler.step(accuracy_val)
        #check for max accuracy
        if accuracy_val > max_val:
            torch.save(model, save_pth)
        losses.append(avg_loss)
        accuracies.append(accuracy_val)
        # print("EPOCH", epoch, " Loss:", avg_loss, " Acc:", accuracy,  " Val_Loss:", avg_loss_val, "Val_Acc:", accuracy_val)
        logging.info("EPOCH: " + str(epoch) + " Loss: "+ str(avg_loss)+ " Acc: " + str(accuracy) + " Val_Loss: " + str(avg_loss_val) + " Val_Acc: " + str(accuracy_val))

        #!TODO save losses and accuracies or plot

def test(model_pth, dataloader_test):
    targs=[]
    preds=[]
    accuracy_test=0
    avg_loss_test=0
    count=0
    for batch_idx, batch in enumerate(dataloader_test):
        count+=1
        inputs = batch["input"].float().to('cuda')
        lengths = batch["length"]
        # print(lengths)
        targets = batch["target"][:, :max(lengths)].to('cuda')  # Pad_packed cuts off at max_length

        outputs = model(inputs, lengths)

        loss = get_loss(outputs, targets)
        lossv=loss.item()
        acc = get_accuracy(outputs, targets)
        accv = acc.item()

        targs.append(targets.detach().cpu().numpy())
        preds.append(outputs.detach().cpu().numpy())

        accuracy_test+=accv
        avg_loss_test+=lossv


    avg_loss_test/=count
    accuracy_test/=count  


    print("Test Loss:", avg_loss_test, "Test Acc:", accuracy_test)

    preds2=[np.array(i).reshape(-1,157) for i in preds]
    preds_a=np.vstack(preds2)
    preds_all=np.argmax(preds_a, axis=1)
    targs2=[np.array(i) for i in targs]
    targs2=[i.reshape(-1) for i in targs2]
    targs_all=np.concatenate(targs2)
    mask = ~(targs_all==-1)
    print(classification_report(targs_all[mask], preds_all[mask]))
    # !TODO Add classes to classification report


if __name__ == '__main__':
    # set logging path and settings
    # !TODO Separate logging for models
    os.makedirs(os.path.join("baseline_training_logs"), exist_ok=True)
    log_path = os.path.join("baseline_training_logs", datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    numeric_level = getattr(logging, "INFO", None)
    logging.basicConfig(
        level = numeric_level,
        format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers = [logging.FileHandler(log_path),logging.StreamHandler()]
        )

    # set base parameters
    input_size = 24
    hidden_size = 300
    vocab_size = 157
    
    dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size,'c')
    model = ChordSequenceModel(input_size, vocab_size, hidden_size)
    train(model, dataloader_train, dataloader_val,"./models/model_baseline.pth")
    # !TODO test
    test("./models/model_baseline.pth", dataloader_test)


    input_size = 25
    dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size,'cd')
    model = ChordSequenceModel(input_size, vocab_size, hidden_size)
    train(model, dataloader_train, dataloader_val,"./models/model_duration.pth")
    test("./models/model_duration.pth", dataloader_test)


    input_size=36
    dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size,'cm')
    model = ChordSequenceModel(input_size, vocab_size, hidden_size)
    train(model, dataloader_train, dataloader_val,"./models/model_melody.pth")
    test("./models/model_melody.pth", dataloader_test)


    input_size = 37
    dataloader_train, dataloader_val, dataloader_test = get_data(vocab_size,'cmd')
    model = ChordSequenceModel(input_size, vocab_size, hidden_size)
    train(model, dataloader_train, dataloader_val,"./models/model_all.pth")
    test("./models/model_all.pth", dataloader_test)


