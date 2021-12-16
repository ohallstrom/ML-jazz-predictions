'''
File name: data.py
Author: Andrew, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import torch
import random
from ast import literal_eval
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.chord_manipulations import create_chord_type_dict, add_chord_cols



class VLDataset(Dataset):
  def __init__(self, sequences, targs, vocab_size):
    self.sequences = sequences
    self.targs = targs
    self.vocab_size = vocab_size
    self.max_length = max(map(len, self.sequences))  # Add max length

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, i):
    sequence = self.sequences[i]
    sequence_length = len(self.sequences[i])
    # print(sequence_length, self.max_length)
    # Pad input with 0 up to max length
    input = np.zeros((self.max_length, self.vocab_size))
    input[:sequence_length,:] = sequence 

    #Pad target with some INVALID value (-1)
    target = np.ones(self.max_length - 1) * -1
    #target[:sequence_length - 1] = sequence[1:]
    # target = np.full((self.max_length-1,), -1)
    
    # print(self.targs[i])
    if sequence_length > 1:
      target[:sequence_length-1] = self.targs[i][1:sequence_length]

    return {
        "input": torch.tensor(input[:-1]),
        "target": torch.tensor(target).long(),
        "length": sequence_length - 1,  # Return the length
    }


def get_data(vocab_size,mtype):
    '''
    Creates the dataloaders containing preprocessed data from the dataset file
    :return: (DataLoader_train, DataLoader_validation, DataLoader_test)
    '''
    data=pd.read_csv('../data/data_preprocessed.csv')
    data['chord_enc']=data['chord_enc'].apply(lambda x: literal_eval(x))
    data['pitch_bow']=data['pitch_bow'].apply(lambda x: literal_eval(x))
    data['duration']=data['onset'] - data['onset'].shift(1)
    data.dropna(subset = ['duration'], inplace=True)
    data['duration'] = data['duration'].apply(lambda x: [x])
    temp=data.copy()
    vocab_size=24
    if mtype=='c':
        temp['data']=data['chord_enc']
    elif mtype=='cd':
        temp['data']=data['chord_enc']+temp['duration']
        vocab_size+=1
    elif mtype=='cm':
        temp['data']=data['chord_enc']+temp['pitch_bow']
        vocab_size+=12
    elif mtype=='cmd':
        temp['data']=data['chord_enc']+temp['pitch_bow']+temp['duration']
        vocab_size+=13

    mel_ids = temp['melid'].unique()
    targets=[]
    sequences = []
    for mel_id in mel_ids:
      # get series with all chords from song with mel_id
      chords = temp[temp['melid'] == mel_id]['data']
      chords_2d_np_array = np.array(chords.tolist())
      sequences.append(np.array(chords_2d_np_array))
      targs=temp[temp['melid'] == mel_id]['chord_class']
      targets.append(np.array(targs.tolist()))

    sequences_clean=[]
    targets_clean=[]
    for i in range(len(sequences)):
      if len(sequences[i])>2:
        sequences_clean.append(sequences[i])
        targets_clean.append(targets[i])

    c = list(zip(sequences_clean, targets_clean))
    random.Random(4).shuffle(c)

    sequences_clean, targets_clean = zip(*c)

    l=len(sequences_clean)
    sequences_clean_train=sequences_clean[:int(0.8*l)]
    sequences_clean_val=sequences_clean[int(0.8*l):int(0.9*l)]
    sequences_clean_test=sequences_clean[int(0.9*l):]
    targets_clean_train=targets_clean[:int(0.8*l)]
    targets_clean_val=targets_clean[int(0.8*l):int(0.9*l)]
    targets_clean_test=targets_clean[int(0.9*l):]

    vl_dataset_train = VLDataset(sequences_clean_train, targets_clean_train ,vocab_size)
    vl_dataset_val = VLDataset(sequences_clean_val, targets_clean_val ,vocab_size)
    vl_dataset_test = VLDataset(sequences_clean_test, targets_clean_test ,vocab_size)

    vl_dl_train = DataLoader(vl_dataset_train, batch_size=10, shuffle=True) 
    vl_dl_val = DataLoader(vl_dataset_val, batch_size=10, shuffle=False)
    vl_dl_test = DataLoader(vl_dataset_test, batch_size=10, shuffle=False)

    return vl_dl_train, vl_dl_val, vl_dl_test