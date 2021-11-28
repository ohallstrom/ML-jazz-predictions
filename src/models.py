'''
File name: models.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class BaselineChordSequenceModel(nn.Module):
    def __init__(self, vocab_size, lstm_hidden_size, classes_size):
        super().__init__()
        self.vocab_size = vocab_size 
        self.lstm_hidden_size = lstm_hidden_size
        self.classes_size = classes_size

        self.lstm = nn.LSTM(
            self.vocab_size,
            self.lstm_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.output = nn.Linear(self.lstm_hidden_size, self.classes_size)

    def forward(self, inputs, lengths):  

        packed = pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)
        output = self.output(relu1)

        return output