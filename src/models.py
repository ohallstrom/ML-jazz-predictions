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


class ChordSequenceModel(nn.Module):
	def __init__(self, input_size,  vocab_size, lstm_hidden_size, dropout_param):
		super().__init__()
		self.vocab_size = vocab_size 
		self.input_size = input_size
		self.lstm_hidden_size = lstm_hidden_size

		self.lstm = nn.LSTM(
			self.input_size,
			self.lstm_hidden_size,
			num_layers=1,
			bidirectional=False,
			batch_first=True,
		)
		
		self.dropout_size = dropout_param
		self.output = nn.Linear(self.lstm_hidden_size, self.vocab_size)

	def forward(self, inputs, lengths):  
		# Shape = batch, sequence, vocab_size
		packed = pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
		lstm_out_packed, hidden_out = self.lstm(packed)
		lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
		relu1 = F.relu(lstm_out)

		# Shape = batch, sequence, lstm_hidden_size
		drop = self.dropout(relu1)
		output = self.output(drop)

		# Shape = batch, sequence, vocab_size
		return output  
