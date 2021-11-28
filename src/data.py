'''
File name: data.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import torch
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
from utils.mappings import chord_to_big_hot, multi_hot_to_int

DATA_PATH = "../data/wjazzd.db" 

def get_baseline_dataloader(vocab_size):
	# TODO: add doc-string

	baseline_sequences = get_baseline_sequences()
	dataset = BaselineDataset(baseline_sequences, vocab_size)
	return DataLoader(dataset, batch_size=10, shuffle=False)

def get_baseline_dataframe():
	# TODO: add doc-string
	# load dataframe
	engine = create_engine(f"sqlite:///{DATA_PATH}")
	beats = pd.read_sql("beats", engine)

	# remove empty chord rows
	beats_clean = beats.copy()
	beats_clean['chord'] = beats_clean['chord'].replace(r'^\s*$', np.nan, regex=True)
	beats_clean.dropna(subset = ['chord'], inplace=True)

	#
	beats_clean['chord'] = beats_clean['chord'].apply(lambda x: chord_to_big_hot(x))
	return beats_clean

def get_baseline_sequences():
	# TODO: add doc-string, explanations
	beats_clean = get_baseline_dataframe()
	mel_ids = beats_clean['melid'].unique()
	sequences = []
	for mel_id in mel_ids:
		# get series with all chords from song with mel_id
		chords = beats_clean[beats_clean['melid'] == mel_id]['chord']
		chords_2d_np_array = np.array(chords.tolist())
		sequences.append(np.array(chords_2d_np_array))

	sequences_clean=[]
	for i in sequences:
		if len(i)>16:
			sequences_clean.append(i)
	return sequences_clean

class BaselineDataset(Dataset):
	def __init__(self, sequences, vocab_size):
		self.sequences = sequences
		self.vocab_size = vocab_size
		self.max_length = max(map(len, self.sequences))  # Add max length

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, i):
		sequence = self.sequences[i]
		sequence_length = len(self.sequences[i])

		# Pad input with 0 up to max length
		input = np.zeros((self.max_length, 24))
		input[:sequence_length,:] = sequence 

		#Pad target with some INVALID value (-1)
		target = np.full((self.max_length-1,), -1)

		if sequence_length > 1:
			# map multi_hot of seq to class representation
			target[:sequence_length-1] = np.apply_along_axis(multi_hot_to_int, 1, sequence[1:]) 

		return {
			"input": torch.tensor(input[:-1]),
			"target": torch.tensor(target).long(),
			"length": sequence_length - 1,  
		}