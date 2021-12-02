'''
File name: data.py
Author: Andrew, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import torch
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
from utils.chord_manipulations import create_chord_type_dict, add_chord_cols

DATA_PATH = "../data/wjazzd.db" 

def get_baseline_dataloader(vocab_size):
	'''
	Creates a dataloader containing preprocessed data from the dataset file
	:return: (DataLoader, Int - number of chord classes)
	'''
	multi_hot_sequences, class_sequences, classes_size = get_baseline_sequences()
	dataset = BaselineDataset(multi_hot_sequences, class_sequences, vocab_size)
	return DataLoader(dataset, batch_size=10, shuffle=False), classes_size

def get_baseline_dataframe():
	'''
	Loads the dataframe from database-file
	and preprocess the data
	:return beats_clean: the preprocessed dataframe
	:return classes_size: int - number of classes in the dataset
	'''
	engine = create_engine(f"sqlite:///{DATA_PATH}")
	beats = pd.read_sql("beats", engine)

	# remove rows without chords
	beats_clean = beats.copy()
	beats_clean['chord'] = beats_clean['chord'].replace(r'^\s*$', np.nan, regex=True)
	beats_clean.dropna(subset = ['chord'], inplace=True)

	chord_type_dict = create_chord_type_dict(beats_clean['chord'])
	classes_size = len(chord_type_dict) * 12 + 1

	# add columns for reduced chord, multihot-encoding and class int, remove consecutive duplicates
	beats_clean = beats_clean.apply(lambda row: add_chord_cols(row, chord_type_dict), axis = 1)
	beats_clean = beats_clean[beats_clean['chord_class'] != beats_clean['chord_class'].shift(1)]
	return beats_clean, classes_size

def get_baseline_sequences():
	'''
	Retrieves sequences of chord data
	for each song in the dataset, can
	be used to create the dataset
	:return multi_hot_sequences: list containing multi_hots per song
	:return class_sequences: list containing class labels per song
	:return classes_size: int - number of classes in the dataset
	'''
	beats_clean, classes_size = get_baseline_dataframe()

	# for each melody, append the multi_hots and classes to a list if valid
	mel_ids = beats_clean['melid'].unique()
	multi_hot_sequences = []
	class_sequences = []
	for mel_id in mel_ids:
		# select data points with current mel_id
		current_beats = beats_clean[beats_clean['melid'] == mel_id]
		
		# append multi_hot / labels that are longer than 1 to its list
		class_labels = np.array(current_beats['chord_class'].tolist())

		# Verify that the there are enough chords in the song before adding
		if len(class_labels) > 1:
			class_sequences.append(np.array(class_labels))
			multi_hots = np.array(current_beats['chord_enc'].tolist())
			multi_hot_sequences.append(np.array(multi_hots))

	return multi_hot_sequences, class_sequences, classes_size

class BaselineDataset(Dataset):
	def __init__(self, multi_hot_sequences, class_sequences, vocab_size):
		self.multi_hot_sequences = multi_hot_sequences
		self.class_sequences = class_sequences
		self.vocab_size = vocab_size
		self.max_length = max(map(len, self.class_sequences))  # Add max length

	def __len__(self):
		return len(self.class_sequences)

	def __getitem__(self, i):
		multi_hot_sequence = self.multi_hot_sequences[i]
		class_sequence = self.class_sequences[i]
		sequence_length = len(class_sequence)

		# Pad input with 0 up to max length
		input = np.zeros((self.max_length, 24))
		input[:sequence_length,:] = multi_hot_sequence

		#Pad target with some INVALID value (-1)
		target = np.full((self.max_length-1,), -1)

		if sequence_length > 1:
			target[:sequence_length-1] = class_sequence[1:] 

		return {
			"input": torch.tensor(input[:-1]),
			"target": torch.tensor(target).long(),
			"length": sequence_length - 1,  
		}