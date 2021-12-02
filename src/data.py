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
from utils.mappings import binary_to_int, chord_to_big_hot, multi_hot_to_int, binary_to_int

DATA_PATH = "../data/wjazzd.db" 

def get_baseline_dataloader(vocab_size):
	# TODO: add doc-string

	baseline_sequences = get_baseline_sequences()
	dataset = BaselineDataset(baseline_sequences, vocab_size)
	return DataLoader(dataset, batch_size=10, shuffle=False)

def get_chord_type_dict(series):
	'''
	Creates a dictionary which maps
	each chord type into a class int
	in range [0, number_chord_types - 1]
	:param series: series containing multi_hot_rep of chords
	:return dict: dictionary which maps 
	the int corresponding to the binary number
	of the chord vector (sparse), to its class int (dense)
	'''
	# transorm each chord type into an int 
	series = series.apply(binary_to_int)

	# sort ints and remove duplicates
	series = series.drop_duplicates().sort_values(ascending = True)
	
	series = series.reset_index(drop = True)

	return dict(zip(series, series.index))


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
	#beats_clean['chord'] = beats_clean['chord'].apply(lambda x: chord_to_big_hot(x))
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

# might want to move these into utils

def remove_chord_root(chord):
	'''
	Returns the chord without its root and potential bass pitch
	:param chord: str corresponding to chord
	:return: str corresponding to chord type
	'''
	if len(chord) > 1:
		if chord == "NC":
			return 'NC'
		elif chord[1] == 'b' or chord[1] == '#': 
			chord = chord[2:]
		else:
			chord = chord[1:]
	else:
		chord = ""
	
	return chord.split('/')[0] 

def get_chord_type_number(chord_type, chord_type_dict):
	'''
	Projects the chord_type to the most similar
	chord_type in chord_type_dict and return its
	class number
	:param chord_type: str of chord type to assign number
	:param chord_type_dict: dict of {str, int} mapping chord type to class number
	OBS: Should be sorted by descending length of keys
	:return: int - class number
	'''
	if chord_type in chord_type_dict:
		return chord_type_dict[chord_type]
	for key in chord_type_dict.keys():
		if chord_type.startswith(key):
			next_char = chord_type[len(key)]
			if next_char not in ['b', "#"]: # chec
				return chord_type_dict[key]
	return len(chord_type_dict) - 1 # will never be run if "" is the last key of chord_type_dict

def create_chord_type_dict(series):
	'''
	Creates a dictionary that maps chord_types to their class int.
	Only includes essential chord_types and chord_types with at least
	1 % of the total occurrences of chord_types
	:param series: pd.Series containing all chord occurences
	:return chord_type_dict_sorted: dict of {str, int} mapping chord type to class int,
	sorted by descending length of keys
	'''
	n = series.shape[0]

	# get chord types and filter away NC and types with too few occurences
	chord_types = series.apply(lambda x: remove_chord_root(x))
	chord_types = chord_types[~chord_types.str.contains('NC')]
	chord_type_percentage = chord_types.value_counts()/n

	chord_types_filtered = chord_type_percentage[chord_type_percentage > 0.01].index.to_list()

	# create dictionary and add essential types if missing
	chord_type_dict = dict(zip(chord_types_filtered, range(len(chord_types_filtered))))
	essential_types = ['o', 'j7', '+', 'sus', ''] # covers essential types in the database
	for essential_type in essential_types:
		if essential_type not in chord_type_dict: 
			chord_type_dict[essential_type] = len(chord_type_dict)
	
	# sort dict according to chord_type length
	chord_type_dict_sorted = {k: chord_type_dict[k] for k in sorted(chord_type_dict, key = lambda k: -len(k))}

	return chord_type_dict_sorted
