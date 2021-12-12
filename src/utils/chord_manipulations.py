'''
File name: chord_manipulations.py
Author: Oskar
Date created: 02/12/2021
Date last modified: 02/12/2021
Python Version: 3.8
'''
import numpy as np

root_mappings = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11
}

quality_mappings = {
    "m": np.array([3,7]),
    "": np.array([4,7]),
    "sus": np.array([5,7]), # assumes occurences only of sus and sus7
    "o" : np.array([3,6]), # dimimnished
    "j" : np.array([4,7]), # major
    '+': np.array([4,8]), # augmented
    '-': np.array([3,7]), # minor
    'alt': np.array([3,4,8])
}

step_mappings = { 
    5: 7,
    6: 9, 
    7: 10,
	9: 2
}

def chord_to_big_hot(chord):
    '''
    Projects a chord into a
    multi-hot-encoded representation of length 24
    :param chord: String representation of chord
    :return: the chord in multi-hot-encoding,
    first 12 positions corresponds to the chord format (without root),
    and the 12 last positions to the root position
    ''' 
    # separate the root from the rest of the string
    if len(chord) > 1:
        if chord == 'NC':
            return np.zeros(24)
        elif chord[1] == 'b' or chord[1] == '#': 
            chord_root = root_mappings[chord[:2]]
            chord_rest = chord[2:]
        else:
            chord_root = root_mappings[chord[0]]
            chord_rest = chord[1:]
    else:
        chord_root = root_mappings[chord[0]]
        chord_rest = ""

    # create multi-hot-encoding and add root
    chord_multi_hot = np.zeros(24)
    chord_multi_hot[chord_root+12] = 1

    # retrieve chord quality and steps
    chord_quality = ""
    chord_steps = []

    # check for exception cases
    if chord_rest == 'm7b5':
        chord_multi_hot[[3, 6, 10]] = 1

    # for standard cases
    else:
        offsets = {} # keeps track of cases in which the step is sharp or flat
        for ch in chord_rest:
            if ch == 'b':
                offsets[chord_steps[-1]] = -1
            elif ch == '#':
                offsets[chord_steps[-1]] = 1
            elif ch.isalpha() or ch == '+' or ch == '-': 
                chord_quality += ch
            elif ch.isdigit():
                chord_steps.append(int(ch))
            elif ch == '/':
                break


        # add pitches according to steps
        for step in chord_steps:
            if step in step_mappings:
                idx = step_mappings[step]
                if step == 7:
                    if chord_quality == "j" : 
                        idx += 1
                    elif chord_quality =='o':
                        idx -= 1
                elif step == 5:
                    chord_quality = "None"
                else: 
                    if step in offsets:
                        idx += offsets[step]
                chord_multi_hot[idx] = 1
        
        # add pitches according to chord quality
        if chord_quality in quality_mappings:
            chord_multi_hot[quality_mappings[chord_quality]] = 1

    return chord_multi_hot

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

def split_chord(chord):
	'''
	Returns the root and the rest of the chord 
	without potential bass pitch
	:param chord: str corresponding to chord
	:return: (root, rest)
	'''
	if len(chord) > 1:
		if chord == "NC":
			return ('NC', 'NC')
		elif chord[1] == 'b' or chord[1] == '#': 
			root = chord[:2]
			rest = chord[2:].split('/')[0]
		else:
			root = chord[0]
			rest = chord[1:].split('/')[0]
	else:
		root = chord 
		rest = ""
	
	return (root, rest) 

def get_chord_type_reduction(chord_type, chord_type_dict):
	'''
	Projects the chord_type to the most similar
	chord_type in chord_type_dict and return its
	class number
	:param chord_type: str of chord type to assign number
	:param chord_type_dict: dict of {str, int} mapping chord type to class number
	OBS: Should be sorted by descending length of keys
	:return: (reduced chord_type, class_number)
	'''
	if chord_type in chord_type_dict:
		return (chord_type, chord_type_dict[chord_type])
	elif chord_type == 'NC':
		return ("NC", -1)
	for key in chord_type_dict:
		if chord_type.startswith(key):
			next_char = chord_type[len(key)]
			if next_char not in ['b', "#"]: # check that step is not sharp or flat
				return (key, chord_type_dict[key])
	return ("NC", -1)

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
	essential_types = ['o', '+', '-', ''] # covers essential types in the database 
	for essential_type in essential_types:
		if essential_type not in chord_type_dict: 
			chord_type_dict[essential_type] = len(chord_type_dict)
	
	# sort dict according to chord_type length
	chord_type_dict_sorted = {k: chord_type_dict[k] for k in sorted(chord_type_dict, key = lambda k: -len(k))}

	return chord_type_dict_sorted

def add_chord_cols(row, chord_type_dict):
	'''
	Adds columns with the reduced chord, its class number
	and its multi_hot_encoding
	:param row: row of df containing a chord column
	:param chord_type_dict: dict of {str, int} mapping chord type to class number
	OBS: Should be sorted by descending length of keys
	:return row: row with specified columns added
	'''
	chord_types_n = len(chord_type_dict)
	chord = row['chord']
	(root, chord_type) = split_chord(chord)
	chord_type_r, chord_type_class = get_chord_type_reduction(chord_type, chord_type_dict)

	if root in root_mappings:
		row['chord_red'] = root + chord_type_r
		row['chord_enc'] = chord_to_big_hot(root + chord_type_r)
		row['chord_class'] = 1 + root_mappings[root] * chord_types_n + chord_type_class
	else:
		row['chord_red'] = "NC"
		row['chord_enc'] = chord_to_big_hot("NC")
		row['chord_class'] = 0
		
	return row
