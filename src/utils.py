'''
File name: utils.py
Author: Oskar
Date created: 17/11/2021
Date last modified: 18/11/2021
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
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11
}

quality_mappings = { # TODO: add potential other "qualities"
    "m": np.array([3,7]),
    "": np.array([4,7]),
    "maj": np.array([4,7]),
    "sus": np.array([7]),
    "aug": np.array([4,8]),
    "dim": np.array([3,6]),
    "o" : np.array([3,6]), #dimimnished
    "j" : np.array([4,7]),#major
    '+': np.array([4,8]),#augmented
    '-': np.array([3,7]),#minor
    
    
}

step_mappings = { # TODO: add more?
    2: 2,
    4: 5,
    6: 9,
    7: 10,
    9: 2
}

def chord_to_hot(chord):
    '''
    Projects a chord into a
    multi-hot-encoded representation
    :param chord: String representation of chord
    :return: the chord in multi-hot-encoding
    ''' 
    # separate the root from the rest of the string
    if len(chord) > 1:
        if chord == 'NC':
            return np.zeros(12)
        elif chord[1] == 'b' or chord[1] == '#': # TODO: add character used for #
            chord_root = root_mappings[chord[:2]]
            chord_rest = chord[2:]
        else:
            chord_root = root_mappings[chord[0]]
            chord_rest = chord[1:]
    else:
        chord_root = root_mappings[chord[0]]
        chord_rest = ""

    # retrieve chord quality and steps
    chord_quality = ""
    chord_steps = []

    for ch in chord_rest:
        if ch.isalpha():  # TODO : change to include other characters
            chord_quality += ch
        elif ch.isdigit():
            chord_steps.append(int(ch))
        elif ch == '/':
            break
    
    # for debugging purposes
    print(chord)
    print(chord_root)
    print(chord_quality)
    print(chord_steps)

    # create multi-hot-encoding according to root, quality and steps
    chord_multi_hot = np.zeros(12)
    # add root pitch to 1
    chord_multi_hot[chord_root] = 1
    # add pitches according to chord quality
    chord_multi_hot[(quality_mappings[chord_quality] + chord_root) % 12] = 1
    # add pitches according to extra numbers
    for num in chord_steps:
        if num in step_mappings:
            idx = chord_root + step_mappings[num]
            if num == 7:
                if chord_quality == "maj" or chord_quality == "j" : 
                    idx += 1
                elif chord_quality == "dim" or chord_quality =='o':
                    idx -= 1
            chord_multi_hot[idx % 12] = 1

    return chord_multi_hot

def multi_hot_to_int(multi_hot):
    '''
    Maps multi-hot representation to 
    integer representation by interpretating
    the multi_hot as a binary number. 
    :param multi_hot: multi_hot representation of chord
    :return: integer representation of chord
    '''
    return int("".join(str(x) for x in multi_hot), 2)
