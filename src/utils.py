'''
File name: utils.py
Author: Oskar, Nikunj
Date created: 17/11/2021
Date last modified: 25/11/2021
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

rev_root_mapping = dict((v,k) for k,v in root_mappings.items())

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
    'alt': np.array([3,4,8])
}

step_mappings = { # TODO: add more?
    2: 2,
    4: 5,
    6: 9, # are there any flat or sharp cases? handle these in this case
    7: 10,
    # 9: 2 or handle flat and sharp cases
}

num_quality_mappings_to_int = { # TODO: add potential other "qualities"
    '4711': 1,
    '4710': 2,
    '369': 3,
    '3710': 4,
    '710': 5,
    '4810': 6,
    '34810': 7,
    
    '237': 8,
    '247': 9,
    '27': 10,
    '248': 11,
    '236': 12,
    '2348': 13,
    
    '357': 14,
    '457': 15,
    '57': 16,
    '458': 17,
    '356': 18,
    '3458': 19,
    
    '379': 20,
    '479': 21,
    '79': 22,
    '489': 23,
    '369': 24,
    '3489': 25
}

def chord_to_hot(chord):
    '''
    Projects a chord into a
    multi-hot-encoded representation corresponding
    to a midi-representation
    :param chord: String representation of chord
    :return: the chord in multi-hot-encoding
    ''' 
    # separate the root from the rest of the string
    if len(chord) > 1:
        if chord == 'NC':
            return np.zeros(12)
        elif chord[1] == 'b' or chord[1] == '#': 
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
        if ch.isalpha() or ch == '+' or ch == '-':  
            chord_quality += ch
        elif ch.isdigit():
            chord_steps.append(int(ch))
        elif ch == '/':
            break
    
    # create multi-hot-encoding according to root, quality and steps
    chord_multi_hot = np.zeros(12)
    # add root pitch to 1
    chord_multi_hot[chord_root] = 1
    # add pitches according to chord quality
    if chord_quality in quality_mappings.keys():
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

    # retrieve chord quality and steps
    chord_quality = ""
    chord_steps = []

    for ch in chord_rest:
        if ch.isalpha() or ch == '+' or ch == '-':  # TODO : change to include other characters
            chord_quality += ch
        elif ch.isdigit():
            chord_steps.append(int(ch))
        elif ch == '/':
            break
    
    # create multi-hot-encoding according to root, quality and steps
    chord_multi_hot = np.zeros(24)
    # add root pitch to second part of encoding
    chord_multi_hot[chord_root+12] = 1
    # add pitches according to chord quality
    if chord_quality in quality_mappings.keys():
        chord_multi_hot[quality_mappings[chord_quality] % 12] = 1
    # add pitches according to extra numbers
    for num in chord_steps:
        if num in step_mappings:
            idx = step_mappings[num]
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
    integer representation of the chord's
    class
    :param multi_hot: 24 element multi_hot representation of chord
    :return: integer representation of chord
    '''
    #case with all zeros
    if multi_hot == np.zeros(12):
        return 0
    #case with one instance of one from index 12 to 23
    else:
        root_array = multi_hot[12:]
        root_index = root_array.argmax() 

        chord_form_array = multi_hot[:12]

        string_ints = [str(int) for int in [i for i, e in enumerate(chord_form_array) if e == 1]]
        str_of_ints = "".join(string_ints) 

        j = num_quality_mappings_to_int[str_of_ints]

        return (25*root_index) + j
        
    
def pitch_to_note(pitch): #Should I change it to int value rather than string then
    '''
    Projects a pitch into the respective note value
    :pitch: pitch value to be converted
    :return: the string of note it represents 
    ''' 
    mapping = (pitch - 24) % 12
    return rev_root_mapping[mapping]

