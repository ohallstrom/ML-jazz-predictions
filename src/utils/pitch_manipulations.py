'''
File name: pitch_manipulations.py
Author: Nikunj
Date created: 17/11/2021
Date last modified: 02/11/2021
Python Version: 3.8
'''
import numpy as np

rev_root_mapping = dict((v,k) for k,v in root_mappings.items())

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

def pitch_to_note(pitch): #Should I change it to int value rather than string then
    '''
    Projects a pitch into the respective note value
    :pitch: pitch value to be converted
    :return: the string of note it represents 
    ''' 
    mapping = pitch % 12
    return rev_root_mapping[mapping]


