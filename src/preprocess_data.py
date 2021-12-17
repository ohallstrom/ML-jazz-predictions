'''
File name: preprocess_data.py
Author: Anmol, Oskar
Date created: 16/12/2021
Date last modified: 16/12/2021
Python Version: 3.8
'''
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from utils.chord_manipulations import create_chord_type_dict, add_chord_cols

DATA_PATH = "../data/wjazzd.db"

if __name__ == '__main__':
    engine = create_engine(f"sqlite:///{DATA_PATH}")
    beats = pd.read_sql("beats", engine)
    beats_clean = beats.copy()

    # remove rows without chords
    beats_clean['chord'] = beats_clean['chord'].replace(r'^\s*$', np.nan, regex=True)
    beats_clean.dropna(subset = ['chord'], inplace=True)

    # create dictionaries mapping between string and int representation of chord, save inv version
    chord_type_dict, chord_type_dict_inv = create_chord_type_dict(beats_clean['chord'])
    with open('./../data/chord_class_inv.pickle', 'wb') as handle:
        pickle.dump(chord_type_dict_inv, handle)

    classes_size = len(chord_type_dict) * 12 + 1

    # add columns for reduced chord, multihot-encoding and class int, remove consecutive duplicates
    beats_clean = beats_clean.apply(lambda row: add_chord_cols(row, chord_type_dict), axis = 1)
    beats_clean = beats_clean[beats_clean['chord_class'] != beats_clean['chord_class'].shift(1)]
    print(beats_clean.head())

    # prepare pitch column
    melody = pd.read_sql("melody", engine)
    melody_clean = melody.copy()
    melody_clean['pitch'] = melody_clean['pitch'].replace(r'^\s*$', np.nan, regex=True)
    melody_clean.dropna(subset=['pitch'],inplace=True)
    melody_clean['pitch'] = melody_clean['pitch'].apply(lambda x: x % 12)

    # Filter by melody
    mel_ids = beats_clean['melid'].unique()
    pitch_vals=[]

    # TODO: add explanation
    for mel_id in mel_ids:
        # print(mel_id)
        onsets = beats_clean[beats_clean['melid'] == mel_id]['onset']
        for i in range(len(onsets)-1):
            st = onsets.iloc[i]
            end = onsets.iloc[i+1]
            mel = melody_clean[melody_clean['melid']==mel_id]
            inds = mel['onset'].between(st, end, inclusive='left')
            pitches = mel[inds]['pitch']
            bag_of_notes = np.zeros((12))
            d = pitches.value_counts().to_dict()
            for key in d.keys():
                bag_of_notes[int(key)]=d[key]

            bon=np.array(bag_of_notes)
            bon=bon/(np.sum(bon)+0.000001)
            bon_l=bon.tolist()
            
            pitch_vals.append([mel_id, st, bon_l])

    pitch_df = pd.DataFrame(pitch_vals, columns=['melid', 'onset','pitch_bow'])
    print(pitch_df.shape)
    beats_clean = pd.merge(beats_clean, pitch_df, on=['melid','onset'])
    # correct format of chord_enc
    beats_clean['chord_enc'] = beats_clean['chord_enc'].apply(lambda x: list(x))
    beats_clean.to_csv('../data/data_preprocessed.csv')
