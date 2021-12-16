'''
File name: test.py
Author: Oskar
Date created: 17/11/2021
Date last modified: 08/12/2021
Python Version: 3.8
'''
import numpy as np
from utils.chord_manipulations import chord_to_big_hot

chord_mappings = {
   "C+": np.array([0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C": np.array([0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "Co": np.array([0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C6": np.array([0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C-": np.array([0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C7": np.array([0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C-6": np.array([0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "Cj7": np.array([0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]),
   "C-7": np.array([0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "Csus": np.array([0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),   
   "C79b": np.array([0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]),   
   "Csus7": np.array([0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "Cm7b5": np.array([0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]),
   "Fm7b5": np.array([0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]),
   "C#-6": np.array([0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
   "Cb-6": np.array([0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
}

def test_chord_to_hot(ground_truth):
   '''
   Testes if the chord_to_hot
   function is correct according to
   the ground_truth
   :param ground_truth: dict, maps str to correct multi-hot-representation
   '''
   print("Testing chord_to_hot")

   for chord_str in ground_truth.keys():
      np.testing.assert_array_equal(chord_to_big_hot(chord_str), ground_truth[chord_str], chord_str + " maps incorrectly")
      print(chord_str + " mapped correctly")

   print("All tests passed.")

   
if __name__ == '__main__':
   test_chord_to_hot(chord_mappings)
   
