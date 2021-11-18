'''
File name: test.py
Author: Oskar
Date created: 17/11/2021
Date last modified: 18/11/2021
Python Version: 3.8
'''
import numpy as np
from utils import chord_to_hot

chord_mappings = {
  "C": np.array([1,0,0,0,1,0,0,1,0,0,0,0]),
   "Cm7": np.array([1,0,0,1,0,0,0,1,0,0,1,0]),
   "Cmaj7": np.array([1,0,0,0,1,0,0,1,0,0,0,1]),
   "Caug7": np.array([1,0,0,0,1,0,0,0,1,0,1,0]),
   "Cdim7": np.array([1,0,0,1,0,0,1,0,0,1,0,0]),
   "C7": np.array([1,0,0,0,1,0,0,1,0,0,1,0]),
   "Db7": np.array([0,1,0,0,0,1,0,0,1,0,0,1]),
   "Bm7": np.array([0,0,1,0,0,0,1,0,0,1,0,1]),
   "Bmaj7": np.array([0,0,0,1,0,0,1,0,0,0,1,1]),
   "Csus2": np.array([1,0,1,0,0,0,0,1,0,0,0,0]),

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
      np.testing.assert_array_equal(chord_to_hot(chord_str), ground_truth[chord_str], chord_str + " maps incorrectly")
      print(chord_str + " mapped correctly")

   print("All test passed.")
   
if __name__ == '__main__':
   test_chord_to_hot(chord_mappings)
