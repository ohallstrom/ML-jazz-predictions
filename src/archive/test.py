'''
File name: test.py
Author: Oskar
Date created: 17/11/2021
Date last modified: 23/11/2021
Python Version: 3.8
'''
import numpy as np
from utils import chord_to_hot, multi_hot_to_int

chord_mappings = { # TODO: Should add some other test cases
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
   "C7alt": np.array([1,0,0,1,1,0,0,0,1,0,1,0]),
}

binary_to_integer_mappings = {
   "000000010001": 17,
   "000000000001": 1,
   "000000000001": 1,
   "000000000111": 7,
   "100000000101": 2053,
   # np.array([0,0,0,0,0,0,0,1,0,0,0,1]): 17,
   # np.array([0,0,0,0,0,0,0,0,0,0,0,1]): 1,
   # np.array([0,0,0,0,0,0,0,0,0,0,0,0]): 0,
   # np.array([0,0,0,0,0,0,0,0,0,1,1,1]): 7,
   # np.array([1,0,0,0,0,0,0,0,0,1,0,1]): 2053,
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

   print("All tests passed.")

def test_multi_hot_to_int(ground_truth):
   '''
   Testes if the multi_hot_to_int
   function is correct according to
   the ground_truth
   :param ground_truth: dict, maps multi-hot-representation to correct int
   '''
   print("Testing multi_hot_to_int")
   
   for binary in ground_truth.keys():
      # create np_array representation from string
      binary_np_arr = np.array([bin for bin in binary])
      
      np.testing.assert_equal(multi_hot_to_int(binary_np_arr), ground_truth[binary], binary + " maps incorrectly")
      print(binary + " mapped correctly")

   print("All tests passed.")
   
if __name__ == '__main__':
   test_chord_to_hot(chord_mappings)
   test_multi_hot_to_int(binary_to_integer_mappings)
   
