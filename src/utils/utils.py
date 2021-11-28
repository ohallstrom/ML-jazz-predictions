'''
File name: utils.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 28/11/2021
Python Version: 3.8
'''
import torch.nn as nn
import torch.nn.functional as F

def get_loss(outputs, targets):
	# TODO: add docstring
	return F.cross_entropy(outputs.transpose(1, 2), targets, ignore_index=-1)

def get_accuracy(outputs, targets):
	# TODO: add docstring
	flat_outputs = outputs.argmax(dim=2).flatten()
	flat_targets = targets.flatten()

	# Mask the outputs and targets
	mask = flat_targets != -1
	return 100 * (flat_outputs[mask] == flat_targets[mask]).sum() / sum(mask)

