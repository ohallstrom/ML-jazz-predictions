'''
File name: utils.py
Author: Andrew, Anmol, Oskar
Date created: 28/11/2021
Date last modified: 16/12/2021
Python Version: 3.8
'''
import logging
import numpy as np
import pickle
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import pandas as pd

root_mappings_inv = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
	4: "E",
	5: "F",
	6: "Gb",
	7: "G",
	8: "Ab",
    9: "A",
	10: "Bb",
	11: "Cb",
}

def get_loss(outputs, targets):
	'''	
	Returns cross entropy loss for lstm predictions
	:param outputs: the outputs of the lstm
	:param targets: the ground truth of chords
	:return: loss
	'''
	return F.cross_entropy(outputs.transpose(1, 2), targets, ignore_index=-1)

def get_accuracy(outputs, targets):
	'''
	Returns accuracy for lstm
	:param outputs: the outputs of the lstm
	:param targets: the ground truth of chords
	:return: accuracy
	'''
	flat_outputs = outputs.argmax(dim=2).flatten()
	flat_targets = targets.flatten()

	# Mask the outputs and targets
	mask = flat_targets != -1
	return 100 * (flat_outputs[mask] == flat_targets[mask]).sum() / sum(mask)

def train(model, dataloader_train, dataloader_val, save_pth, lr, weight_decay):
	'''
	Trains the given model
	during maximum 400 epochs, the results
	are saved into a log file.
	:param model: LSTM-model to train
	:param dataloader_train: dataloader containing training data
	:param dataloader_val: dataloader containing validation data
	:param save_pth: saving path for model
	:param lr: learning rate
	:param weight_decay: weight decay
	:return:
		- losses_val - list of all validation losses
		- accuracies_val - list of all validation accuracies
		- losses - list of all testing losses
		- accuracies - list of all testing accuracies
		- max_val - highest validation accuracy
		- max_val_model - best model in terms of validation accuracy
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=model.to(device) 
	losses = []
	losses_val = []
	accuracies_val = []
	accuracies = []
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10,verbose=True)

	max_val=0
	min_val_loss=10000
	last_max_val_epoch = 0
	last_min_val_epoch = 0
	for epoch in range(400):
		model.train()
		accuracy=0
		avg_loss=0
		accuracy_val=0
		avg_loss_val=0
		count=0
		for batch_idx, batch in enumerate(dataloader_train):
			count+=1
			inputs = batch["input"].float().to(device)
			lengths = batch["length"]
			# print(lengths)
			targets = batch["target"][:, :max(lengths)].to(device)  # Pad_packed cuts off at max_length

			optimizer.zero_grad()
			outputs = model(inputs, lengths)
			# print(inputs.shape,targets.shape, outputs.shape)
			loss = get_loss(outputs, targets)
			lossv=loss.item()
			acc = get_accuracy(outputs, targets)
			accv = acc.item()

			accuracy+=accv
			avg_loss+=lossv

			loss.backward()
			optimizer.step()
		
		avg_loss/=count
		accuracy/=count  

		count=0
		model.eval()
		for batch_idx, batch in enumerate(dataloader_val):
			count+=1
			inputs = batch["input"].float().to(device)
			lengths = batch["length"]
			# print(lengths)
			targets = batch["target"][:, :max(lengths)].to(device)  # Pad_packed cuts off at max_length

			optimizer.zero_grad()
			outputs = model(inputs, lengths)
			# print(inputs.shape,targets.shape, outputs.shape)
			loss = get_loss(outputs, targets)
			lossv=loss.item()
			acc = get_accuracy(outputs, targets)
			accv = acc.item()

			accuracy_val+=accv
			avg_loss_val+=lossv

		avg_loss_val/=count
		accuracy_val/=count  

		scheduler.step(avg_loss_val)

		if max_val < accuracy_val:
			max_val = accuracy_val
			max_val_model = model

		if avg_loss_val < min_val_loss:
			min_val_loss = avg_loss_val
			last_min_val_epoch = epoch
		else:
			if last_min_val_epoch + 20 < epoch:
				logging.info("Training was stopped at epoch: " + str(epoch))
				break

		losses.append(avg_loss)
		losses_val.append(avg_loss_val)
		accuracies.append(accuracy)
		accuracies_val.append(accuracy_val)
		logging.info("EPOCH: " + str(epoch) + " Loss: "+ str(avg_loss)+ " Acc: " + str(accuracy) + " Val_Loss: " + str(avg_loss_val) + " Val_Acc: " + str(accuracy_val))

	return losses_val, accuracies_val, losses, accuracies, max_val, max_val_model

def test(model_pth, dataloader_test, setup):
	'''
	Tests the given model on the given dataloader.
	Writes results to csv
	:param model_pth: the pth of trained model for the setup
	:param dataloader_test: dataloader for test data
	:param setup: string rep of the setup
	'''
	targs=[]
	preds=[]
	accuracy_test=0
	avg_loss_test=0
	count=0
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.load(model_pth)
	model.eval()
	for batch_idx, batch in enumerate(dataloader_test):
		count+=1
		inputs = batch["input"].float().to(device)
		lengths = batch["length"]
		# print(lengths)
		targets = batch["target"][:, :max(lengths)].to(device)  # Pad_packed cuts off at max_length
		# model load model_pth?
		outputs = model(inputs, lengths)

		loss = get_loss(outputs, targets)
		lossv=loss.item()
		acc = get_accuracy(outputs, targets)
		accv = acc.item()

		targs.append(targets.detach().cpu().numpy())
		preds.append(outputs.detach().cpu().numpy())

		accuracy_test+=accv
		avg_loss_test+=lossv


	avg_loss_test/=count
	accuracy_test/=count  


	print("Test Loss:", avg_loss_test, "Test Acc:", accuracy_test)

	preds2=[np.array(i).reshape(-1,157) for i in preds]
	preds_a=np.vstack(preds2)
	preds_all=np.argmax(preds_a, axis=1)
	targs2=[np.array(i) for i in targs]
	targs2=[i.reshape(-1) for i in targs2]
	targs_all=np.concatenate(targs2)
	mask = ~(targs_all==-1)
	print(classification_report(targs_all[mask], preds_all[mask]))
	
	results = pd.DataFrame({'target': targs_all[mask], 'predictions': preds_all[mask]})
	results['target'] = results['target'].apply(chord_class_to_str)
	results['predictions'] = results['predictions'].apply(chord_class_to_str)
	results.to_csv('./../data/predictions_' + setup + '.csv')

def chord_class_to_str(chord_class):
	'''
	Maps chord class back to
	string representation, might use G# instead of Ab and so on
	:param chord_class: int, class of chord
	:returb chord: str of chord
	'''
	if chord_class == 0:
		return 'NC'
	else:
		with open('./../data/chord_class_inv.pickle', 'rb') as handle:
			chord_type_dict_inv = pickle.load(handle)
		chord_types_n = len(chord_type_dict_inv)
		type_int = (chord_class - 1) % chord_types_n
		type_str = chord_type_dict_inv[type_int]
		root_int = int((chord_class - 1 - type_int) / chord_types_n)
		root_str = root_mappings_inv[root_int]
		return root_str + type_str
