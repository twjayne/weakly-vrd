import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import util.gpu

import pdb

DTYPE = torch.float32
DO_CUDA = True

class LossCalculator(object):
	# @arg input_key can be a lambda or a dictionary key
	def __init__(self, model, **opts):
		self.model      = model
		self.input_key  = opts.get('input_key', 'X')
		self.target_key = opts.get('target_key', 'y')
		self.loss_fn    = opts.get('loss', nn.CrossEntropyLoss())
		if DO_CUDA: self.loss_fn.cuda()

	def calc(self, data, do_recall=False):
		if isinstance(data, dict):
			return self._calc_on_batch(data)
		elif isinstance(data, DataLoader):
			return self._calc_on_dataloader(data, do_recall)
		else:
			raise Exception('invalid input type %s' % type(data))

	# Calculate loss, accuracy, recall on an entire dataset (not just a batch)
	def _calc_on_dataloader(self, dataloader, do_recall=False):
		total_loss = 0.
		quotient = 0.
		n_preds = 0
		if do_recall:
			self.targets = []
			self.confidences = []
			self.predicted_classes = []
		for i, batch in enumerate(dataloader):
			loss = self._calc_on_batch(batch).item()
			assert isinstance(loss, float)
			total_loss += loss
			quotient += self.prediction.shape[0]
			n_preds += self.prediction.shape[0]
			if i % 50 == 0:
				mm = util.gpu.get_memory_map()
				print('test batch %5d completed. total pairs %8d' % (i, n_preds), mm)
			# Compute intermediates for recall
			if do_recall:
				self.targets.append(self.target.cpu())
				cons, pcls = self.prediction.max(1) # Get the top-1 predicted class and its confidence
				self.confidences.append(torch.Tensor([x.item() for x in cons.cpu()])) # kludge. for some reason, I get a memory leak with `# confidences.append(cons.cpu())`
				self.predicted_classes.append(pcls.cpu())
		avg_loss = total_loss / quotient
		if do_recall:
			# Collect stored intermediates
			self.targets = torch.cat(self.targets)
			self.confidences = torch.cat(self.confidences)
			self.predicted_classes = torch.cat(self.predicted_classes)
		return avg_loss

	# Compute Recall@x
	def recall(self, k):
		_,reversed_order = self.confidences.sort()
		correct_predictions = self.predicted_classes[reversed_order][-k:] == self.targets[reversed_order][-k:]
		return correct_predictions.sum().item() / float(k)

	# Set self.target, self.prediction, self.acc
	def _calc_on_batch(self, batch):
		# Get Y
		self.target = self._do_cuda(batch[self.target_key])
		# Pass X through model
		if callable(self.input_key): # lambda or function
			self.prediction = self.input_key(self.model, batch)
		elif isinstance(self.input_key, str):
			self.prediction = self.model(self._do_cuda(batch[self.input_key]))
		# Compute accuracy
		correct_predictions = torch.sum( torch.argmax(self.prediction, 1) == self.target ).item()
		self.acc = (correct_predictions / float(self.target.shape[0]))
		# Return loss
		return self.loss_fn( self.prediction, self.target.reshape(-1) )

	def _do_cuda(self, tensor):
		if not tensor.is_cuda and next(self.model.parameters()).is_cuda:
			return tensor.cuda()
		else:
			return tensor
