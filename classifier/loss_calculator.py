import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import sys
import dataset
import util.gpu

import pdb

DTYPE = torch.float32
DO_CUDA = True

# Accumulates loss across an epoch
class LossCalculator(object):
	# @arg input_key can be a lambda or a dictionary key
	def __init__(self, model, **opts):
		self.model      = model
		self.input_key  = opts.get('input_key', 'X')
		self.target_key = opts.get('target_key', 'y')
		self.loss_fn    = opts.get('loss', nn.CrossEntropyLoss())
		self.recall_ks  = opts.get('recall', [50])
		self.n_klasses  = opts.get('n', len(list(model.parameters())[-1]))
		self.data_name  = opts.get('name', 'TRAIN')
		if DO_CUDA: self.loss_fn.cuda()
		self.init_epoch()

	def _new_stats(self):
		return Stats(len(self.recall_ks), self.n_klasses)

	# Initialize accumulator for computing epoch peformance
	def init_epoch(self):
		self.epoch_stats = self._new_stats()
		self.init_batch()

	# Computes final stats
	def end_epoch(self):
		stats = self.epoch_stats.compute()
		self.init_epoch()
		return stats

	# Initialize accumulator for computing batch peformance
	def init_batch(self):
		self.batch_stats = self._new_stats()

	# Computes final stats
	def end_batch(self):
		stats = self.batch_stats.compute()
		self.init_batch()
		return stats

	# Perform forward pass on batch or dataset and compute loss
	def __call__(self, data, verbose=True):
		if isinstance(data, dict): # Handle batch (useful for 'train')
			loss = self.calc_on_image(*self.predict(data))
			self.epoch_stats.accumulate(self.batch_stats)
			return loss
		elif isinstance(data, DataLoader): # Handle entire dataset (useful for 'test')
			return self.calc_on_dataloader(data, verbose)
		else:
			raise Exception('invalid input type %s' % type(data))

	# Calculate loss, accuracy, recall on an entire dataset (not just a batch)
	# @return loss, accuracy, recall, recall2
	def calc_on_dataloader(self, dataloader, verbose=True):
		self.init_epoch()
		if verbose: sys.stdout.write('test batch')
		for i, batch in enumerate(dataloader):
			self(batch) # compute and accumulate stats for batch
			assert isinstance(self.epoch_stats.sum_loss, float)
			if verbose and i % 50 == 0:
				mm = util.gpu.get_memory_map()
				sys.stdout.write(' %d' % i)
		if verbose: print()
		return self.end_epoch()

	# For the sake of recall, a batch is believed to be one image
	# Set self.target, self.prediction, self.acc
	# @return loss
	def calc_on_image(self, predictions, targets):
		N = targets.shape[0]
		self.batch_stats.n_example += N
		# Compute loss
		loss = self.loss_fn( predictions, targets.reshape(-1) )
		self.batch_stats.sum_loss += loss.item()
		# Compute/accumulate Recall@X
		confidences, pred_klasses = predictions.max(1) # get confidences and predicted classes
		_,reversed_order = confidences.sort()
		for recall_row, k in enumerate(self.recall_ks):
			if k >= len(confidences): # Use all
				top_k_targets = targets
				top_k_preds   = pred_klasses
			else: # Sort and use top k
				top_k_targets = targets[reversed_order][-k:]
				top_k_preds   = pred_klasses[reversed_order][-k:]
			# Update counts on self
			is_correct = top_k_preds == top_k_targets
			for i, yes in enumerate(is_correct):
				klass = top_k_targets[i]
				self.batch_stats.n_example_by_class[recall_row, klass] += 1
				if yes: self.batch_stats.correct_by_class[recall_row, klass] += 1
		# Return
		return loss

	# Extracts X and Y and makes a forward pass on self.model
	# This is abstracted to its own method in case you want to divide up loss
	# calculation by image (for purposes of Recall@X) but do forward passes
	# with more than one image.
	# Sets self.prediction, self.target
	# Returns (prediction, target)
	def predict(self, batch):
		# Get Y
		if callable(self.target_key):
			self.target = self.target_key(batch)
		elif isinstance(self.target_key, str):
			self.target = self._do_cuda(batch[self.target_key])
		else:
			raise Exception('illegal target type')
		# Pass X through model
		if callable(self.input_key): # lambda or function
			self.prediction = self.input_key(self.model, batch)
		elif isinstance(self.input_key, str):
			self.prediction = self.model(self._do_cuda(batch[self.input_key]))
		else:
			raise Exception('illegal input type')
		# Return
		return self.prediction, self.target

	def _do_cuda(self, tensor):
		if not tensor.is_cuda and next(self.model.parameters()).is_cuda:
			return tensor.cuda()
		else:
			return tensor

class Stats(object):
	def __init__(self, recall_rows, n_klasses):
		self.sum_loss  = 0
		self.n_example = 0
		self.correct_by_class   = torch.zeros(recall_rows, n_klasses, dtype=torch.int32)
		self.n_example_by_class = torch.zeros_like(self.correct_by_class)

	def accumulate(self, other):
		self.sum_loss += other.sum_loss
		self.n_example += other.n_example
		self.correct_by_class += other.correct_by_class
		self.n_example_by_class += other.n_example_by_class

	def compute(self):
		if self.n_example == 0:
			import traceback
			for line in traceback.format_stack(): print(line.strip())
		self.loss = self.sum_loss / self.n_example
		self.acc = self.correct_by_class.sum().item() / self.n_example
		self._compute_recall()
		return self

	# unrel_recall : Number of correct predictions, divided by number of predictions (up to K per image). This biases frequently-seen classes.
	# rec          : Recall is computed class-by-class, then averaged. (It will be nan if not all classes have been seen.)
	# rec2         : Same as `rec`, but only classes have have been seen are used to compute the average.
	def _compute_recall(self):
		self.unrel_recall = self.correct_by_class.sum(1).float() / self.n_example_by_class.sum(1).float()
		recall_by_class_f = self.correct_by_class.float() / self.n_example_by_class.float()
		self.rec = recall_by_class_f.mean(1)
		N = self.n_example_by_class.shape[0]
		self.rec2 = torch.zeros_like(self.rec)
		for row in range(N):
			nonzero_classes = self.n_example_by_class[row,:] != 0 # classes for which we have actually seen examples. ideally this is all classes, but maybe we're working with a subset of the dataset
			if nonzero_classes.sum().item() == 0: # because we might get an empty tensor for recall_by_class_f[row,nonzero_classes]
				self.rec2[row] = math.nan
			else:
				self.rec2[row] = recall_by_class_f[row,nonzero_classes].mean().item() # In case there are classes unrepresented in the dataset, this will give a more useful recall
