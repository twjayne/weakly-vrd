import os
import torch
import copy
import numpy as np
import dataset.dataset as dataset
import pdb

# This class will split a testset into two testsets:
#	(a) has all of the seen triplets
#	(b) has all of the unseen (zeroshot) triplets
class Splitter(object):

	# The arguments should be of class dataset.Dataset
	def __init__(self, trainset, testset):
		self.trainset = trainset
		self.testset = testset

	def split(self):
		zs_triplets_s = self.unseen_triplets_set() # Get set of triplets which appear in testset but not trainset
		te_triplets_l = self.testset.triplets() # Get list of triplet in testset
		is_unseen     = [x in zs_triplets_s for x in te_triplets_l]
		unseen        = dataset.Subset(self.testset, np.nonzero(is_unseen)[0], 'unseen')
		seen          = dataset.Subset(self.testset, np.nonzero([not x for x in is_unseen])[0], 'seen')
		return (seen, unseen)

	# Get unseen (zero_shot) triplets
	def unseen_triplets_set(self, force=False):
		if force or not os.path.exists(UNSEEN_TRIPLETS_FILE):
			tr_triplets = self.training_triplets_set()
			te_triplets = self.testing_triplets_set()
			zs_triplets = te_triplets - tr_triplets
			torch.save(zs_triplets, UNSEEN_TRIPLETS_FILE)
		else:
			zs_triplets = torch.load(UNSEEN_TRIPLETS_FILE)
		return zs_triplets

	def training_triplets_set(self, force=False):
		return self._training_or_testing_triplets(TRAINING_TRIPLETS_FILE, self.trainset, force)

	def testing_triplets_set(self, force=False):
		return self._training_or_testing_triplets(TESTING_TRIPLETS_FILE, self.testset, force)

	def _training_or_testing_triplets(self, fpath, dataset, force=False):
		if force or not os.path.exists(fpath):
			triplets = set(dataset.triplets())
			torch.save(triplets, fpath)
		else:
			triplets = torch.load(fpath)
		return triplets

UNSEEN_TRIPLETS_FILE = os.path.join(os.path.dirname(__file__), 'unseen_triplets.npy')
TESTING_TRIPLETS_FILE = os.path.join(os.path.dirname(__file__), 'testing_triplets.npy')
TRAINING_TRIPLETS_FILE = os.path.join(os.path.dirname(__file__), 'training_triplets.npy')
