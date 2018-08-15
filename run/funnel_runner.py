# E.g.
# 	python funnel_runner.py --geom '1000 700 ; ; 1700 800 300 70' --lr 0.001

import shared
from classifier import split_model
import re, os
import dataset.funnel_dataset as dset
from torch.utils.data import DataLoader
import dataset.zeroshot as zeroshot

import pdb

class Runner(shared.Runner):

	def setup_model(self):
		subnets = [[int(x) for x in re.findall(r'\d+', sub)] for sub in self.opts.geometry.split(';')]
		self.model = split_model.Funnel(*subnets)

	def setup_data(self):
		dataroot = self.opts.dataroot
		# Initialize trainset
		scenic_features_dir = self.args.pop(0)
		image_name_map = self.args.pop(0)
	
		_trainset = dset.FunnelDataset(dataroot, 'train', scenic_features_dir, image_name_map, pairs='annotated')
		self.trainloader = DataLoader(_trainset, batch_size=self.opts.batch_size, shuffle=True, num_workers=4)

		# Use subset of train data
		if self.opts.train_size: # if --N: override the __len__ method of the dataset so that only the first N items will be used
			def train_size(unused): return self.opts.train_size
			_trainset.__class__.__len__ = train_size

		# Initialize testset
		if self.opts.do_validation: # Defatult True
			scenic_features_dir = self.args.pop(0)
			image_name_map = self.args.pop(0)
			_testset = dset.FunnelDataset(dataroot, 'test', scenic_features_dir, image_name_map, pairs='annotated')
			if self.opts.split_zeroshot: # Split testset into seen and zeroshot sets
				test_sets = zeroshot.Splitter(_trainset, _testset).split()
				self.testloaders = [DataLoader(data, batch_size=len(data), num_workers=4) for data in test_sets]
			else: # Use a single (unified) testset
				testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=4)
				self.testloaders = [testdata]
		else: # if --noval
			self.testloaders = []

if __name__ == '__main__':
	r = Runner()
	r.setup()
	r.train()
