# E.g.
# 	python e2e_runner.py --geom '100 100 100 100 1000 700 ; ; 1700 800 300 70' --lr 0.001

shared = __import__(__package__ or '__init__')


import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler

import dataset.faster_rcnn as data
import unrel.unrel_model as unrel
import classifier.generic_solver as generic_solver
import classifier.classifier as cls
from classifier.loss_calculator import LossCalculator

import pdb

parser = shared.parser
parser.add_option('--bp_every', dest='backprop_every', default=4, type='int') # Don't backprop on every 'batch'. Instead backprop after multiple batches.
assert parser.has_option('--bs')
parser.defaults['batch_size'] = 1
assert parser.has_option('--tbs')
parser.defaults['test_batch_size'] = 1
assert parser.has_option('--outdir')
parser.defaults['outdir'] = 'log/e2e/vgg16'
assert parser.has_option('--geom')
parser.defaults['geom'] = '1400 1024 700 71'
assert parser.has_option('--test_every')
parser.defaults['test_every'] = 2048

class Model(unrel.Model):
	def __init__(self, n_vis_features, *args, **kwargs):
		super(Model, self).__init__(*args, **kwargs)
		self.classifier = nn.Sequential(*list(self.classifier.children())[:-1], nn.Linear(4096, n_vis_features))
		self.predicate_classifier = cls.sequential( [1400,1024,700,71], batchnorm=False )

	def forward(self, batch):
		appearance_features = super(Model, self).forward(batch)
		# Get spatial features
		spatial_features = batch['spatial'].float().cuda()
		if len(spatial_features.shape) > 2: spatial_features.unsqueeze_(0) # kludge. find out why dimensions are inconsistent
		# Concatenate apperance and spatial features => visual features
		visual_features = torch.cat( [spatial_features, appearance_features[0::2,:], appearance_features[1::2,:]], 1 )
		# Compute predicate prediction
		predicate_scores = self.predicate_classifier(visual_features)
		return predicate_scores


class Solver(generic_solver.GenericSolver):
	def __init__(self, *args, **kwargs):
		super(Solver, self).__init__(*args, **kwargs)
		self.loss = 0
		self.pairs_n = 0
		self.verbose = False

	def _train_step(self, batch):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.loss_calculator.calc(batch)
		self.pairs_n += batch['preds'].shape[0]
		self.loss = self.loss + loss
		if self.iteration and self.iteration % r.opts.backprop_every == 0: # If this is not the first iteration and we have enough iterations to merit a backprop
			self.loss.backward()
			self._print(self.loss / self.pairs_n, 'ACC_TRAIN')
			self.pairs_n = 0
			self.loss = 0
			self.optimizer.step()
		return loss

class Runner(shared.Runner):
	def setup_model(self):
		super(Runner, self).setup_model()
		self.model.cuda()

	def _build_model(self, n_vis_features=500):
		print('Building model')
		return Model(n_vis_features)

	def setup_opt(self):
		# Define optimizer, scheduler, solver
		if self.optimizer == None:
			print('Building optimizer...')
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opts.lr)
		if self.scheduler == None:
			print('Building scheduler...')
			self.scheduler = None if self.opts.no_scheduler else torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [x * self.opts.backprop_every for x in [35, 75, 120, 200, 400, 600, 800]], 0.5)
		if self.solver == None:
			print('Building solver...')
			loss_calculator = LossCalculator(self.model, input_key=lambda model, batch: model(batch), target_key='preds')
			self.solver = Solver(self.model, self.optimizer, verbose=True, scheduler=self.scheduler, loss_calculator=loss_calculator, **self.opts.__dict__)

	def setup_data(self):
		transform = unrel.TRANSFORM
		# Initialize trainset
		self.trainset = data.Dataset(split='train', pairs='annotated', transform=transform)
		if self.opts.train_size:
			print('Using subset of %d from train_set' % self.opts.train_size)
			batch_sampler = sampler.SequentialSampler(range(self.opts.train_size))
		else:
			batch_sampler = None
		self.trainloader = data.FauxDataLoader(self.trainset, sampler=batch_sampler, batch_size=self.opts.batch_size)
		# Initialize testset
		if self.opts.do_validation:
			self.testset = data.Dataset(split='test', pairs='annotated', transform=transform)
			self.testloaders = [data.FauxDataLoader(self.testset, batch_size=self.opts.test_batch_size)]
		else:
			print('No testset')
			self.testloaders = []

if __name__ == '__main__':
	r = Runner()
	r.setup()
	r.train()
