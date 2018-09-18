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
parser.add_option('--nowt', dest='weighted_loss', action='store_false', default=True)
assert parser.has_option('--lr')
parser.defaults['batch_size'] = 0.00001
assert parser.has_option('--bs')
parser.defaults['batch_size'] = 1
assert parser.has_option('--tbs')
parser.defaults['test_batch_size'] = 1
assert parser.has_option('--outdir')
parser.defaults['outdir'] = 'log/e2e/vgg16'
assert parser.has_option('--geom')
parser.defaults['geometry'] = '1400 600 300 70'
assert parser.has_option('--test_every')
parser.defaults['test_every'] = 1024

class Model(unrel.Model):
	def __init__(self, n_vis_features, *args, **kwargs):
		super(Model, self).__init__(*args, **kwargs)
		self.classifier = nn.Sequential(*(list(self.classifier.children())[:-1] + [nn.Linear(4096, n_vis_features)])) # Replace last layer of classifier for our desired output dimension
		geom = [ int(x) for x in kwargs['opts'].geometry.split() ]
		self.apperance_normalizer = nn.BatchNorm1d( 500 )
		self.predicate_classifier = cls.sequential( geom, batchnorm=False )

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
		self.verbose = False
		self.optimizer.zero_grad()

	def _train_step(self, batch):
		self.model.train()
		loss = self.train_loss(batch)
		loss.backward()
		if self.iteration and self.iteration % self.opts.get('backprop_every') == 0: # If this is not the first iteration and we have enough iterations to merit a backprop
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.pairs_n = 0
			self.loss = 0
		if self.iteration % self.print_every == 0:
			self._print('TRAIN_BCH', self.train_loss.end_batch())
		if self.scheduler and testloader is None:
			self.scheduler.step(loss.item())
		return loss

class Runner(shared.Runner):
	def setup_model(self):
		super(Runner, self).setup_model()
		self.model.cuda()

	def _build_model(self, n_vis_features=500):
		print('Building model')
		return Model(n_vis_features, opts=self.opts)

	def setup_opt(self, optimizer_lambda=None, scheduler_lambda=None, solver_lambda=None):
		# Define optimizer, scheduler, solver
		if self.optimizer == None:
			print('Building optimizer...')
			self.optimizer = optimizer(self.model.parameters(), self.opts) if optimizer_lambda else torch.optim.Adam(self.model.parameters(), lr=self.opts.lr)
		if self.scheduler == None and not self.opts.no_scheduler:
			print('Building scheduler...')
			self.scheduler = scheduler_lambda(self.optimizer) if scheduler_lambda else torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [x * self.opts.backprop_every for x in [35, 75, 120, 200, 400, 600, 800]], 0.5)
		if self.solver == None:
			print('Building solver...')
			if solver_lambda:
				self.solver = solver_lambda(self.model, self.optmizer, self.scheduler, loss_calculator, self.opts.__dict__)
			else:
				if self.opts.weighted_loss:
					print('Using weighted loss...')
					pred_klasses = torch.Tensor(self.trainloader.dataset.triplets())[:,1] - 1
					assert pred_klasses.max() == 69, pred_klasses.max()
					assert pred_klasses.min() == 0,  pred_klasses.min()
					weights = torch.histc(pred_klasses, bins=70, min=0, max=69)
					train_loss_fn = nn.CrossEntropyLoss(weights)
					train_loss = LossCalculator(self.model, input_key=lambda model, batch: model(batch), target_key='preds', loss_fn=train_loss_fn)
				else:
					print('NOT using weighted loss...')
					train_loss = LossCalculator(self.model, input_key=lambda model, batch: model(batch), target_key='preds')
				test_loss = LossCalculator(self.model, input_key=lambda model, batch: model(batch), target_key='preds')
				self.solver = Solver(self.model, self.optimizer, verbose=True, scheduler=self.scheduler, train_loss=train_loss, test_loss=test_loss, **self.opts.__dict__)

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
			batch_sampler = sampler.BatchSampler(sampler.SequentialSampler(self.testset), self.opts.test_batch_size, False) # make test set load without shuffling so that we can use Tyler's RecallEvaluator
			self.testloaders = [data.FauxDataLoader(self.testset, sampler=batch_sampler)]
		else:
			print('No testset')
			self.testloaders = []

if __name__ == '__main__':
	r = Runner()
	r.setup()
	r.train()
