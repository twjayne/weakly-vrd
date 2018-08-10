import numpy as np
import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import os, datetime as dt
import torch.nn as nn
from classifier.generic_solver import GenericSolver as Solver
from torch.utils.data import DataLoader
from optparse import OptionParser
import pdb

import util.loss
import util.logger as logger
import dataset.dataset as dset
import dataset.zeroshot as zeroshot


DEFAULT_DATAROOT = os.path.join(os.path.expanduser('~'), 'proj/weakly-vrd/data/vrd-dataset')


parser = OptionParser()
parser.add_option('--data', dest='dataroot', default=DEFAULT_DATAROOT)
parser.add_option('--lr', dest='lr', default=0.001, type="float")
parser.add_option('--bs', dest='batch_size', default=32, type="int")
parser.add_option('--ep', dest='num_epochs', default=30, type="int")
parser.add_option('-N', dest='train_size', default=None, type="int")
parser.add_option('--noval', action='store_false', default=True, dest='do_validation')
parser.add_option('--cpu', action='store_false', default=True, dest='cuda')
parser.add_option('--log', dest='logdir', default='log')
parser.add_option('--geom', dest='geometry', default='1000 2000 2000 70')
parser.add_option('--nosched', dest='no_scheduler', default=False, action='store_true')
parser.add_option('--patience', dest='patience', default=10, type="int")
parser.add_option('--test_every', dest='test_every', default=None)
parser.add_option('--print_every', dest='print_every', default=None)
parser.add_option('--save', dest='save_every', default=None)
parser.add_option('--end-save', dest='save_at_end', default=False, action='store_true')
parser.add_option('--save-best', dest='save_best', default=False, action='store_true')
parser.add_option('--nosplitzs', dest='split_zeroshot', default=True, action='store_false')
opts, args = parser.parse_args()

logger.Logger(opts.logdir,
	'' if opts.do_validation else 'noval',
	'N-%d ep-%d lr-%f geom-%s hash-%d.log' %
	(opts.train_size or 0, opts.num_epochs, opts.lr, opts.geometry, hash(frozenset(opts.__dict__))))
print('starting ...')
print(opts)

# Define model
print('Building model')
layer_widths = [int(x) for x in opts.geometry.split(' ')]
print('Geometry: %s' % (' '.join((str(x) for x in layer_widths))))
def model_generator(layer_widths, is_batch_gt_1):
	for i in range(1, len(layer_widths)):
		yield nn.Linear(layer_widths[i-1], layer_widths[i])
		if i < len(layer_widths) - 1: # All except the last
			yield nn.Dropout()
			yield nn.BatchNorm1d(layer_widths[i])
			yield nn.ReLU()
layers = list(model_generator(layer_widths, opts.train_size == 1))
model  = nn.Sequential(*layers).double()

# Define optimizer, scheduler, solver
print('Building optimizer, scheduler, solver...')
optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
scheduler = None if opts.no_scheduler else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=opts.patience)
solver    = Solver(model, optimizer, verbose=True, scheduler=scheduler, **opts.__dict__)

# Initialize train and test sets
print('Initializing dataset')
dataroot = opts.dataroot
_trainset = dset.Dataset(dataroot, 'train', pairs='annotated')
trainloader = DataLoader(_trainset, batch_size=opts.batch_size, shuffle=True, num_workers=4)

# Use subset of train data
if opts.train_size: # if --N: override the __len__ method of the dataset so that only the first N items will be used
	def train_size(unused): return opts.train_size
	_trainset.__class__.__len__ = train_size

if opts.do_validation: # Defatult True
	_testset = dset.Dataset(dataroot, 'test', pairs='annotated')
	if opts.split_zeroshot: # Split testset into seen and zeroshot sets
		test_sets = zeroshot.Splitter(_trainset, _testset).split()
		testloaders = [DataLoader(data, batch_size=len(data), num_workers=4) for data in test_sets]
	else: # Use a single (unified) testset
		testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=4)
		testloaders = [testdata]
else: # if --noval
	testloaders = []

def train():
	print('Training...')
	solver.train(trainloader, *testloaders)

# Train and test
if __name__ == '__main__':
	train()
