import numpy as np
import torch
assert torch.__version__ == '0.4.0'
import os, datetime as dt
import torch.nn as nn
from classifier.generic_solver import GenericSolver as Solver
from torch.utils.data import DataLoader
import pdb
from optparse import OptionParser

import dataset.dataset as dset

print('starting ...')

parser = OptionParser()
parser.add_option('--cpu', action='store_false', default=True, dest='cuda')
parser.add_option('--lr', dest='lr', default=None, type="float")
parser.add_option('--bs', dest='batch_size', default=32, type="int")
parser.add_option('--ep', dest='num_epochs', default=30, type="int")
parser.add_option('-N', dest='train_size', default=None, type="int")
parser.add_option('--noval', action='store_false', default=True, dest='do_validation')
opts, args = parser.parse_args()
print(opts)

# Define topography
layer_widths = [1000, 2000, 800, 70]
linear_layers = [nn.Linear(layer_widths[l-1], layer_widths[l]) for l in range(1,len(layer_widths))]

# Define model, optimizer, solver
print('Building model, optimizer, solver...')
model = nn.Sequential(
	linear_layers[0],
	nn.BatchNorm1d(2000),
	nn.ReLU(),
	linear_layers[1],
	nn.BatchNorm1d(800),
	nn.ReLU(),
	linear_layers[2],
).double()
# Remove batchnorm if N == 1
if opts.train_size == 1:
	model = nn.Sequential( *[m for m in model.children() if type(m) is not nn.BatchNorm1d] )
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
solver = Solver(model, optimizer, verbose=True, scheduler=scheduler, lr=opts.lr, num_epochs=opts.num_epochs)

# Initialize train and test sets
print('Initializing dataset')
dataroot = os.path.join(os.path.expanduser('~'), 'proj/weakly-vrd/data/vrd-dataset')
_trainset = dset.Dataset(dataroot, 'train', pairs='annotated')
traindata = DataLoader(_trainset, batch_size=opts.batch_size, shuffle=True, num_workers=4)
if opts.train_size: # if --N
	def train_size(unused): return opts.train_size
	_trainset.__class__.__len__ = train_size
if opts.do_validation: # Defatult True
	_testset = dset.Dataset(dataroot, 'test', pairs='annotated')
	testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=4)
else: # if --noval
	testdata = None

# Train and test
if __name__ == '__main__':
	print('Training...')
	solver.train(traindata, testdata)
