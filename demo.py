import numpy as np
assert np.__version__ == '1.14.5'
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
parser.add_option('--cpu', action='store_false', default=True, dest='do_cuda')
opts, args = parser.parse_args()

layer_widths = [1000, 2000, 800, 70] # Should this be 71?
linear_layers = [nn.Linear(layer_widths[l-1], layer_widths[l]) for l in range(1,len(layer_widths))]

print('Building model, optimizer, solver...')
model = nn.Sequential(
	linear_layers[0],
	nn.ReLU(),
	linear_layers[1],
	nn.ReLU(),
	linear_layers[2],
).double()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
solver = Solver(model, optimizer, verbose=True, scheduler=scheduler, cuda=opts.do_cuda)

print('Initializing dataset')
dataroot = os.path.join(os.path.expanduser('~'), 'proj/unrel/data/vrd-dataset')
_trainset = dset.Dataset(dataroot, 'train', pairs='annotated')
traindata = DataLoader(_trainset, batch_size=32, shuffle=True, num_workers=4)
_testset = dset.Dataset(dataroot, 'test', pairs='annotated')
testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=4)

if __name__ == '__main__':
	print('Training...')
	solver.train(traindata, testdata)
