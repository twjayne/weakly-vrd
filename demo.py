import numpy as np
import torch
import os, datetime as dt
import torch.nn as nn
from classifier.generic_solver import GenericSolver as Solver
from torch.utils.data import DataLoader
import pdb

import dataset.dataset as dset

print('starting ...')

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
solver = Solver(model, optimizer, verbose=True, scheduler=scheduler)

print('Initializing dataset')
dataroot = os.path.join(os.path.expanduser('~'), 'proj/unrel/data/vrd-dataset')
_trainset = dset.Dataset(dataroot, 'train', pairs='annotated')
traindata = DataLoader(_trainset, batch_size=32, shuffle=True, num_workers=1)
_testset = dset.Dataset(dataroot, 'test', pairs='annotated')
testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=1)

print('Training...')
solver.train(traindata, testdata)
