import torch
import os, datetime as dt
import torch.nn as nn
import classifier.solver as solver
from torch.utils.data import DataLoader

from basicData import BasicData

print('starting ...')

layer_widths = [1000, 2000, 800, 71]
linear_layers = [nn.Linear(layer_widths[l-1], layer_widths[l]) for l in range(1,len(layer_widths))]

print('Building model, optimizer, solver...')
model = nn.Sequential(
	linear_layers[0],
	nn.ReLU(),
	linear_layers[1],
	nn.ReLU(),
	linear_layers[2],
)
optimizer = torch.optim.Adam(model.parameters())
solver = solver.Solver(model, optimizer, verbose=True, print_every=10)

print('Initializing dataset')
trainset = BasicData(os.path.join(os.path.expanduser('~'), 'proj/unrel/data/vrd-dataset'))
traindata = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
testset = BasicData(os.path.join(os.path.expanduser('~'), 'proj/unrel/data/vrd-dataset'))
testset.train_pairs = testset.test_pairs # kludge. The BasicData class should have a design change.

print('Training...')
solver.train(traindata, testset)
