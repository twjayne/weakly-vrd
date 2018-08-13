import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(__file__,'../..')))
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

DEFAULT_DATAROOT = os.path.join(os.path.dirname(__file__), '..', 'data/vrd-dataset')
DEFAULT_LOGDIR   = os.path.join(os.path.dirname(__file__), '..', 'log')

class Runner(object):

    def __init__(self):
        # Initialize
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.solver = None
        # Parse command line args
        parser = OptionParser()
        parser.add_option('--data', dest='dataroot', default=DEFAULT_DATAROOT)
        parser.add_option('--lr', dest='lr', default=0.001, type="float")
        parser.add_option('--bs', dest='batch_size', default=32, type="int")
        parser.add_option('--ep', dest='num_epochs', default=30, type="int")
        parser.add_option('-N', dest='train_size', default=None, type="int")
        parser.add_option('--noval', action='store_false', default=True, dest='do_validation')
        parser.add_option('--cpu', action='store_false', default=True, dest='cuda')
        parser.add_option('--logdir', dest='logdir', default='log')
        parser.add_option('--log', '--logfile', dest='logfile', default=None)
        parser.add_option('--geom', dest='geometry', default='1000 2000 2000 70')
        parser.add_option('--nosched', dest='no_scheduler', default=False, action='store_true')
        parser.add_option('--patience', dest='patience', default=10, type="int")
        parser.add_option('--test_every', dest='test_every', default=None)
        parser.add_option('--print_every', dest='print_every', default=None)
        parser.add_option('--save', dest='save_every', default=None)
        parser.add_option('--end-save', dest='save_at_end', default=False, action='store_true')
        parser.add_option('--save-best', dest='save_best', default=False, action='store_true')
        parser.add_option('--outdir', dest='outdir', default=None, help="Used for saving checkpoints, etc.")
        parser.add_option('--nosplitzs', dest='split_zeroshot', default=True, action='store_false')
        self.opts, self.args = parser.parse_args()
        opts = self.opts
        # Set logger
        if self.opts.logfile:
            logger.Logger(self.opts.logfile)
        elif self.opts.logdir:
            logger.Logger(self.opts.logdir, 'N-%d ep-%d lr-%f geom-%s hash-%d.log' %
                (opts.train_size or 0, opts.num_epochs, opts.lr, opts.geometry, hash(frozenset(opts.__dict__))))
        print(opts)

    def setup(self):
        print('Init...')

        # Define model
        if self.model == None:
            print('Building model')
            layer_widths = [int(x) for x in self.opts.geometry.split(' ')]
            print('Geometry: %s' % (' '.join((str(x) for x in layer_widths))))
            def model_generator(layer_widths, is_batch_gt_1):
                for i in range(1, len(layer_widths)):
                    yield nn.Linear(layer_widths[i-1], layer_widths[i])
                    if i < len(layer_widths) - 1: # All except the last
                        yield nn.Dropout()
                        yield nn.BatchNorm1d(layer_widths[i])
                        yield nn.ReLU()
            layers = list(model_generator(layer_widths, self.opts.train_size == 1))
            self.model  = nn.Sequential(*layers).double()

        # Define optimizer, scheduler, solver
        if self.optimizer == None:
            print('Building optimizer...')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opts.lr)
        if self.scheduler == None:
            print('Building scheduler...')
            self.scheduler = None if self.opts.no_scheduler else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=self.opts.patience)
        if self.solver == None:
            print('Building solver...')
            self.solver = Solver(self.model, self.optimizer, verbose=True, scheduler=self.scheduler, **self.opts.__dict__)

        # Initialize trainset
        print('Initializing dataset')
        dataroot = self.opts.dataroot
        _trainset = dset.Dataset(dataroot, 'train', pairs='annotated')
        self.trainloader = DataLoader(_trainset, batch_size=self.opts.batch_size, shuffle=True, num_workers=4)

        # Use subset of train data
        if self.opts.train_size: # if --N: override the __len__ method of the dataset so that only the first N items will be used
            def train_size(unused): return self.opts.train_size
            _trainset.__class__.__len__ = train_size

        # Initialize testset
        if self.opts.do_validation: # Defatult True
            _testset = dset.Dataset(dataroot, 'test', pairs='annotated')
            if self.opts.split_zeroshot: # Split testset into seen and zeroshot sets
                test_sets = zeroshot.Splitter(_trainset, _testset).split()
                self.testloaders = [DataLoader(data, batch_size=len(data), num_workers=4) for data in test_sets]
            else: # Use a single (unified) testset
                testdata = DataLoader(_testset, batch_size=len(_testset), num_workers=4)
                self.testloaders = [testdata]
        else: # if --noval
            self.testloaders = []

    def train(self):
        print('Training...')
        self.solver.train(self.trainloader, *self.testloaders)
