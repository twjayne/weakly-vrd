import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.nn as nn
import scipy.io
from classifier.generic_solver import GenericSolver as Solver
from torch.utils.data import DataLoader
import dataset.dataset as dset
from optparse import OptionParser
import numpy as np 
import os
from dataset.example import BasicTestingExample

DEFAULT_DATAROOT = '/home/SSD2/tyler-data/unrel/data/vrd-dataset' # for vision2

def define_model(dimensions):
    # copied from demo.py for defining a model
   

    parser = OptionParser()
    parser.add_option('--data', dest='dataroot', default=DEFAULT_DATAROOT)
    parser.add_option('--lr', dest='lr', default=0.001, type="float")
    parser.add_option('--bs', dest='batch_size', default=32, type="int")
    parser.add_option('--ep', dest='num_epochs', default=30, type="int")
    parser.add_option('-N', dest='train_size', default=None, type="int")
    parser.add_option('--noval', action='store_false', default=True, dest='do_validation')
    parser.add_option('--cpu', action='store_false', default=True, dest='cuda')
    parser.add_option('--log', dest='logdir', default='log')
    parser.add_option('--geom', dest='geometry', default=dimensions)
    parser.add_option('--nosched', dest='no_scheduler', default=False, action='store_true')
    parser.add_option('--patience', dest='patience', default=10, type="int")
    parser.add_option('--test_every', dest='test_every', default=None)
    parser.add_option('--print_every', dest='print_every', default=None)
    parser.add_option('--save', dest='save_every', default=None)
    parser.add_option('--end-save', dest='save_at_end', default=False, action='store_true')
    opts, args = parser.parse_args()

    print('starting ...')
    print(opts)

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

    return model


def load_pretrained_model(model_path):
    # loads the static dict of a pretrained model
    pre_trained_model = torch.load(model_path)

    return pre_trained_model['state_dict']

def load_test_data(pairs):
    _testset = dset.Dataset(DEFAULT_DATAROOT, 'test', pairs=pairs, klass=BasicTestingExample)
    testdata = DataLoader(_testset, batch_size=len(_testset),num_workers=4)
    # testloaders = [testdata]

    return testdata

def save_to(model_path, dimensions, dest, pairs, exp_num):
    pre_trained_model = load_pretrained_model(model_path)

    model = define_model(dimensions)
    # model = define_model('1000 500 250 70')
    model.load_state_dict(pre_trained_model)
    model.eval()
    for setting in pairs:
        testloader = load_test_data(setting)

        for testbatch in testloader:
            prediction = model(testbatch['X'])
            scores_cpu = prediction.cpu()
            scores_np = scores_cpu.data.numpy()
            mydict = {'scores':scores_np}
            print(mydict['scores'])
            print(mydict['scores'].shape)
            scipy.io.savemat(os.path.join(dest,f'{setting}_scores_{exp_num}.mat'), mydict)

if __name__ == "__main__":
    # path of pretrained model
    exp_num = 3
    model_path = f"/home/SSD2/markham-data/weakly-vrd/out/run4/run4/out-exp{exp_num}/best.pth"

    # dimensions of said model
    dimensions = '1000 500 250 70'

    # destination of the prediction scores .mat file
    dest = '/home/tylerjan/code/vrd/unrel/scores'

    # must be in this order for predicate first and then relation/phrase
    settings = ['annotated', 'Lu-candidates']

    save_to(model_path, dimensions, dest, settings, exp_num)
