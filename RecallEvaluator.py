import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import dataset.dataset as dset
from dataset.example import BasicTestingExample
import os
import scipy.io
from subprocess import Popen, PIPE, STDOUT

# path to folder containing vrd-dataset and others
# DEFAULT_DATAROOT = '/home/SSD2/tyler-data/unrel/data'
# vision5
# DEFAULT_DATAROOT = "/data/tyler/unrel/data/vrd-dataset"
# UNREL_PATH = "/home/tylerjan/code/vrd/unrel"
# SCORES_PATH = "/home/tylerjan/code/vrd/unrel/scores"


class RecallEvaluator(object):
    def __init__(self, dataroot, unrel_path, scores_path):
        # model: current model weights
        # dimensions: dimensions of each layer in a list, originally empty
        # prediction: output scores of a forward pass of testset
        self.model = None
        # self.dimensions = '1000 2000 2000 70'
        self.prediction = {}
        self.split = 'test'
        self.pairs = None
        self.DEFAULT_DATAROOT = dataroot
        self.UNREL_PATH = unrel_path
        self.SCORES_PATH = scores_path
        
    def predict(self):
        # load model, testdata, and save scores in self.prediction
        # load testdata and calculate self.prediction
        model = self.model
        testloader = self.load_test_data()
        for testbatch in testloader:
            self.prediction = model(testbatch['X'])
        # load annotations
        annotations = self.load_full_annotations()
        # load candidatepairs
        pairs = self.load_pairs()

        return (pairs, self.prediction, annotations)

    def load_test_data(self):
        # load testdata into a testloader
        testset = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT, self.dataset),
                                self.split, self.opts.candidatepairs, 
                                klass=BasicTestingExample)
        testdata = DataLoader(testset, batch_size=len(_testset), num_workers=4)
        return testdata

    def load_annotated(self):
        # loads pairs.mat from vrd-dataset/test/annotated
        testset = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT, self.dataset), self.split, self.opts.candidatepairs, klass=BasicTestingExample)
        testdata = DataLoader(testset, batch_size=len(_testset), num_workers=4)
        return None

    def load_candidates(self):

        return None

    def define_model(self, dimensions):
        # copied from demo.py for defining a model
        parser2 = OptionParser()
        parser2.add_option('--data', dest='dataroot', default=self.DEFAULT_DATAROOT)
        parser2.add_option('--lr', dest='lr', default=0.001, type="float")
        parser2.add_option('--bs', dest='batch_size', default=32, type="int")
        parser2.add_option('--ep', dest='num_epochs', default=30, type="int")
        parser2.add_option('-N', dest='train_size', default=None, type="int")
        parser2.add_option('--noval', action='store_false', default=True, dest='do_validation')
        parser2.add_option('--cpu', action='store_false', default=True, dest='cuda')
        parser2.add_option('--log', dest='logdir', default='log')
        parser2.add_option('--geom', dest='geometry', default=dimensions)
        parser2.add_option('--nosched', dest='no_scheduler', default=False, action='store_true')
        parser2.add_option('--patience', dest='patience', default=10, type="int")
        parser2.add_option('--test_every', dest='test_every', default=None)
        parser2.add_option('--print_every', dest='print_every', default=None)
        parser2.add_option('--save', dest='save_every', default=None)
        parser2.add_option('--end-save', dest='save_at_end', default=False, action='store_true')
        opts, args = parser2.parse_args()

        # print('starting ...')
        # print(opts)

        # print('Building model')
        layer_widths = [int(x) for x in opts.geometry.split(' ')]
        # print('Geometry: %s' % (' '.join((str(x) for x in layer_widths))))
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

    def update_model(self, model):
        # should be called by runner whenever testing needs to be done
        
        # for testing
        # pre_trained_model = torch.load("/home/SSD2/markham-data/weakly-vrd/out/geom 1000 2000 2000 70/exp2-3/best.pth")['state_dict']
        # model = self.define_model('1000 2000 2000 70')
        # model.load_state_dict(pre_trained_model)
        # model.eval()

        self.model = model

    def test_subprocess(self, unrel, scores, dimensions, exp_id):
        # rc = Popen(f"{unrel}/run_recall.sh baseline full {scores} '{dimensions}' '{exp_id}'", shell=True)
        rc = Popen(f"{unrel}/run_recall.sh baseline full {scores} '{dimensions}' '{exp_id}'", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        rc_out = str(rc.stdout.read(), 'utf-8')
        # with open("rc_out.txt","w+") as text_file:

        results = []
        # print(rc_out)
        data = rc_out.split('\n')
        print(f"{data}")
        for line in data[-8:-1]:
            print(f"line: {line}")
            data = line.split()[-1]
            print(f"data: {data}")
            if data[0] != 'z':
                results.append(line.split()[-1])
        print(f"results: {results}")
        recalls = {}
        recalls['seen_predicate'] = results[0]
        recalls['seen_phrase'] = results[1]
        recalls['seen_relationship'] = results[2]
        recalls['unseen_predicate'] = results[3]
        recalls['unseen_phrase'] = results[4]
        recalls['unseen_relationship'] = results[5]
        print(f"recalls: {recalls}")

    def recall_from_matlab(self, model):
        # save to .mat, call infer_from_scores.m to evaluate recall
        # assume model is given by runner, use it to run predictions from both annotated and Lu-candidates testset
        # also requires experiment name
        # return a dictionary of recalls{'seen/unseen_predicate/phrase/relationship'}
        # update model and put in eval() mode
        self.update_model(model)
        self.model.eval()
        # settings, should always be in this order for testing
        settings = ['annotated', 'Lu-candidates']
        # save the predictions
        _testset = {}
        testdata = {}
        for setting in settings:
            print('loading datasets...')
            # initialize dataloaders for both testset
            _testset[setting] = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT,'vrd-dataset'), 'test', pairs=setting,klass=BasicTestingExample)
            testdata[setting] = DataLoader(_testset[setting], 
                                batch_size=len(_testset[setting]), 
                                num_workers=4)
            # run prediction for each and save in .mat
            for testbatch in testdata[setting]:
                print('calculating scores...')
                self.prediction[setting] = self.model(testbatch['X'].cuda())
                scores_cpu = self.prediction[setting].cpu()
                scores_np = scores_cpu.data.numpy()
                mydict = {'scores':scores_np}
                # sanity check
                # print(mydict['scores'])
                # print(f"from dataset: {setting}\nshape: {mydict['scores'].shape}")
                # save to unrel folder as (ex) "/annotated_<dim>_<id>.mat"
                print('saving .mat files')
                scipy.io.savemat(os.path.join(self.SCORES_PATH, f'{setting}.mat'), mydict)
                # print(f"{setting}.mat file is saved")
                self.prediction = {}
                
        # use subprocess to run
        print('starting matlab...')
        # rc = Popen(f"{self.UNREL_PATH}/run_recall.sh baseline full {self.SCORES_PATH}", shell=True)
        # return {}
        rc = Popen(f"{self.UNREL_PATH}/run_recall.sh baseline full {self.SCORES_PATH}", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        rc_out = str(rc.stdout.read(), 'utf-8')

        results = []
        # print(rc_out)
        data = rc_out.split('\n')
        # print(f"{data}")
        for line in data[-8:-1]:
            # print(f"line: {line}")
            data = line.split()[-1]
            # print(f"data: {data}")
            if data[0] != 'z':
                results.append(line.split()[-1])
        # print(f"results: {results}")
        recalls = {}
        recalls['seen_predicate'] = results[0]
        recalls['seen_phrase'] = results[1]
        recalls['seen_relationship'] = results[2]
        recalls['unseen_predicate'] = results[3]
        recalls['unseen_phrase'] = results[4]
        recalls['unseen_relationship'] = results[5]
        # print(f"recalls: {recalls}")

        return recalls

if __name__ == '__main__':
    evalr = RecallEvaluator('/home/SSD2/tyler-data/unrel/data',"/home/tylerjan/code/vrd/unrel","/home/tylerjan/code/vrd/unrel/scores")