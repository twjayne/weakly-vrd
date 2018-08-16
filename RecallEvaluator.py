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
        self.pairs = ['annotated', 'Lu-candidates']
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

    def calc_scores(self, testloader):

        return None

    def load_annotated(self):
        # loads pairs.mat from vrd-dataset/test/annotated
        testset = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT, self.dataset), self.split, self.pairs[0], klass=BasicTestingExample)
        annotated = DataLoader(testset, batch_size=len(_testset), num_workers=4)
        return annotated

    def load_candidates(self):
        testset = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT, self.dataset), self.split, self.pairs[1], klass=BasicTestingExample)
        candidates = DataLoader(testset, batch_size=len(_testset), num_workers=4)
        return candidates

    def update_model(self, model):
        # should be called by runner whenever testing needs to be done
        self.model = model

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
            print(f'loading datasets...{setting}')
            # initialize dataloaders for both testset
            _testset[setting] = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT,'vrd-dataset'), 'test', pairs=setting, klass=BasicTestingExample)
            testdata[setting] = DataLoader(_testset[setting], 
                                batch_size=100, 
                                num_workers=4) 
            # run prediction for each and save in .mat
            print(f'calculating scores...')
            for testbatch in testdata[setting]:
                with torch.no_grad():
                    scores = self.model(torch.autograd.Variable(testbatch['X'].cuda()))
                cur_prediction = self.prediction.get(setting, None)
                if type(cur_prediction) is torch.Tensor:
                    self.prediction[setting] = torch.cat((cur_prediction, scores.cpu()),0)
                else:
                    self.prediction[setting] = scores.cpu()

                # self.prediction[setting] = self.model(testbatch['X'].cuda()) 
            # scores_cpu = self.prediction[setting].cpu()
            print(f"size of {setting} is: {self.prediction[setting].shape}")
            scores_np = self.prediction[setting].data.numpy()
            mydict = {'scores':scores_np}
            # sanity check
            # print(mydict['scores'])
            # print(f"from dataset: {setting}\nshape: {mydict['scores'].shape}")
            # save to unrel folder as (ex) "/annotated_<dim>_<id>.mat"
            print(f'saving .mat files...{setting}')
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