import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import dataset.dataset as dset
from dataset.example import BasicTestingExample
import os, sys, math
import scipy.io
from subprocess import Popen, PIPE, STDOUT
import numpy as np

# path to folder containing vrd-dataset and others
# DEFAULT_DATAROOT = '/home/SSD2/tyler-data/unrel/data'
# vision5
# DEFAULT_DATAROOT = "/data/tyler/unrel/data/vrd-dataset"
# UNREL_PATH = '/home/tylerjan/code/vrd/unrel'
# SCORES_PATH = '/home/tylerjan/code/vrd/unrel/scores'


class RecallEvaluator(object):
    def __init__(self, dataroot=None, unrel_path=None, scores_path=None):
        # model: current model weights
        # dimensions: dimensions of each layer in a list, originally empty
        # prediction: output scores of a forward pass of testset
        self.prediction = {}

        self.split = 'test'
        self.pairs = ['annotated', 'Lu-candidates']
        self.DEFAULT_DATAROOT = dataroot or os.environ['UNREL_DATA']
        self.UNREL_PATH = unrel_path or os.environ['UNREL_CODE']
        self.SCORES_PATH = scores_path or '/tmp/recall-evaluator'
        os.makedirs(self.SCORES_PATH, exist_ok=True)

        self.recalls = {}
        
        self.model = None
        self.supervision = 'full'
        self.num_negatives = 0
        self.split = 'test'
        self.use_languagescores = False
        self.Nre = 50
        self.zeroshot = False
        self.use_objectscores = False
        self.dataset = 'vrd-dataset'
        self.candidatespairs = 'annotated'
        self.annotatedpairs = 'annotated'
        self.alpha_objects = 0.5


    def evaluate_recall(self, model, supervision, language):
        self.model = model
        self.model.eval()
        self.supervision = supervision
        self.use_languagescores = language
        seen_recalls = self.infer(False) 
        unseen_recalls = self.infer(True) # zeroshot
        self.recalls = {'seen_predicate':seen_recalls[0], 'seen_phrase':seen_recalls[1], 'seen_relationship':seen_recalls[2], 'unseen_predicate':unseen_recalls[0], 'unseen_phrase':unseen_recalls[1], 'unseen_relationship':unseen_recalls[2]}

        return self.recalls

    def infer(self, zeroshot):
        self.zeroshot = zeroshot
        self.candidatespairs = 'annotated'
        self.use_objectscores = False
        # print(f"Zeroshot setting...{zeroshot}")
        # print(f"Using language scores...{self.use_languagescores}")
        # print(f"Predicate Detection...")
        pairs, scores, annotations = self.predict()
        # np.save('predictions1.npy', scores)
        # scores = np.load('predictions1.npy')
        candidates, groundtruth = self.format_testdata_recall(pairs, scores, annotations)
        recall_predicate, _ = self.top_recall_Relationship(self.Nre, candidates, groundtruth)
        self.candidatespairs = 'Lu-candidates'
        self.use_objectscores = True
        # print(f'Phrase/Relationship Detection...')
        pairs, scores, annotations = self.predict()
        # np.save('predictions2.npy', scores)
        # scores = np.load('predictions2.npy')
        candidates, groundtruth = self.format_testdata_recall(pairs, scores, annotations)
        recall_relationship, _ = self.top_recall_Relationship(self.Nre, candidates, groundtruth)
        recall_phrase, _ = self.top_recall_Phrase(self.Nre, candidates, groundtruth)

        return [recall_predicate, recall_phrase, recall_relationship]
    def top_recall_Phrase(self, Nre, candidates, groundtruth):
        tuple_confs_cell = candidates['scores']
        tuple_labels_cell = candidates['triplet']
        sub_bboxes_cell = candidates['sub_box']
        obj_bboxes_cell = candidates['obj_box']

        gt_tuple_label = groundtruth['triplet']
        gt_sub_bboxes = groundtruth['sub_box']
        gt_obj_bboxes = groundtruth['obj_box']
        # sort candidates by confidence scores
        num_images = len(gt_tuple_label)
        for i in range(num_images):
            ind = tuple_confs_cell[i].argsort(axis=0)[::-1]
            ind = np.squeeze(ind,axis=1)
            if len(ind) >= Nre:
                ind = ind[0:Nre]
            tuple_confs_cell[i] = tuple_confs_cell[i][ind]
            tuple_labels_cell[i] = tuple_labels_cell[i][:,ind]
            obj_bboxes_cell[i] = obj_bboxes_cell[i][ind,:]
            sub_bboxes_cell[i] = sub_bboxes_cell[i][ind,:]

        
        num_pos_tuple = 0
        for i in range(num_images):
            num_pos_tuple += len(gt_tuple_label[i][0])

        tp_cell = []
        fp_cell = []

        gt_thr = 0.5
        

        for i in range(num_images):

            gt_tupLabel = gt_tuple_label[i]

            if gt_obj_bboxes[i].shape[0] == 0 or gt_obj_bboxes[i].shape[1] == 0:
                gt_box_entity = []
            else:
                c = np.minimum(gt_obj_bboxes[i][:,0:2],gt_sub_bboxes[i][:,0:2])
                d = np.maximum(gt_obj_bboxes[i][:,2:4],gt_sub_bboxes[i][:,2:4])

                gt_box_entity = np.hstack((c,d))

            num_gt_tuple = len(gt_tupLabel[0])
            gt_detected = np.zeros(num_gt_tuple)

            labels = tuple_labels_cell[i]

            boxObj = obj_bboxes_cell[i]
            boxSub = sub_bboxes_cell[i]

            if boxObj.shape[0] == 0 or boxObj.shape[1] == 0:
                box_entity_our = []
            else:
                c = np.minimum(boxObj[:,0:2],boxSub[:,0:2])
                d = np.maximum(boxObj[:,2:4],boxSub[:,2:4])

                box_entity_our = np.hstack((c,d))

            num_obj = len(labels[0])
            tp = np.zeros([1, num_obj])
            fp = np.zeros([1, num_obj])

            for j in range(num_obj):

                bbO = box_entity_our[j,:]
                ovmax = -math.inf
                kmax = -1

                for k in range(num_gt_tuple):
                    if np.linalg.norm(labels[:,j]-gt_tupLabel[:,k], 2) != 0:
                        continue
                    if gt_detected[k] > 0:
                        continue

                    bbgtO = gt_box_entity[k,:]
                    biO = [max([bbO[0],bbgtO[0]]), max([bbO[1],bbgtO[1]]), min([bbO[2],bbgtO[2]]), min([bbO[3],bbgtO[3]])]
                    iwO = float(biO[2]) - float(biO[0]) + 1
                    ihO = float(biO[3]) - float(biO[1]) + 1



                    if iwO > 0 and ihO > 0:
                        # compute overlap as area of intersection / area of union
                        uaO = (bbO[2]-bbO[0]+1)*(bbO[3]-bbO[1]+1) + (bbgtO[2]-bbgtO[0]+1)*(bbgtO[3]-bbgtO[1]+1) - iwO*ihO
                        ov = iwO * ihO / uaO


                        if ov >= gt_thr and ov > ovmax:
                            ovmax = ov
                            kmax = k
                            # count5 += 1
                if kmax >= 0:
                    tp[:,j] = 1
                    gt_detected[kmax] = 1

                else:
                    fp[:,j] = 1

            tp_cell.append(tp)
            fp_cell.append(fp)

        tp_all = np.asarray(tp_cell[0])
        fp_all = np.asarray(fp_cell[0])
        confs = np.asarray(tuple_confs_cell[0])

        for i in range(1,num_images):
            tp_all = np.hstack((tp_all, tp_cell[i]))
            fp_all = np.hstack((fp_all, fp_cell[i]))
            confs = np.vstack((confs, tuple_confs_cell[i]))


        ind = confs.argsort(axis=0)[::-1]
        ind = np.squeeze(ind,axis=1)
        tp_all = tp_all[:,ind]
        fp_all = fp_all[:,ind]


        tp = np.cumsum(tp_all,axis=1)
        fp = np.cumsum(fp_all,axis=1)
        recall = tp / num_pos_tuple
        precision = tp / (fp + tp)

        top_recall = recall[:,-1][0]
        top_recall = np.round(top_recall*100,decimals=1)
        ap = self.VOCap(recall, precision)
        ap = np.round(ap*100, decimals=1)

        return str(top_recall), str(ap)

    def top_recall_Relationship(self, Nre, candidates, groundtruth):
        tuple_confs_cell = candidates['scores']
        tuple_labels_cell = candidates['triplet']
        sub_bboxes_cell = candidates['sub_box']
        obj_bboxes_cell = candidates['obj_box']

        gt_tuple_label = groundtruth['triplet']
        gt_sub_bboxes = groundtruth['sub_box']
        gt_obj_bboxes = groundtruth['obj_box']
        # sort candidates by confidence scores
        num_images = len(gt_tuple_label)
        for i in range(num_images):
            ind = tuple_confs_cell[i].argsort(axis=0)[::-1]
            ind = np.squeeze(ind,axis=1)
            if len(ind) >= Nre:
                ind = ind[0:Nre]
            tuple_confs_cell[i] = tuple_confs_cell[i][ind]



            # print(f'sorted confidence: {tuple_confs_cell[i]}\n shape: {tuple_confs_cell[i].shape}')
            # print(f'after cut: {tuple_confs_cell[i]}\n shape: {tuple_confs_cell[i].shape}')
            # print(f'before: {tuple_labels_cell[i]}\n shape: {tuple_labels_cell[i].shape}')
            tuple_labels_cell[i] = tuple_labels_cell[i][:,ind]
            # print(f'after: {tuple_labels_cell[i]}\n shape: {tuple_labels_cell[i].shape}')
            obj_bboxes_cell[i] = obj_bboxes_cell[i][ind,:]
            # print(f'obj box: {obj_bboxes_cell[i]}\n shape: {obj_bboxes_cell[i].shape}')
            sub_bboxes_cell[i] = sub_bboxes_cell[i][ind,:]
            # print(f'sub box: {sub_bboxes_cell[i]}\n shape: {sub_bboxes_cell[i].shape}')
        
        num_pos_tuple = 0
        for i in range(num_images):
            num_pos_tuple += len(gt_tuple_label[i][0])

        tp_cell = []
        fp_cell = []

        gt_thr = 0.5
        
        # count1 = 0
        # count2 = 0
        # count3 = 0
        # count4 = 0
        # count5 = 0

        for i in range(num_images):
            gt_tupLabel = gt_tuple_label[i]
            gt_objBox = gt_obj_bboxes[i]
            gt_subBox = gt_sub_bboxes[i]

            num_gt_tuple = len(gt_tupLabel[0])
            gt_detected = np.zeros(num_gt_tuple)

            labels = tuple_labels_cell[i]
            boxObj = obj_bboxes_cell[i]
            boxSub = sub_bboxes_cell[i]

            num_obj = len(labels[0])
            tp = np.zeros([1, num_obj])
            fp = np.zeros([1, num_obj])

            for j in range(num_obj):

                bbO = boxObj[j,:]
                bbS = boxSub[j,:]
                # for i in range(4):
                #     bbO[i] = float(bbO[i])
                #     bbS[i] = float(bbS[i])
                ovmax = -math.inf
                kmax = -1

                for k in range(num_gt_tuple):
                    if np.linalg.norm(labels[:,j]-gt_tupLabel[:,k], 2) != 0:
                        # count1 += 1
                        # if i == 43: print(j+1,k+1,'case1',gt_detected, kmax)
                        continue
                    if gt_detected[k] > 0:
                        # if i == 43: print(j+1,k+1,'case2',gt_detected, kmax)
                        # count2 += 1

                        continue
                    # count3 += 1
                    # if i == 43: print(j+1,k+1,'case3',gt_detected, kmax)

                    bbgtO = gt_objBox[k,:]
                    bbgtS = gt_subBox[k,:]
                    # for i in range(4):
                    #     bbgtO[i] = float(bbgtO[i])
                    #     bbgtS[i] = float(bbgtS[i])
                    # print(f'i: {i}\nbbgtO: {bbgtO}\nbbO: {bbO}\nbbS: {bbS}\nbbgtS: {bbgtS}')
                    biO = [max([bbO[0],bbgtO[0]]), max([bbO[1],bbgtO[1]]), min([bbO[2],bbgtO[2]]), min([bbO[3],bbgtO[3]])]
                    # print(f'biO: {biO}')
                    iwO = float(biO[2]) - float(biO[0]) + 1
                    # print(f'iwO: {iwO}')
                    ihO = float(biO[3]) - float(biO[1]) + 1
                    # print(f'ihO: {ihO}')
                    biS = [max([bbS[0],bbgtS[0]]), max([bbS[1],bbgtS[1]]), min([bbS[2],bbgtS[2]]), min([bbS[3],bbgtS[3]])]
                    iwS = float(biS[2]) - float(biS[0]) + 1
                    ihS = float(biS[3]) - float(biS[1]) + 1
                    # print(f'biS: {biS}\niwS: {iwS}\nihS: {ihS}')
                    # print(f" biO: {biO}\n iwO: {iwO}\n ihO: {ihO}\n biS: {biS}\n iwS: {iwS}\n ihS: {ihS}\n")
                    if iwO > 0 and ihO > 0 and iwS > 0 and ihS > 0:
                        # compute overlap as area of intersection / area of union
                        # print(f'iwO: {iwO} ihO: {ihO} iwS: {iwS} ihS: {ihS}')
                        uaO = (bbO[2]-bbO[0]+1)*(bbO[3]-bbO[1]+1) + (bbgtO[2]-bbgtO[0]+1)*(bbgtO[3]-bbgtO[1]+1) - iwO*ihO
                        ovO = iwO * ihO / uaO
                        # print(f'uaO: {uaO}\novO: {ovO}')
                        uaS = (bbS[2]-bbS[0]+1) * (bbS[3]-bbS[1]+1) + (bbgtS[2]-bbgtS[0]+1) * (bbgtS[3]-bbgtS[1]+1) - (iwS*ihS)
                        ovS = iwS * ihS / uaS
                        ov = min([ovO,ovS])

                        # print(f'uaO: {uaO}\novO: {ovO}\nuaS: {uaS}\novS: {ovS}\nov: {ov}\n')
                        # print(f'ov: {ov}')
                        # count4 += 1
                        if ov >= gt_thr and ov > ovmax:
                            ovmax = ov
                            kmax = k
                            # count5 += 1
                if kmax >= 0:
                    tp[:,j] = 1
                    gt_detected[kmax] = 1

                else:
                    fp[:,j] = 1

            tp_cell.append(tp)
            fp_cell.append(fp)
        # print(count1, count2, count3, count4, count5)
        # print(f'tp_cell: {tp_cell}\ntype:{type(tp_cell)}')
        # print(f'fp_cell: {fp_cell}\ntype:{type(fp_cell)}')
        tp_all = np.asarray(tp_cell[0])
        fp_all = np.asarray(fp_cell[0])
        confs = np.asarray(tuple_confs_cell[0])
        # print(f'tp_all: {tp_all} type: {type(tp_all)}, shape: {tp_all.shape}\nfp_all: {fp_all} type: {type(fp_all)} shape: {fp_all.shape}\nconfs: \n{confs} type: {type(confs)} shape: {confs.shape}')
        for i in range(1,num_images):
            tp_all = np.hstack((tp_all, tp_cell[i]))
            fp_all = np.hstack((fp_all, fp_cell[i]))
            confs = np.vstack((confs, tuple_confs_cell[i]))
            # print(f'tp_all: {tp_all} type: {type(tp_all)}, shape: {tp_all.shape}\nfp_all: {fp_all} type: {type(fp_all)} shape: {fp_all.shape}\nconfs: \n{confs} type: {type(confs)} shape: {confs.shape}')


        ind = confs.argsort(axis=0)[::-1]
        ind = np.squeeze(ind,axis=1)
        tp_all = tp_all[:,ind]
        fp_all = fp_all[:,ind]


        tp = np.cumsum(tp_all,axis=1)
        fp = np.cumsum(fp_all,axis=1)
        recall = tp / num_pos_tuple
        precision = tp / (fp + tp)

        top_recall = recall[:,-1][0]
        top_recall = np.round(top_recall*100,decimals=1)
        ap = self.VOCap(recall, precision)
        ap = np.round(ap*100, decimals=1)

        return str(top_recall), str(ap)

    def VOCap(self, rec, prec):
        mrec = np.hstack((np.hstack((np.zeros((1,1)), rec)), np.ones((1,1))))
        # print(f'mrec: {mrec}\nshape: {mrec.shape}')
        mpre = np.hstack((np.hstack((np.zeros((1,1)), prec)), np.zeros((1,1))))
        # print(f'mpre: {mpre}\nshape: {mpre.shape}')
        for i in range(mpre.shape[1]-2,-1,-1):
            # print(f'i: {i}')
            # print(f'i: {mpre[:,i]}\ni+1: {mpre[:,i+1]}')
            mpre[:,i] = max(mpre[:,i], mpre[:,i+1])
            # print(f'updated: {mpre[:,i]}')

        # print(f'mpre: {mpre}\nshape: {mpre.shape}')
        # print(f'mrec: {mrec}\nshape: {mrec.shape}')
        ind = np.where(mrec[:,1:] != mrec[:,:-1])[1]+1
        # print(f'ind: {ind}, shape: {ind.shape}')
        ap = np.sum((mrec[:,ind]-mrec[:,ind-1])*mpre[:,ind],axis=1)[0]
        return ap

    def format_testdata_recall(self, pairs, scores, annotations):
        # format outputs to match evaluation code of Lu16
        # load zeroshot setting
        if self.zeroshot:
            ind_zeroshot = scipy.io.loadmat(os.path.join(self.DEFAULT_DATAROOT, self.dataset, 'test', 'annotated', 'ind_zeroshot.mat'))['ind_zeroshot']
            ind_zeroshot = ind_zeroshot - 1
            ind_zeroshot = np.squeeze(ind_zeroshot, axis=1)
            # print(f'new index: {ind_zeroshot}, shape: {ind_zeroshot.shape}')
            annotations = self.select(annotations, ind_zeroshot)
        # image number and counts
        images = self.get_images(annotations)
        num_images = len(images)
        # print(images, type(images), len(images), num_images)
        # format groundtruth
        groundtruth = {'triplet':[], 'sub_box':[], 'obj_box':[]}
        for i in range(num_images):
            im_id = images[i]
            idx = self.get_pairs_in_image(annotations, im_id)
            groundtruth['triplet'].append(np.asarray([annotations['sub_cat'][idx], annotations['rel_cat'][idx], annotations['obj_cat'][idx]]))
            groundtruth['sub_box'].append(np.asarray(annotations['subject_box'][idx,:]))
            groundtruth['obj_box'].append(np.asarray(annotations['object_box'][idx,:]))
        # format candidates
        candidates = {'scores':[], 'triplet':[], 'sub_box':[], 'obj_box':[], 'rel_id':[], 'im_id':[]}
        for i in range(num_images):
            im_id = images[i]
            idx = self.get_pairs_in_image(pairs, im_id)
            candidates['rel_id'].append(np.asarray(pairs['rel_id'][idx]))
            candidates['im_id'].append(np.asarray(pairs['im_id'][idx]))
            candidates['sub_box'].append(np.asarray(pairs['subject_box'][idx,:]))
            candidates['obj_box'].append(np.asarray(pairs['object_box'][idx,:]))
            # get top 1 prediction for each pair of boxes
            scores_pred = np.zeros([len(idx),1])
            rel_pred = np.zeros([len(idx),1])
            for k in range(len(idx)):
                scores_pred[k] = max(scores[idx[k],:])
                rel_pred[k] = np.argmax(scores[idx[k],:])

            candidates['scores'].append(np.asarray(scores_pred))
            candidates['triplet'].append(np.asarray([pairs['sub_cat'][idx], rel_pred+1, pairs['obj_cat'][idx]]))

        return candidates, groundtruth

    def get_pairs_in_image(self, pairs, im_id):
        # return index of pairs matching the im_id
        idx = np.where(pairs['im_id'] == im_id)[0]

        return idx

    def get_images(self, pairs):
        # return list of images in pairs

        return np.unique(pairs['im_id'])

    def select(self, pairs, idx):
        # select only rows in pairs.mat matching the index
        for key in pairs.keys():
            pairs[key] = pairs[key][idx]

        return pairs

    def predict(self):
        # use model to predict scores, and also return pairs.mat and gt_annotations

        # load annotations
        annotations = self.get_full_annotations()

        # load candidatespairs
        pairs = self.load_candidates(self.candidatespairs)

        testloader = self.create_testloader()
        prediction = self.calc_scores(testloader)
        N, C = prediction.shape

        if self.use_languagescores:
            # print(f'Computing language scores...')
            features_language = self.load_languagefeatures(pairs)

        object_scores = np.ones(prediction.shape)
        if self.use_objectscores:
            # print(f'Loading the object scores...')
            datafolder = os.path.join(self.DEFAULT_DATAROOT, self.dataset, self.split, self.candidatespairs)
            object_scores = self.load_objectscores(datafolder, pairs)
            # add scores to each column
            a = object_scores.reshape([-1,1])
            b = np.ones([1,C])
            object_scores = object_scores.reshape([-1,1])*np.ones([1,C])
            prediction = prediction + self.alpha_objects*object_scores

        return (pairs, prediction, annotations)

    def load_languagefeatures(self, pairs):
        obj2vec = scipy.io.loadmat(os.path.join(self.DEFAULT_DATAROOT, self.dataset,'obj2vec.mat'))
        vocab_objects = scipy.io.loadmat(os.path.join(self.DEFAULT_DATAROOT, self.dataset,'vocab_objects.mat'))['vocab_objects']
        print(obj2vec)
        N = pairs['im_id'].shape[0]
        X = np.zeros([N, 600])

        sub_cat = pairs['sub_cat'][0]-1
        print(sub_cat)
        obj_cat = pairs['obj_cat'][0]-1
        print(obj_cat)
        print(type(vocab_objects))
        sub_cat = str(vocab_objects[sub_cat][0,0][0])
        print(sub_cat)
        obj_cat = str(vocab_objects[obj_cat][0,0][0])
        print(obj_cat)
        X[0,:] = [obj2vec[sub_cat], obj2vec[obj_cat]]


        return None

    def load_objectscores(self, datafolder, pairs):
        N = pairs['im_id'].shape[0]
        scores = np.zeros((N))
        objectscores = scipy.io.loadmat(os.path.join(datafolder, 'objectscores.mat'))['scores']

        for j in range(N):
            sub_id = pairs['sub_id'][j]
            sub_cat = pairs['sub_cat'][j]
            idx = objectscores[:,0] == sub_id
            sub_scores = np.squeeze(objectscores[idx,1:],axis=0)
            obj_id = pairs['obj_id'][j]
            obj_cat = pairs['obj_cat'][j]
            idx = objectscores[:,0] == obj_id
            obj_scores = np.squeeze(objectscores[idx,1:],axis=0)
            scores[j] = sub_scores[sub_cat] + obj_scores[obj_cat]

        return scores

    def create_testloader(self):
        # print(f"loading dataset...{self.candidatespairs}")
        testset = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT, self.dataset), self.split, pairs=self.candidatespairs, klass=BasicTestingExample)
        testloader = DataLoader(testset, batch_size=100, num_workers=4)
        return testloader

    def calc_scores(self, testloader):
        # print(f"calculating scores...")
        prediction = None
        for testbatch in testloader:
            with torch.no_grad():
                scores = self.model(torch.autograd.Variable(testbatch['X'].cuda()))
            if type(prediction) is torch.Tensor:
                prediction = torch.cat((prediction, scores.cpu()),0)
            else:
                prediction = scores.cpu()
            scores = None

        return prediction.data.numpy()

    def convert_to_dict(self, pairs):
        results = {}
        results['im_id'] = pairs['im_id']
        results['rel_id'] = pairs['rel_id']
        results['sub_id'] = pairs['sub_id']
        results['obj_id'] = pairs['obj_id']
        results['sub_cat'] = pairs['sub_cat']
        try:
            results['rel_cat'] = pairs['rel_cat']
        except:
            pass
        results['obj_cat'] = pairs['obj_cat']
        results['subject_box'] = pairs['subject_box']
        results['object_box'] = pairs['object_box']
        # print(results.keys())

        return results

    def get_full_annotations(self):
        # loads pairs.mat from vrd-dataset/test/annotated in...dict?
        annotated = scipy.io.loadmat(os.path.join(self.DEFAULT_DATAROOT, self.dataset, self.split, 'annotated', 'pairs.mat'))['pairs'][0,0]
        annotated = self.convert_to_dict(annotated)
        # to access: annotated['rel_id']
        return annotated

    def load_candidates(self, candidatespairs):
        # loads pairs.mat from vrd-dataset/test/Lu-candidates in...dict?
        candidatespairs = scipy.io.loadmat(os.path.join(self.DEFAULT_DATAROOT, self.dataset, self.split, candidatespairs, 'pairs.mat'))['pairs'][0,0]
        candidatespairs = self.convert_to_dict(candidatespairs)
        # to access: candidates['sub_cat']
        return candidatespairs

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
        # settings = ['annotated']
        # save the predictions
        _testset = {}
        testdata = {}
        for setting in settings:
            # print(f'loading datasets...{setting}')
            # initialize dataloaders for both testset
            _testset[setting] = dset.Dataset(os.path.join(self.DEFAULT_DATAROOT,self.dataset), 'test', pairs=setting, klass=BasicTestingExample)
            testdata[setting] = DataLoader(_testset[setting], 
                                batch_size=100, 
                                num_workers=4) 
            # run prediction for each and save in .mat
            # print(f'calculating scores...')
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
            # print(f"size of {setting} is: {self.prediction[setting].shape}")
            scores_np = self.prediction[setting].data.numpy()
            mydict = {'scores':scores_np}
            # sanity check
            # print(mydict['scores'])
            # print(f"from dataset: {setting}\nshape: {mydict['scores'].shape}")
            # save to unrel folder as (ex) "/annotated_<dim>_<id>.mat"
            # print(f'saving .mat files...{setting}')
            scipy.io.savemat(os.path.join(self.SCORES_PATH, f'{setting}.mat'), mydict)
            # print(f"{setting}.mat file is saved")
            self.prediction = {}

        # use subprocess to run
        print('starting matlab...')

        rc = Popen(f"{self.UNREL_PATH}/run_recall.sh baseline full {self.SCORES_PATH}", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        rc_out = str(rc.stdout.read(), 'utf-8')
        # rc = Popen(f"{self.UNREL_PATH}/run_recall.sh baseline full {self.SCORES_PATH}", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, bufsize=1)
        # for line in iter(rc.stdout.readline,b''):
        #     print(line)
        # rc.stdout.close()
        # rc.wait()
        # return {}

        results = []
        data = rc_out.split('\n')

        for line in data[-8:-1]:
            data = line.split()[-1]
            if data[0] != 'z':
                results.append(line.split()[-1])

        recalls = {}
        recalls['seen_predicate'] = results[0]
        recalls['seen_phrase'] = results[1]
        recalls['seen_relationship'] = results[2]
        recalls['unseen_predicate'] = results[3]
        recalls['unseen_phrase'] = results[4]
        recalls['unseen_relationship'] = results[5]

        return recalls

if __name__ == '__main__':
    evalr = RecallEvaluator('/home/SSD2/tyler-data/unrel/data',"/home/tylerjan/code/vrd/unrel","/home/tylerjan/code/vrd/unrel/scores")
    










