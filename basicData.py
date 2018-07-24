from torch.utils.data.dataset import Dataset
import scipy.io
import os
import numpy as np

class BasicData(Dataset):
    def __init__(self, dataroot, split, **kwargs):
        # dataroot is the path to data
        self.rootdir = dataroot
        sup = kwargs.get('pairs', 'annotated')
        data = scipy.io.loadmat(os.path.join(dataroot, split, sup, 'pairs.mat'))['pairs'][0][0]
        # test = scipy.io.loadmat(os.path.join(dataroot, 'test/annotated/pairs.mat'))['pairs'][0][0] # test data to be separated into different datasets
        header = ['im_id','rel_id','sub_id','obj_id','sub_cat','rel_cat','obj_cat','subject_box','object_box']
        pairs = {}
        # test_pairs = {}
        for i in range(9):
            pairs[header[i]] = data[i]
            # test_pairs[header[i]] = test[i]
        self.pairs = pairs
        # self.test_pairs = test_pairs
        self.objects = scipy.io.loadmat(os.path.join(dataroot, 'vocab_objects.mat'))['vocab_objects']
        self.predicates = scipy.io.loadmat(os.path.join(dataroot, 'vocab_predicates.mat'))['vocab_predicates']
        self.rel_cat = None
        self.sub_cat = None
        self.obj_cat = None

    def __len__(self):
        # length of dataset, equivalent to number of triplets in the training set
        return len(self.pairs['rel_id'])

        # for testing set
        # return len(self.test_pairs['rel_id'])

    def loadSpatial(self, index):
        i = index
        im_id = int(self.pairs['im_id'][i][0])
        spatial_path = os.path.join(self.rootdir, 'train/annotated/features/spatial-full', '%d.mat' % im_id)
        rel_id = int(self.pairs['rel_id'][i][0])
        data = scipy.io.loadmat(spatial_path)['spatial']
        rels = {}
        for rel in data:
            rels[int(rel[0])] = rel[1:]
        return rels[rel_id]

    def loadAppearance(self, index):
        i = index

        im_id = int(self.pairs['im_id'][i][0])
        app_path = os.path.join(self.rootdir, 'train/annotated/features/appearance-full', '%d.mat' % im_id)
        sub_id = int(self.pairs['sub_id'][i][0])
        app = scipy.io.loadmat(app_path)['appearance']
        apps = {}
        for o in app:
            apps[int(o[0])] = o[1:]
        sub = apps[sub_id]

        obj_id = int(self.pairs['obj_id'][i][0])
        obj = apps[obj_id]
        return np.concatenate((sub,obj),axis=0)

    def loadTriplet(self, index):
        return self.loadSubject(index), self.loadPredicate(index), self.loadObject(index)

    def loadPredicate(self, index):
        i = index

        rel_id = int(self.pairs['rel_id'][i][0])
        rel_cat = self.pairs['rel_cat'][rel_id-1][0]-1 # reference number in matlab indexed from 1, but python indexd from 0 so when running in python need to -1
        self.rel_cat = rel_cat
        # print(f'rel_id: {rel_id}, rel_cat: {rel_cat}')
        return self.predicates[rel_cat][0][0]


    def loadSubject(self, index):
        i = index
        
        sub_id = int(self.pairs['sub_id'][i][0])
        sub_cat = self.pairs['sub_cat'][sub_id-1][0]-1
        self.sub_cat = sub_cat
        return self.objects[sub_cat][0][0]

    def loadObject(self, index, ):
        i = index
        
        obj_id = int(self.pairs['obj_id'][i][0])
        obj_cat = self.pairs['obj_cat'][obj_id-1][0]-1
        self.obj_cat = obj_cat
        return self.objects[obj_cat][0][0]

    def __getitem__(self, index):
        output = {
            'predicate': self.loadPredicate(index),
            'spatial': self.loadSpatial(index),
            'appearance': self.loadAppearance(index),
            'subject': self.loadSubject(index),
            'object': self.loadObject(index),
            'triplet': self.loadTriplet(index),
            'predicate_index': self.rel_cat,
            'subject_index': self.sub_cat,
            'object_index': self.obj_cat
        }
        return output

# if __name__ == "__main__":
#     print('testing...init')
#     dataroot = '/Users/Linen/GoogleDrive/data/unrel/vrd-dataset'
#     myData = BasicData(dataroot,'train', pairs='annotated')
#     # print(myData)
#     # print(type(myData.objects))
#     # print(type(myData.predicates))
#     # test = myData.test_pairs
#     # print('testing...len')
#     # print('traing set length:', len(myData))
#     print('testing...predicate')
#     pred = myData.loadPredicate(0)
#     print(myData.loadPredicate(0))
#     subject = myData.loadSubject(0)
#     print(f'testing...subject\n{subject}')
#     obj = myData.loadObject(0)
#     print(f'testing...object\n{obj}')
#     triplet = myData.loadTriplet(0)
#     print(f'triplets:......<{triplet[0]},{triplet[1]},{triplet[2]}>')
#     print('testing...spatial')
#     spatial = myData.loadSpatial(0)
#     shape = spatial.shape
#     print(f'{shape}')
#     print('testing...appearance')
#     appearance = myData.loadAppearance(1)
#     length = appearance.shape
#     print(f'{length}')
#     print('testing...__getitem__')
#     print(myData[0])