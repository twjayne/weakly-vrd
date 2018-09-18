__import__(__package__ or '__init__')
import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.utils.data.dataset
import scipy.io
import os
import re
import numpy as np
import util.config as config
if __name__ == '__main__':
    from example import BasicExample
else:
    from dataset.example import BasicExample
import pdb


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataroot, split, **kwargs):
        self.split = split
        self.example_klass = kwargs.get('klass', BasicExample)
        self.name = kwargs.get('name', None)
        self.sup = kwargs.get('pairs', 'candidates')
        self.supervision = kwargs.get('supervision', 'weak')
        self.rootdir = dataroot # dataroot is the path to data
        self.scenic_features_dir = kwargs.get('scenic', None)
        self.pairs = scipy.io.loadmat(os.path.join(dataroot, split, self.sup, 'pairs.mat'))['pairs'][0,0]
        for key in self.__keys():
            if key[-3:] == '_id': self.pairs[key] = self.pairs[key].astype(np.int32).reshape(-1)
            if key[-4:] == '_cat': self.pairs[key] -= 1
            if key[-4:] == '_box': self.pairs[key] = torch.from_numpy(self.pairs[key].astype(np.int32))
        self.objects = scipy.io.loadmat(os.path.join(dataroot, 'vocab_objects.mat'))['vocab_objects']
        self.predicates = scipy.io.loadmat(os.path.join(dataroot, 'vocab_predicates.mat'))['vocab_predicates']
        self.__file_cache = {}

    def __len__(self):
        return len(self.pairs['rel_id'])

    def __getitem__(self, i):
        copyvals = { k: self.pairs[k][i] for k in self.__keys() }

        return self.example_klass(copyvals, 
            self._spatial_features(i),
            self._appearance_features(i))

    def __keys(self):
        return self.pairs.dtype.names

    def _file_cache_get(self, mkey, fpath, filekey=None):
        dkey = '%s<v>' % mkey
        if self.__file_cache.get(mkey, None) != fpath:
            self.__file_cache[mkey] = fpath
            self.__file_cache[dkey] = torch.from_numpy(scipy.io.loadmat(fpath)[filekey or mkey])
        return self.__file_cache[dkey]

    # @arg key is 'spatial' or 'appearance'
    # @arg i is the index in self.pairs
    # @arg ent_id is the id of the sub or obj
    def _load_matlab_features(self, key, i, ent_id):
        img_i = self.pairs['im_id'][i]
        selected_dir = '%s/%s/features/%s-%s' % (self.split, self.sup, key, self.supervision)
        path = os.path.join(self.rootdir, selected_dir, '%d.mat' % img_i)
        rows = self._file_cache_get(key, path)
        # The first column indicates the id
        for row in rows:
            if int(row[0]) == ent_id: break
        assert int(row[0]) == ent_id
        return row[1:]

    def _spatial_features(self, i):
        return self._load_matlab_features('spatial', i, self.pairs['rel_id'][i])

    def _appearance_features(self, i):
        sub = self._load_matlab_features('appearance', i, self.pairs['sub_id'][i])
        obj = self._load_matlab_features('appearance', i, self.pairs['obj_id'][i])
        return np.concatenate((sub, obj))

    def triplets(self):
        keys = ('sub_cat', 'rel_cat', 'obj_cat')
        return [tuple([self.pairs[key][i][0] for key in keys]) for i in range(len(self))]

class DatasetWithScenicFeatures(Dataset):
    def __init__(self, rootdir, split, scenic_features_dir, image_name_map, **kwargs):
        super(DatasetWithScenicFeatures, self).__init__(rootdir, split, **kwargs)
        self.scenic_features_dir = scenic_features_dir
        self.image_names = scipy.io.loadmat(image_name_map)['image_filenames']

    def _scenic_features(self, i):
        img_idx = self.pairs['im_id'][i] - 1
        name = re.sub('\..+$', '.mat', self.image_names[0,img_idx][0])
        path = os.path.join(self.scenic_features_dir, name)
        features = self._file_cache_get('scenic', path, 'features').reshape(-1).type(config.dtype)
        return features

# This class is used to treat a subset of a given Dataset ('parent') as its own dataset
# without duplicating the data. This is useful for segregating a main Dataset into 
# 'zeroshot' and 'unseen' subsets for the purposes of taking distinct measurements from 
# each.
class Subset(torch.utils.data.dataset.Dataset):
    def __init__(self, parent, indices, name=None):
        self.parent = parent
        self.indices = indices
        self.name = name
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.parent[self.indices[i]]


if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'test'
    curdir = os.path.split(os.path.realpath(__file__))[0]
    parent = os.path.split(curdir)[0]
    datadir = os.path.join(parent, 'data/vrd-dataset')
    
    # Vanilla
    ds = Dataset(datadir, split, pairs='annotated', supervision='weak')
    print(len(ds))
    for i in range(len(ds)):
        ex = ds[i]
        print('%5d - %4d %2d %4d' % (i, ex.sub_cat, ex.rel_cat, ex.obj_cat))
        if i > 4: break

    # Dataset with scenic features
    scenic_features_dir = os.path.join(os.path.expanduser('~'), 'data/sg_dataset/scenic/pca-1000-resnet18-f/test')
    image_name_map = os.path.join(os.path.expanduser('~'), 'data/unrel/data/vrd-dataset/image_filenames_test.mat')
    ds = DatasetWithScenicFeatures(datadir, split, scenic_features_dir, image_name_map, pairs='annotated', supervision='weak')
    for i in range(15):
        print(ds._scenic_features(i)[0,:2], ds._appearance_features(i)[:2])
