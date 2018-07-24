import torch.utils.data.dataset
import scipy.io
import os
import numpy as np
from example import Example
import pdb

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataroot, split, **kwargs):
        self.split = split
        self.sup = kwargs.get('pairs', 'candidates')
        self.supervision = kwargs.get('supervision', 'weak')
        self.rootdir = dataroot # dataroot is the path to data
        self.pairs = scipy.io.loadmat(os.path.join(dataroot, split, self.sup, 'pairs.mat'))['pairs'][0,0]
        self.objects = scipy.io.loadmat(os.path.join(dataroot, 'vocab_objects.mat'))['vocab_objects']
        self.predicates = scipy.io.loadmat(os.path.join(dataroot, 'vocab_predicates.mat'))['vocab_predicates']
        self.__file_cache = {}

    def __len__(self):
        return len(self.pairs['rel_id'])

    def __getitem__(self, i):
        output = {}
        for key in self.__keys():
            output[key] = self.pairs[key][i]
            if key[-4:] == '_cat': output[key] -= 1
        return Example(output, 
            self._spatial_features(i),
            self._appearance_features(i))

    def __keys(self):
        return self.pairs.dtype.names

    def _file_cache_get(self, mkey, fpath):
        dkey = '%s<v>' % mkey
        if self.__file_cache.get(mkey, None) != fpath:
            self.__file_cache[mkey] = fpath
            self.__file_cache[dkey] = scipy.io.loadmat(fpath)[mkey]
        return self.__file_cache[dkey]

    # @arg key is 'spatial' or 'appearance'
    # @arg i is the index in self.pairs
    # @arg ent_id is the id of the sub or obj
    def _load_matlab_features(self, key, i, ent_id):
        img_i = self.pairs['im_id'][i][0]
        selected_dir = '%s/%s/features/%s-%s' % (self.split, self.sup, key, self.supervision)
        path = os.path.join(self.rootdir, selected_dir, '%d.mat' % img_i)
        rows = self._file_cache_get(key, path)
        # The first column indicates the id
        for row in rows:
            if row[0] == ent_id: break
        assert row[0] == ent_id
        return row[1:]

    def _spatial_features(self, i):
        return self._load_matlab_features('spatial', i, self.pairs['rel_id'][i])

    def _appearance_features(self, i):
        sub = self._load_matlab_features('appearance', i, self.pairs['sub_id'][i])
        obj = self._load_matlab_features('appearance', i, self.pairs['obj_id'][i])
        return np.concatenate((sub, obj))

if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'test'
    curdir = os.path.split(os.path.realpath(__file__))[0]
    parent = os.path.split(curdir)[0]
    datadir = os.path.join(parent, 'data/vrd-dataset')
    ds = Dataset(datadir, split, pairs='annotated', supervision='weak')
    print(len(ds))
    for i in range(len(ds)):
        ex = ds[i]
        print('%5d - %4d %2d %4d' % (i, ex.sub_cat, ex.rel_cat, ex.obj_cat))
        if i > 4: break

