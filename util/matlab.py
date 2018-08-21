#!/usr/bin/env python

# For most uses, you want the unrel_data module in this directory. This is a
# lower-level module which is probably less efficient when running in a
# system, but it can be of use for offline work, such as fetching a list of
# object names (cats).

import os
import scipy.io

DEFAULT_DS_DIR = '/home/markham/data/unrel/data/vrd-dataset'
DEFAULT_VOCAB_OBJ_FILE = os.path.join( DEFAULT_DS_DIR, 'vocab_objects.mat' )
DEFAULT_VOCAB_PRED_FILE = os.path.join( DEFAULT_DS_DIR, 'vocab_predicates.mat' )
DEFAULT_TEST_FILENAMES_FILE = os.path.join( DEFAULT_DS_DIR, 'image_filenames_test.mat' )
DEFAULT_TRAIN_FILENAMES_FILE = os.path.join( DEFAULT_DS_DIR, 'image_filenames_train.mat' )

class ObjectSet(object):
	def __init__(self, fpath=None, split='train', style='annotated'):
		self.objects = scipy.io.loadmat(fpath or os.path.join(DEFAULT_DS_DIR, split, style, 'objects.mat'))['objects'][0]
		self.tuples = list(zip(*[self.objects[n][0] for n in self.objects.dtype.names]))
		self.min = self.objects['im_id'][0].min()
		self.max = self.objects['im_id'][0].max()
		self.dic = {k:[] for k in range(self.min, self.max+1)}
		for o in self.tuples:
			self.dic[o[0][0]].append(o)
	def get(self, im_id=None):
		return self.tuples if im_id is None else self.dic[im_id]

# Count the length of a feature in a .mat file containing extracted features for one or more images
def count_features(fpath, key='features', axis=1):
	features = scipy.io.loadmat(fpath)[key]
	return features.shape[axis]

def get_object_cats(fpath=None):
	return scipy.io.loadmat(fpath or DEFAULT_VOCAB_OBJ_FILE)['vocab_objects']

def get_predicate_cats(fpath=None):
	return scipy.io.loadmat(fpath or DEFAULT_VOCAB_PRED_FILE)['vocab_predicates']

def get_test_images(fpath=None):
	return scipy.io.loadmat(fpath or DEFAULT_TEST_FILENAMES_FILE)['image_filenames'][0]

def get_train_images(fpath=None):
	return scipy.io.loadmat(fpath or DEFAULT_TRAIN_FILENAMES_FILE)['image_filenames'][0]

if __name__ == '__main__':
	import sys
	for fpath in sys.argv[1:]:
		print(count_features(fpath))
	with open('objects.tsv','w') as f:
		for i, obj in enumerate(get_object_cats()):
			f.write('%d\t%s\n' % (i, obj[0][0]))
	with open('predicates.tsv','w') as f:
		for i, pred in enumerate(get_predicate_cats()):
			f.write('%d\t%s\n' % (i, pred[0][0]))
	with open('test_images.tsv','w') as f:
		for i, im in enumerate(get_test_images()):
			f.write('%d\t%s\n' % (i, im[0]))
	with open('train_images.tsv','w') as f:
		for i, im in enumerate(get_train_images()):
			f.write('%d\t%s\n' % (i, im[0]))
	