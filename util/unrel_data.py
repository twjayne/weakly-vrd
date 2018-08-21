
# Usage:
# 	fetcher = Builder().split('train', 'annotated')
#   bounding_boxes = fetcher.bbs()
# 	for entity in fetcher.entities:
# 		print(entity.name(), entity.bb())


import sys, os
import numpy as np
import scipy.io
import skimage.io
import PIL
import matplotlib.pyplot as plt

_LEGAL_SPLITS = ['train', 'test']
_LEGAL_SETS   = ['annotated', 'candidates']

DEFAULT_UNRELDIR = '/home/SSD2/markham-data/unrel'
DEFAULT_SGDIR    = '/home/SSD2/markham-data/sg_dataset'

# Auxilliary class only used for building a _Fetcher
class Builder(object):
	def __init__(self, unreldir=None, sgdir=None):
		self.meta_path_prefix = os.path.join(unreldir or DEFAULT_UNRELDIR, 'data/vrd-dataset')
		self.data_path_prefix = sgdir or DEFAULT_SGDIR
		assert os.path.exists(self.meta_path_prefix)
		assert os.path.exists(self.data_path_prefix)

	def split(self, split, which):
		return _Fetcher(split, which, self.meta_path_prefix, self.data_path_prefix)

# Fetch dataset information: image-level or higher
class _Fetcher(object):
	def __init__(self, split, which, meta_path_prefix, data_path_prefix):
		assert split in _LEGAL_SPLITS
		assert which in _LEGAL_SETS
		# Meta data
		objects_file = os.path.join(meta_path_prefix, split, which, 'objects.mat')
		self.meta = scipy.io.loadmat(objects_file)['objects'][0]
		im_names_file = os.path.join(meta_path_prefix, 'image_filenames_%s.mat' % split)
		self.fnames = [x[0] for x in scipy.io.loadmat(im_names_file)['image_filenames'][0]]
		obj_names_file = os.path.join(meta_path_prefix, 'vocab_objects.mat')
		self.obj_names = [x[0][0] for x in scipy.io.loadmat(obj_names_file)['vocab_objects']]
		pred_names_file = os.path.join(meta_path_prefix, 'vocab_predicates.mat')
		self.pred_names = [x[0][0] for x in scipy.io.loadmat(pred_names_file)['vocab_predicates']]
		# Data (image files)
		self.im_dir = os.path.join(data_path_prefix, 'sg_%s_images' % split)

	def _idxs(self, im_id):
		return np.where(self.meta['im_id'][0] == im_id)[0]

	def entities(self, im_id):
		return [Entity(idx, self) for idx in self._idxs(im_id)]

	def bbs(self, im_id):
		return [ent.bb() for ent in self.entities(im_id)]

	def fname(self, im_id):
		return self.fnames[im_id - 1]

	def image(self, im_id_or_fname):
		if type(im_id_or_fname) == str:
			fname = im_id_or_fname
		elif type(im_id_or_fname) == int:
			fname = self.fname(im_id_or_fname)
		else:
			raise Exception('Illegal argument type. Expected int or str. Got %s' % type(im_id_or_fname))
		return PIL.Image.open(os.path.join(self.im_dir, fname))

	def show(self, im_id):
		image = self.image(im_id)
		plt.imshow(image)

	def show_bbs(self, im_id):
		image = self.image(im_id)
		for ent in self.entities(im_id):
			bb = ent.bb()
			print(ent.name(), bb)
			plt.figure(num=ent.name())
			plt.imshow(ent.crop(image))

	def object_cats(fpath=None):
		arrays = scipy.io.loadmat(fpath or DEFAULT_VOCAB_OBJ_FILE)['vocab_objects'][0]
		return [x[0] for x in arrays]

	def predicate_cats(fpath=None):
		arrays = scipy.io.loadmat(fpath or DEFAULT_VOCAB_PRED_FILE)['vocab_predicates'][0]
		return [x[0] for x in arrays]

# Struct to hold info for a single candidate: BB, im_id, obj_cat, etc
class Entity(object):
	def __init__(self, idx, fetcher):
		self.fetcher = fetcher
		meta = self.fetcher.meta
		for key in meta.dtype.names:
			val = meta[key][0][idx]
			if len(val) == 1: val = val[0]
			self.__setattr__(key, val)
	def name(self):
		return self.fetcher.obj_names[self.obj_cat - 1]
	def bb(self):
		bb = self.object_box
		return (bb[0], bb[1], bb[2], bb[3]) # x1, y1, x2, y2
	# Return a crop of the image, containing only this entity's bounding box. Imdata should be a PIL Image
	def crop(self, imdata):
		assert isinstance(imdata, PIL.ImageFile.ImageFile), type(imdata)
		bb = self.bb()
		return imdata.crop(bb)

if __name__ == '__main__':
	fetcher = Builder().split('train', 'annotated')
	def show_all(im_id):
		fetcher.show_bbs(im_id)
		plt.figure('Image %d' % im_id); fetcher.show(im_id)
		plt.show()
	show_all(3)
	# I see that the VRD dataset out of Stanford poses some challenges.
	# 'hat' on the skier in Image 9 is just crazy. You can tell by context, but not by the BB.
	# Same always goes for 'glasses'
	# A lot of the objects for Image 10 are like this.
	# 'street' is wrong in Image 11 and 3, at least.
	# We have a group of people with a lot of space in the middle classified as 'person'. Later we see the same thing classified as 'grass'. It's hard.
