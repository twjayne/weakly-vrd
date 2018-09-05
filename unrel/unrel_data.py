
# Usage:
# 	fetcher = Builder().split('train', 'annotated')
#   bounding_boxes = fetcher.bbs()
# 	for entity in fetcher.entities:
# 		print(entity.name(), entity.bb())


import sys, os
import numpy as np
import torch
import scipy.io
import skimage.io
import PIL
import matplotlib.pyplot as plt

import pdb

_LEGAL_SPLITS = ['train', 'test']
_LEGAL_PAIRS  = ['annotated', 'candidates']
_LEGAL_SUPER  = ['full', 'weak']

DEFAULT_UNRELDIR = '/home/SSD2/markham-data/unrel'
DEFAULT_SGDIR    = '/home/SSD2/markham-data/sg_dataset'

squeeze = lambda x: x.item() if isinstance(x,np.ndarray) and len(x) == 1 else x

# Auxilliary class only used for building a _Fetcher
class Builder(object):
	def __init__(self, unreldir=None, sgdir=None, supervision='full'):
		self.meta_path_prefix = os.path.join(unreldir or DEFAULT_UNRELDIR, 'data/vrd-dataset')
		self.data_path_prefix = sgdir or DEFAULT_SGDIR
		self.supervision = supervision
		assert os.path.exists(self.meta_path_prefix)
		assert os.path.exists(self.data_path_prefix)

	def split(self, split, pairs):
		return _Fetcher(split, pairs, self.meta_path_prefix, self.data_path_prefix, self.supervision)

# Fetch dataset information: image-level or higher
class _Fetcher(object):
	def __init__(self, split, pairs, meta_path_prefix, data_path_prefix, supervision):
		assert split in _LEGAL_SPLITS
		assert pairs in _LEGAL_PAIRS
		assert supervision in _LEGAL_SUPER
		# Meta data
		objects_file    = os.path.join(meta_path_prefix, split, pairs, 'objects.mat')
		self.objects    = scipy.io.loadmat(objects_file)['objects'][0][0] # RoI/proposal/candidate
		self.objs_imap  = {item.item():i for i,item in enumerate(self.objects['obj_id'])} # To find object in self.objects by id
		im_names_file   = os.path.join(meta_path_prefix, 'image_filenames_%s.mat' % split)
		self.fnames     = [x[0] for x in scipy.io.loadmat(im_names_file)['image_filenames'][0]] # Names for image files
		obj_names_file  = os.path.join(meta_path_prefix, 'vocab_objects.mat')
		self.obj_names  = [x[0][0] for x in scipy.io.loadmat(obj_names_file)['vocab_objects']]
		pred_names_file = os.path.join(meta_path_prefix, 'vocab_predicates.mat')
		self.pred_names = [x[0][0] for x in scipy.io.loadmat(pred_names_file)['vocab_predicates']]
		pairs_file      = os.path.join(meta_path_prefix, split, pairs, 'pairs.mat')
		self._pairs     = scipy.io.loadmat(pairs_file)['pairs'][0][0]
		# Pre-extracted features
		self.spatial_features_dir = os.path.join(meta_path_prefix, split, pairs, 'features', 'spatial-%s' % supervision)
		# Data (image files)
		self.im_dir = os.path.join(data_path_prefix, 'sg_%s_images' % split)

	# For unpaired RoIs, this should be your primary function. It returns a
	# list of objects that contain data which the other functions provide
	def entities(self, im_id):
		idxs,_ = np.where(self.objects['im_id'] == im_id)
		return [Entity(idx, self) for i,idx in enumerate(idxs)]

	# For paired RoIs (sub+obj), this should be your primary function.
	def pairs(self, im_id):
		idxs,_ = np.where(self._pairs['im_id'] == im_id)
		keys = self._pairs.dtype.names
		return {k:np.stack([squeeze(self._pairs[k][i]) for i in idxs]) for k in keys}
		
	def spatial(self, im_id, rel_ids=None):
		spatial_features_fpath = os.path.join(self.spatial_features_dir, '%d.mat' % im_id)
		spatial = scipy.io.loadmat(spatial_features_fpath)['spatial']
		if rel_ids:
			selected_pairs = [i for i in range(spatial.shape[0]) if spatial[i,0] in rel_ids]
			spatial = spatial[selected_pairs,:] # Select rows for given pair (aka rel)
			assert len(rel_ids) == spatial.shape[0]
			return torch.from_numpy(spatial[:,1:])
		else:
			return torch.from_numpy(spatial[:,1:])

	def bbs(self, im_id):
		bbs = [ent.bb() for ent in self.entities(im_id)]
		return torch.stack(bbs, 0).view(-1, 4)

	def fname(self, im_id):
		return self.fnames[im_id - 1]

	def image(self, im_id_or_fname):
		if type(im_id_or_fname) == str:
			fname = im_id_or_fname
		elif type(im_id_or_fname) == int:
			fname = self.fname(im_id_or_fname)
		else:
			raise Exception('Illegal argument type. Expected int or str. Got %s' % type(im_id_or_fname))
		exts = ['jpg', 'gif', 'png']
		trunk,_ = os.path.splitext(os.path.join(self.im_dir, fname))
		for ext in exts: # Try multiple file extensions b/c the mat files that come with unrel don't give the extension, and some are gifs
			fpath = '%s.%s' % (trunk, ext)
			if os.path.exists(fpath):
				image = PIL.Image.open(fpath)
				if not isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
					image = image.convert('RGB') # Convert to JPEG s.t. all images have 3 channels
				return image
		raise Exception('File not found %s' % fname)

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

# Struct to hold info for a single candidate/RoI/proposal: BB, im_id, obj_cat, etc
class Entity(object):
	def __init__(self, idx, fetcher):
		self.fetcher = fetcher
		meta = self.fetcher.objects
		for key in meta.dtype.names:
			val = meta[key][idx]
			self.__setattr__(key, squeeze(val))
	def name(self):
		return self.fetcher.obj_names[self.obj_cat - 1]
	def bb(self):
		bb = self.object_box
		return (bb[0], bb[1], bb[2], bb[3]) # x1, y1, x2, y2

	# Return a crop of the image, containing only this entity's bounding box.
	# Imdata should be a PIL Image
	# Or a tensor of C x H x W
	def crop(self, imdata):
		bb = self.bb()
		if isinstance(imdata, PIL.ImageFile.ImageFile):
			return imdata.crop(bb)
		elif isinstance(imdata, torch.Tensor):
			assert imdata.shape[0] == 3, imdata.shape
			return imdata[:, bb[1]:bb[3], bb[0]:bb[2]]

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
