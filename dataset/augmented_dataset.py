# Datasets for full-supervised setting. The trainset provided is small, so we
# augment it with data from 'candidates' set.

__import__(__package__ or '__init__')
import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.utils.data
import torch.utils.data.sampler as torchsampler
import numpy as np
import scipy.io
import os
import unrel.unrel_model as unrel
import unrel.unrel_data as unrel_data
import dataset.faster_rcnn

import pdb

ANNOTATED  = unrel_data._LEGAL_PAIRS[0]
CANDIDATES = unrel_data._LEGAL_PAIRS[1]
assert ANNOTATED  == 'annotated'
assert CANDIDATES == 'candidates'

def thresholded_candidates(ca_list, gt_list):
	good_candidates = []
	is_gt_exact_matched = np.zeros(len(gt_list), np.int8)
	for ca in ca_list:
		for gt_i, gt in enumerate(gt_list):
			# Check for sub+obj category match
			if np.all([gt[key] == ca[key] for key in ('obj_cat', 'sub_cat')]):
				# Check for bb exact match
				ious = [unrel_data.iou(ca[key], gt[key]) for key in ('subject_box', 'object_box')]
				if ious[0] > 0.7 and ious[1] > 0.7:
					if 0 == is_gt_exact_matched[gt_i] and np.array_equal(ca['subject_box'], gt['subject_box']) and np.array_equal(ca['object_box'], gt['object_box']):
						is_gt_exact_matched[gt_i] = 1
						if ious[0] < 0.99 or ious[1] < 0.99: raise Exception('bad iou: %f' % (iou))
					ca['rel_cat'] = gt['rel_cat']
					good_candidates.append(ca)
	for i, is_exact_matched in enumerate(is_gt_exact_matched):
		if 0 == is_exact_matched:
			good_candidates.append(gt_list[i])
	return good_candidates

class Base(dataset.faster_rcnn.Dataset):
	def __init__(self, **kwargs):
		assert kwargs.get('pairs', None) != 'annotated'
		kwargs['pairs'] = 'candidates'
		super(Base, self).__init__(**kwargs)

# Second try: work on an object-by-object basis, and just swap out objects in
# existing triplets. (This attempt was abandoned at an incomplete state
# because the number of object candidates with IoU > 0.7 for some object in
# the annotated trainset was low.)
class Dataset2(Base):
	# Get a dict of (image,subj_id,obj_id) => triplet_data
	def _init(self):
		# Get 'objects' from objects.mat
		aobjs = self.aobjs  = self._get_meta('objects', pairs='annotated')
		cobjs = self.cobjs  = self._get_meta('objects', pairs='candidates')
		def obj_to_dict(ml_objs, i):
			obj = {k:ml_objs[k][i].item() for k in keys[:-1]}
			obj['bb'] = ml_objs['object_box'][i].astype(np.int) # change type because we were hitting overflow using ushort_scalars
			return obj
		# Make dict of (im_id,obj_cat) => [obj,...]
		aobj_dic = {}
		keys     = aobjs.dtype.names
		for i in range(len(aobjs['obj_id'])):
			obj = obj_to_dict(aobjs, i)
			aobj_dic.setdefault( (obj['im_id'],obj['obj_cat']), [] ).append(obj)
		# Collect matches for cobjs into make dict of (im_id,obj_cat) => [obj,...]
		cobj_matches = {}
		for i in range(len(cobjs['obj_id'])):
			obj = obj_to_dict(cobjs, i)
			gt_key = (obj['im_id'],obj['obj_cat'])
			if gt_key not in aobj_dic: continue
			for gt in aobj_dic[gt_key]:
				iou = unrel_data.iou(obj['bb'], gt['bb'])
				if iou > 0.7:
					cobj_matches.setdefault( (gt['im_id'],gt['obj_id']), [] ).append( (iou, obj) )

		print('%7d  %s' % (len(aobjs['obj_id']), 'Number of objects in gt (annotated) set'))
		print('%7d  %s' % (len(cobjs['obj_id']), 'Number of objects in candidates set'))
		n_augmenting_candidates = np.sum([len(x) for k,x in cobj_matches.items()])
		print('%7d  %s' % (n_augmenting_candidates, 'Number of objects with IoU > 0.7 from candidates'))

		# self.apairs = self._get_pairs(pairs='annotated')
		# self.cpairs = self._get_pairs(pairs='candidates')


# First try. Seek triplets from candidate set whose IoU > 0.7 with some
# triplet from the annotated training set for both subject and object.
class Dataset1(Base):
	def __init__(self, **kwargs):
		super(Dataset, self).__init__(**kwargs)
		if not '_data' in kwargs:
			# Now self._data is a list of lists of candidate pairs. Let's go
			# through them and compare them to pairs in a groundtruth dataset.
			gtopts = kwargs.copy()
			gtopts['pairs'] = 'annotated'
			gt_data = dataset.faster_rcnn.Dataset(**gtopts)._data
			gt_dict = {x[0]['im_id']:x for x in gt_data}
			ca_dict = {x[0]['im_id']:x for x in self._data if x[0]['im_id'] in gt_dict}
			# Now ca_dict has data for only the images that show up in GT. Let's
			# iterate over images, excluding pairs in CA that are poor matches for
			# pairs in GT.
			for im_id in gt_dict.keys():
				assert ca_dict[im_id]
				ca_dict[im_id] = thresholded_candidates(ca_dict[im_id], gt_dict[im_id])
			# Now let's overwrite the self._data list.
			self._data = [x for x in ca_dict.values() if x] # list of lists, where each sublist holds all pairs for a single image. The indices in the superlist do *not* correspond to im_id
			print('%d images in dataset' % len(self._data))
			print('%d pairs in dataset' % np.sum([len(x) for x in self._data]))

def load(fpath):
	opts = torch.load(fpath)
	return Dataset(**opts)

# Test this module
if __name__ == '__main__':
	ds = Dataset2(split='train')
