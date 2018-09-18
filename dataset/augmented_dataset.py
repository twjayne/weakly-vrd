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

class Dataset(dataset.faster_rcnn.Dataset):
	def __init__(self, **kwargs):
		assert kwargs.get('pairs', 'candidates') == 'candidates'
		kwargs['pairs'] = 'candidates'
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
	ds = Dataset()
