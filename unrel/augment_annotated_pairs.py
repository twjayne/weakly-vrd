# Build a new pairs.mat file in an 'augmented' directory. This uses the
# 'annotated' dataset as a base and proposals taken from edgeboxes.

__import__(__package__ or '__init__')
import unrel.unrel_data as unrel_data
import numpy as np
import scipy.io
import os
import pickle, logging, glob
import pdb

# An abstract base class which is sub-classed by other classes in this module.
class Base(object):
	def __init__(self, split='train'):
		self.split = split
	def _init_fetcher(self):
		self.fetcher = unrel_data.Builder().split(self.split, 'annotated')
		self.im_ids  = set(self.fetcher._pairs['im_id'].flatten())
		self.unlabelled_boxes_path = os.path.join(self.fetcher.metadir, self.split, 'unlabelled_boxes')
		self.roimaps_dir           = os.path.join(self.fetcher.metadir, self.split, 'roimaps_gt_0.7')
		return self.fetcher, self.im_ids

# Read each file in the 'unlabelled_boxes' directory (which should have been
# populated by the MATLAB script compute_additional_candidates.m). For each
# file, create a dictionary which maps obj_id to a list of all region
# proposals from 'unlabelled_boxes' which have an IoU > 0.7 with any ground-
# truth BB; write that dictionary to a pickle file.
class RoiMapper(Base):
	def run(self):
		# Load 'annotated' data
		fetcher, im_ids = self._init_fetcher()
		# Set input/output dirs
		os.makedirs(self.roimaps_dir, exist_ok=True)
		# Iterate through images, looking for matches in the results from edgeboxes
		n_workers = int(os.environ.get('N_ROI_WORKERS', 1))
		cur_worker = int(os.environ.get('ROI_WORKER_I', 0))
		print('%d workers\t%d' % (n_workers, cur_worker))
		if n_workers == 1:
			self.run_subset(im_ids)
		else:
			selected_ids = [i for i in im_ids if i % n_workers == cur_worker]
			self.run_subset(selected_ids)
			print('started subset %2d on %d' % (cur_worker, len(selected_ids)))
	# Create a map of (gt_obj_id) => [(),(),...] Where the items in the value
	# list are candidates from edgeboxes with IoU > 0.7 for the given gt
	# object. The format of these value items are a tuple containing a numpy
	# array and a float. The numpy array contains the bounding box and the
	# confidence which edgeboxes assigned to the bounding box (how likely it
	# is to be an object, not how likely it is to overlap a gt roi).
	# ([x1, y1, x2, y2, confidence], IoU)
	def run_subset(self, im_ids):
		for im_id in im_ids:
			im_roimap = {} # Maps gt obj_id => list of high-IoU candidates from edgeboxes
			print('im_id %7d' % im_id)
			fname = '%d.mat' % im_id
			# Load proposals from edgeboxes
			proposals = scipy.io.loadmat(os.path.join(self.unlabelled_boxes_path, fname))['unlabelled_boxes']
			proposals[:,2:4] += proposals[:,0:2] # Convert (x y w h) => (x1 y1 x2 y2)
			for row in range(proposals.shape[0]):
				candidate_bb = proposals[row,:-1].astype(np.int32)
				for entity in self.fetcher.entities(im_id):
					iou = unrel_data.iou(proposals[row,:-1], entity.bb(False).astype(np.int32))
					if iou > 0.7:
						im_roimap.setdefault(entity.obj_id, []).append((proposals[row,:], iou))
			# Sort each list of candidates s.t. highest confidence appears earliest
			for key in im_roimap: im_roimap[key].sort(reverse=True, key=lambda l: l[0][4])
			with open(os.path.join(self.roimaps_dir, '%d.pickle' % im_id), 'wb') as f:
				pickle.dump(im_roimap, f)
		return im_roimap

# An abstract base class which makes use of an roimap
class RoiMapUser(Base):
	def __init__(self):
		super(RoiMapUser, self).__init__()
		self._init_fetcher()
		self.roimap = self.aggregate_roimap()
	def aggregate_roimap(self):
		dikts = [load_pickle(fpath) for fpath in glob.glob(os.path.join(self.roimaps_dir, '*.pickle'))]
		roimap = {k:dikt[k] for dikt in dikts for k in dikt}
		logging.info('roimap aggregated. %d obj_ids' % len(roimap))
		return roimap
	def save_to_file(self, save, default_name, save_obj):
		if save:
			savedir   = os.path.join(self.fetcher.metadir, self.split, 'augmented')
			save_path = os.path.join(savedir, default_name) if isinstance(save,bool) else save
			logging.info('saving (%s) ...' % save_path)
			os.makedirs(savedir, exist_ok=True)
			scipy.io.savemat(save_path, save_obj)
		
# Given an roimap from RoiMapper, build a new pairs array (i.e. dataset).
# Proposals are taken from roimap. We treat the top 10 proposals for each gt
# object as valid BBs for that object, so for each gt pair, we make new
# 'augmented' pairs where the gt subject BB and object BB may be swapped out
# for one of the proposal BBs. Thus, the 'augmented' dataset may have many
# examples which fall under the same rel_id, sub_id, and obj_id; such examples
# differ *only* in their 'subject_box' or 'object_box'.
class PairsBuilder(RoiMapUser):
	def run(self, save=True):
		logging.info('starting pairs...')
		self.pairs  = self.fetcher._pairs
		self.keys   = self.pairs.dtype.names # ('im_id', 'rel_id', 'sub_id', 'obj_id', 'sub_cat', 'rel_cat', 'obj_cat', 'subject_box', 'object_box')
		n_gt        = len(self.pairs['rel_id'])
		# Build additional pairs from roimap, yielding a list of tuples which hold pair data s.t. they can be passed to a constructor for a numpy structured array
		augmented_pairs_tups = [pair_tup for gt_idx in range(n_gt) for pair_tup in self.augmented_pairs_for_gt(gt_idx)]
		# Build numpy structured array
		augmented_pairs = tuples2dict(augmented_pairs_tups, self.keys)
		logging.info(' '.join([str(d) for d in augmented_pairs['rel_id'].shape]))
		ag = augmented_pairs
		# pdb.set_trace()
		# Save to file
		if save: self.save_to_file(save, 'pairs.mat', {'pairs':augmented_pairs})
		return augmented_pairs
	# Given a gt pair, return a list which includes the gt pair and all augmented pairs (where the subj bb and obj bb have been swapped)
	def augmented_pairs_for_gt(self, gt_idx, K=10):
		# Get BBs from both gt and proposals
		def select_as_list(key):
			obj_id = self.pairs[key][gt_idx].item()
			return [v[0] for v in self.roimap[obj_id]] if obj_id in self.roimap else []
		# selectasarray = lambda key: 
		sub_selection = select_as_list('sub_id')[:K] # limit to top K
		obj_selection = select_as_list('obj_id')[:K] # limit to top K
		# First 4 elements are the BB
		sub_proposals = [arr[:-1] for arr in sub_selection]
		obj_proposals = [arr[:-1] for arr in obj_selection]
		# Last element is the confidence which edgeboxes computed
		sub_ranks     = [1.0] + [arr[-1] for arr in sub_selection]
		obj_ranks     = [1.0] + [arr[-1] for arr in obj_selection]
		# Get gt
		sub_gt_bb     = self.pairs['subject_box'][gt_idx]
		obj_gt_bb     = self.pairs['object_box'][gt_idx]
		# Concatenate gt with candidates
		sub_bbs       = [sub_gt_bb] + sub_proposals
		obj_bbs       = [obj_gt_bb] + obj_proposals
		# Collect pair tuples for all combinations of subject BBs and object BBs
		attr      = lambda key, sbb, obb: sbb.astype(np.uint16) if key == 'subject_box' else obb.astype(np.uint16) if key == 'object_box' else self.pairs[key][gt_idx].item()
		rank      = np.array([srank + orank  for srank in sub_ranks for orank in obj_ranks])
		rank_idxs = np.argsort(-rank)[:K]
		pair_tups = [tuple([attr(key, sbb, obb) for key in self.keys]) for sbb in sub_bbs for obb in obj_bbs]
		return [pair_tups[idx] for idx in rank_idxs] # Just return the top K

# Given an roimap from RoiMapper, build a new objects array, suitable for
# storing in an objects.mat file to mimic the ones provided in the UnRel
# dataset. This class follows the pattern of PairsBuilder. The resulting set
# of objects may have many that have the same obj_id. They will only differ in
# 'object_box'.
class ObjectsBuilder(RoiMapUser):
	def run(self, save=True):
		logging.info('starting objects...')
		self.objects = self.fetcher.objects
		self.keys    = self.objects.dtype.names
		n_gt         = len(self.objects['obj_id'])
		print(self.objects.dtype)
		# Build additional objects from roimap, yielding a list of tuples which hold object data s.t. they can be passed to a constructor for a numpy structured array
		augmented_objects_tups = [object_tup for gt_idx in range(n_gt) for object_tup in self.augmented_objects_for_gt(gt_idx)]
		# Build numpy structured array
		augmented_objects = tuples2dict(augmented_objects_tups, self.keys)
		logging.info(' '.join([str(d) for d in augmented_objects['obj_id'].shape]))
		# Save to file
		if save: self.save_to_file(save, 'objects.mat', {'objects':augmented_objects})
		return augmented_objects
	def augmented_objects_for_gt(self, gt_idx, K=10):
		# Get BBs from proposals
		obj_id      = self.objects['obj_id'][gt_idx].item()
		proposals   = [tup[0][:-1].astype(np.uint16) for tup in self.roimap[obj_id]] if obj_id in self.roimap else []
		# List gt and propsal BBs
		bbs         = [self.objects['object_box'][gt_idx]] + proposals[:K] # limit to top 10
		# Build object tuples for all bbs
		attr        = lambda key, bb: bb if key == 'object_box' else self.objects[key][gt_idx]
		object_tups = [tuple([attr(key, bb) for key in self.keys]) for bb in bbs]
		return object_tups


# Make a dict of numpy arrays in preparation for saving as a matlab datafile.
# Add extra dimensions because the UnRel pairs.mat files have it.
def tuples2dict(tuples, keys):
	getarray = lambda field_i : np.expand_dims(np.array( [tup[field_i] for tup in tuples] ).astype(np.uint16), -1)
	return { keys[field_i]: getarray(field_i) for field_i in range(len(keys)) }

def load_pickle(fpath):
	with open(fpath, 'rb') as f:
		data = pickle.load(f)
	return data

# Demo
if __name__ == '__main__':
	logging.getLogger().setLevel(logging.INFO)
	roimap = RoiMapper().run()
	augmented_pairs = PairsBuilder().run(True)
	objects = ObjectsBuilder().run(True)
