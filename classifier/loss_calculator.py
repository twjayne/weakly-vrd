import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math
import sys
import dataset
import util.gpu
from RecallEvaluator import RecallEvaluator

import pdb

DTYPE = torch.float32
DO_CUDA = True

# Accumulates loss across an epoch
class LossCalculator(object):
	# @arg input_key can be a lambda or a dictionary key
	def __init__(self, model, **opts):
		self.model      = model
		self.input_key  = opts.get('input_key', 'X')
		self.target_key = opts.get('target_key', 'y')
		self.loss_fn    = opts.get('loss', nn.CrossEntropyLoss())
		self.recall_ks  = opts.get('recall', [50])
		self.n_klasses  = opts.get('n', len(list(model.parameters())[-1]))
		self.data_name  = opts.get('name', 'TRAIN')
		if DO_CUDA: self.loss_fn.cuda()
		self.init_epoch()

	def _new_stats(self):
		return Stats(len(self.recall_ks), self.n_klasses)

	# Initialize accumulator for computing epoch peformance
	def init_epoch(self):
		self.epoch_stats = self._new_stats()
		self.init_batch()

	# Computes final stats
	def end_epoch(self, recall=None):
		stats = self.epoch_stats.compute(recall)
		self.init_epoch()
		return stats

	# Initialize accumulator for computing batch peformance
	def init_batch(self):
		self.batch_stats = self._new_stats()

	# Computes final stats
	def end_batch(self):
		stats = self.batch_stats.compute()
		self.init_batch()
		return stats

	# Perform forward pass on batch or dataset and compute loss
	def __call__(self, data, verbose=True):
		if isinstance(data, dict): # Handle batch (useful for 'train')
			loss = self.calc_on_image(*self.predict(data))
			self.epoch_stats.accumulate(self.batch_stats)
			return loss
		elif isinstance(data, DataLoader): # Handle entire dataset (useful for 'test')
			return self.calc_on_dataloader(data, verbose)
		else:
			raise Exception('invalid input type %s' % type(data))

	# Calculate loss, accuracy, recall on an entire dataset (not just a batch)
	# @return loss, accuracy, recall, recall2
	def calc_on_dataloader(self, dataloader, verbose=True):
		self.init_epoch()
		if verbose: sys.stdout.write('test batch')
		rev = RecallEvaulatorOverride()
		for i, batch in enumerate(dataloader):
			predictions, targets = self.predict(batch)
			rev.accumulate(predictions)
			loss = self.calc_on_image(predictions, targets)
			self.epoch_stats.accumulate(self.end_batch())
			assert isinstance(self.epoch_stats.sum_loss, float)
			if verbose and i % 50 == 0:
				mm = util.gpu.get_memory_map()
				sys.stdout.write(' %d' % i)
				sys.stdout.flush()
		if verbose: print()
		rev_recall = rev.compute()
		return self.end_epoch(rev_recall)

	# For the sake of recall, a batch is believed to be one image
	# Set self.target, self.prediction, self.acc
	# @return loss
	def calc_on_image(self, predictions, targets):
		N = targets.shape[0]
		self.batch_stats.n_example += N
		# Compute loss
		loss = self.loss_fn( predictions, targets.reshape(-1) )
		self.batch_stats.sum_loss += loss.item()
		# Compute/accumulate Recall@X
		confidences, pred_klasses = predictions.max(1) # get confidences and predicted classes
		_,reversed_order = confidences.sort()
		for recall_row, k in enumerate(self.recall_ks):
			if k >= len(confidences): # Use all
				top_k_targets = targets
				top_k_preds   = pred_klasses
			else: # Sort and use top k
				top_k_targets = targets[reversed_order][-k:]
				top_k_preds   = pred_klasses[reversed_order][-k:]
			# Update counts on self
			is_correct = top_k_preds == top_k_targets
			for i, yes in enumerate(is_correct):
				klass = top_k_targets[i]
				self.batch_stats.n_example_by_class[recall_row, klass] += 1
				if yes:
					self.batch_stats.tp[recall_row, klass] += 1
				else:
					self.batch_stats.fp[recall_row, klass] += 1
		# Return
		return loss

	# Extracts X and Y and makes a forward pass on self.model
	# This is abstracted to its own method in case you want to divide up loss
	# calculation by image (for purposes of Recall@X) but do forward passes
	# with more than one image.
	# Sets self.prediction, self.target
	# Returns (prediction, target)
	def predict(self, batch):
		# Get Y
		if callable(self.target_key):
			self.target = self.target_key(batch)
		elif isinstance(self.target_key, str):
			self.target = self._do_cuda(batch[self.target_key])
		else:
			raise Exception('illegal target type')
		# Pass X through model
		if callable(self.input_key): # lambda or function
			self.prediction = self.input_key(self.model, batch)
		elif isinstance(self.input_key, str):
			self.prediction = self.model(self._do_cuda(batch[self.input_key]))
		else:
			raise Exception('illegal input type')
		# Return
		return self.prediction, self.target

	def _do_cuda(self, tensor):
		if not tensor.is_cuda and next(self.model.parameters()).is_cuda:
			return tensor.cuda()
		else:
			return tensor

class Stats(object):
	def __init__(self, recall_rows, n_klasses):
		self.sum_loss  = 0
		self.n_example = 0
		self.tp = torch.zeros(recall_rows, n_klasses, dtype=torch.int32)
		self.fp = torch.zeros_like(self.tp)
		self.n_example_by_class = torch.zeros_like(self.tp)

	def accumulate(self, other):
		self.sum_loss += other.sum_loss
		self.n_example += other.n_example
		self.tp += other.tp
		self.fp += other.fp
		self.n_example_by_class += other.n_example_by_class

	def compute(self, recall=None):
		if self.n_example == 0:
			import traceback
			for line in traceback.format_stack(): print(line.strip())
		self.loss = self.sum_loss / self.n_example
		self.acc = self.tp.sum().item() / self.n_example
		self._compute_recall()
		if not isinstance(recall, type(None)): self.rec = torch.Tensor([recall])
		return self

	# unrel_recall : Number of correct predictions, divided by number of predictions (up to K per image). This biases frequently-seen classes.
	# rec          : Recall is computed class-by-class, then averaged. (It will be nan if not all classes have been seen.)
	# rec2         : Same as `rec`, but only classes have have been seen are used to compute the average.
	def _compute_recall(self):
		self.unrel_recall = self.tp.sum(1).float() / self.n_example_by_class.sum(1).float()
		recall_by_class_f = self.tp.float() / self.n_example_by_class.float()
		self.rec = recall_by_class_f.mean(1)
		N = self.n_example_by_class.shape[0]
		self.rec2 = torch.zeros_like(self.rec)
		for row in range(N):
			nonzero_classes = self.n_example_by_class[row,:] != 0 # classes for which we have actually seen examples. ideally this is all classes, but maybe we're working with a subset of the dataset
			if nonzero_classes.sum().item() == 0: # because we might get an empty tensor for recall_by_class_f[row,nonzero_classes]
				self.rec2[row] = math.nan
			else:
				self.rec2[row] = recall_by_class_f[row,nonzero_classes].mean().item() # In case there are classes unrepresented in the dataset, this will give a more useful recall

# Most of the code in this class is copied from Tyler's RecallEvaluator. See him for doc.
# Tyler's RecallEvaluator is a translation of the MATLAB project for the paper "Weakly-supervised learning of visual relations"
class RecallEvaulatorOverride(RecallEvaluator):
	def __init__(self):
		super(RecallEvaulatorOverride, self).__init__()
		self.predictions = None

	def accumulate(self, predictions):
		o = predictions.cpu().data
		if isinstance(self.predictions, type(None)):
			self.predictions = o
		else:
			self.predictions = torch.cat((self.predictions, o))

	def compute(self):
		self.zeroshot = False
		self.candidatespairs = 'annotated'
		self.use_objectscores = False
		pairs, scores, annotations = self.predict()
		candidates, groundtruth = self.format_testdata_recall(pairs, scores, annotations)
		recall_predicate, _ = self.top_recall_Relationship(self.Nre, candidates, groundtruth)
		return recall_predicate

	def predict(self):
		annotations = self.get_full_annotations()
		pairs = self.load_candidates(self.candidatespairs)
		prediction = self.predictions.numpy()
		return (pairs, prediction, annotations)

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
		top_recall = top_recall
		ap = self.VOCap(recall, precision)
		ap = ap*100

		return top_recall, ap
