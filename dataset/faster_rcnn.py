import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.utils.data
import torch.utils.data.sampler as torchsampler
import numpy as np
import scipy.io
import os
import unrel.unrel_model as unrel
import unrel.unrel_data as unrel_data

import pdb

def relpath(*args):
	return os.path.realpath(os.path.join(__file__, '..', *args))


class Dataset(torch.utils.data.Dataset):
	def __init__(self, **kwargs):
		self.pairs    = kwargs.get('pairs', 'annotated') # annotated|candidates
		self.split    = kwargs.get('split', 'test') # test|train
		self.imagedir = kwargs.get('image', relpath('images')) # Path to images dir (jpg)
		self.metadir  = kwargs.get('info', relpath('data/vrd-dataset')) # Path to meta info dir (matlab)
		self.name     = kwargs.get('name', None) # For logging
		self.xform    = kwargs.get('transform', kwargs.get('xform', unrel.TRANSFORM))
		self.fetcher  = unrel_data.Builder().split(self.split, self.pairs)
		self._init()

	def __len__(self):
		return len( self._data )

	def __getitem__(self, im_idx):
		pairs = self._data[im_idx]
		im_id = pairs[0]['im_id']
		cats  = [pair[key] for pair in pairs for key in ('sub_cat', 'obj_cat')] # these alternate subject, object, ...
		bbs   = [pair[key] for pair in pairs for key in ('subject_box', 'object_box')] # these alternate subject, object, ...
		preds = [pair['rel_cat']-1 for pair in pairs]
		image = self.fetcher.image( im_id ) # PIL image
		entities = self.fetcher.entities( im_id )
		spatial  = self.fetcher.spatial( im_id, [p['rel_id'] for p in pairs] )
		if self.xform: image = self.xform(image)
		if isinstance(image, torch.Tensor): assert len(image.shape) == 4, image.shape
		return {
			'preds':   torch.tensor(preds),  # the 1st one relates to the 1st TWO cats/bbs; the 2nd one relates to the 2nd TWO cats/bbs, etc. 
			'cats':    torch.tensor(cats),   # category ids for subj,obj (i.e. class labes)
			'bbs':     torch.stack(bbs, 0),  # bounding boxes (x1, y1, x2, y2)
			'spatial': spatial,              # a dict of (sub_id,obj_id) => 400-d features for all candidate pairs in this image
			'image':   image,
			'imid':    im_id,
		}

	def _init(self):
		matlab = scipy.io.loadmat(os.path.join(self.metadir, self.split, self.pairs, 'pairs.mat'))['pairs'][0,0]
		self._keys = matlab.dtype.names
		map_from_matlab = { im_id.item() : [] for im_id in matlab['im_id'] }
		get = lambda key, i: torch.from_numpy(matlab[key][i].astype(np.int32)) if len(matlab[key][i]) > 1 else matlab[key][i].item()
		for i in range(len(matlab['im_id'])):
			im_id = matlab['im_id'][i].item()
			map_from_matlab[im_id].append( { key: get(key,i) for key in self._keys } )
		self._data = list(map_from_matlab.values())

	def __iter__(self):
		return _DataLoaderIter(self)

# Behaves like a dataloader but doesn't stack all fields (because some of them, such as images, may be of differing dimensions)
class FauxDataLoader(torch.utils.data.DataLoader):
	def __init__(self, dataset, **kwargs):
		self.dataset    = dataset
		self.sampler    = self._init_sampler(**kwargs)
		self.batch_size = self.sampler.batch_size
		self.iterator   = iter(self.sampler)
	def __iter__(self):
		return self
	def __len__(self):
		return len([item for batch in self.sampler for item in batch])
	def __next__(self):
		# Get batch
		try:
			batch = next(self.iterator)
		except StopIteration:
			self.iterator = iter(self.sampler)
			raise StopIteration
		if isinstance(batch[0], int):
			batch = [self.dataset[i] for i in batch]
		# Return dict
		return {
			'im_id':    [d['imid'] for d in batch],
			'image':   [d['image'] for d in batch],
			'bbs':     [d['bbs'] for d in batch],
			'preds':   torch.cat([d['preds'] for d in batch]),
			'spatial': torch.cat([d['spatial'] for d in batch]),
			'N':       len(batch),
		}
	def _init_sampler(self, **kwargs):
		sampler = kwargs.get('sampler')
		batch_size  = kwargs.get('batch_size', 1)
		drop_last   = kwargs.get('drop_last', False)
		if isinstance(sampler, torchsampler.BatchSampler):
			return sampler
		if sampler == None:
			sampler = torchsampler.RandomSampler(self.dataset)
		elif not isinstance(sampler, torchsampler.Sampler):
			sampler = torchsampler.RandomSampler(sampler)
		return torchsampler.BatchSampler(sampler, batch_size, drop_last)

# Test this module
if __name__ == '__main__':
	N_TESTS = 4
	passed = 0
	dataset = Dataset(transform=unrel.TRANSFORM)
	# Test on a subset sampler
	batch_sampler = torchsampler.SequentialSampler(range(14))
	dataloader = FauxDataLoader(dataset, sampler=batch_sampler)
	for batch_i, batch in enumerate(dataloader):
		assert isinstance(batch['image'], list)
		for image in batch['image']:
			assert isinstance(image, torch.Tensor)
		print('dataset count %3d / %3d' % ((1+batch_i) * dataloader.sampler.batch_size, len(dataloader)))
	passed += 1; print('OK %d/%d' % (passed, N_TESTS))
	# Test on a batched subset sampler
	batch_sampler = torchsampler.BatchSampler(torchsampler.SequentialSampler(range(14)), 3, False)
	dataloader = FauxDataLoader(dataset, sampler=batch_sampler)
	for batch_i, batch in enumerate(dataloader):
		assert isinstance(batch['image'], list)
		for image in batch['image']:
			assert isinstance(image, torch.Tensor)
		print('dataset count %3d / %3d' % ((1+batch_i) * dataloader.sampler.batch_size, len(dataloader)))
	passed += 1; print('OK %d/%d' % (passed, N_TESTS))
	# Test second time on same dataloader to ensure that reset works
	for batch_i, batch in enumerate(dataloader):
		assert isinstance(batch['image'], list)
		for image in batch['image']:
			assert isinstance(image, torch.Tensor)
		print('dataset count %3d / %3d' % ((1+batch_i) * dataloader.sampler.batch_size, len(dataloader)))
	passed += 1; print('OK %d/%d' % (passed, N_TESTS))
	# Test without supplying sampler
	dataloader = FauxDataLoader(dataset, batch_size=2)
	for batch_i, batch in enumerate(dataloader):
		assert isinstance(batch['image'], list)
		for image in batch['image']:
			assert isinstance(image, torch.Tensor)
		if batch_i % 50 == 0:
			print('dataset count %3d / %3d' % ((1+batch_i) * dataloader.sampler.batch_size, len(dataloader)))
	passed += 1; print('OK %d/%d' % (passed, N_TESTS))
