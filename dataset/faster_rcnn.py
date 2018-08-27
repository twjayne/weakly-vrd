import torch
assert torch.__version__.startswith('0.4'), 'wanted version 0.4, got %s' % torch.__version__
import torch.utils.data
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
		preds = [pair['rel_cat'] for pair in pairs]
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

class FauxDataLoader(object):
	def __init__(self, dataset, batch_size=1):
		self.dataset = dataset
		self.batch_size = batch_size
		self.cur = 0
	def __iter__(self):
		return self
	def __len__(self):
		return int(np.ceil(len(self.dataset) / float(self.batch_size)))
	def __next__(self):
		remaining = len(self.dataset) - self.cur - 1
		interval  = min(remaining, self.batch_size)
		if interval == 0:
			self.cur = 0
			raise StopIteration
		else:
			selection = [self.dataset[i] for i in range(self.cur, self.cur+interval)]
			self.cur += interval
			return {
				'im_id':    [d['imid'] for d in selection],
				'image':   [d['image'] for d in selection],
				'bbs':     [d['bbs'] for d in selection],
				'preds':   torch.cat([d['preds'] for d in selection]),
				'spatial': torch.cat([d['spatial'] for d in selection]),
				'N':       len(selection),
			}

if __name__ == '__main__':
	dataset = Dataset(transform=unrel.TRANSFORM)
	dataloader = FauxDataLoader(dataset, 2)
	count = 0
	for batch in dataloader:
		assert isinstance(batch['image'], list)
		for image in batch['image']:
			assert isinstance(image, torch.Tensor)
			count += 1
		print('dataset count %3d / %3d' % (dataloader.cur, len(dataset)))
	print('OK')
