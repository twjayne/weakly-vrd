#!/usr/bin/env python

# This vgg16 model comes from the caffe files provided by the writers of
# "Weakly-supervised learning of visual relations." See
# https://www.di.ens.fr/willow/research/unrel/

# You must add faster-rcnn.pytorch-master/lib to the PYTHONPATH so that this module
# can make use of faster-rcnn.

# If you get error
# RuntimeError: cuda runtime error (77) : an illegal memory access was encountered at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCTensorCopy.c:70
# then run with CUDA_LAUNCH_BLOCKING=1

# If you get error
# RuntimeError: expected stride to be a single integer value or a list of 3 values to match the convolution dimensions, but got stride=[1, 1]
# then add img.unsqueeze_(0)

import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms

# Import local modules
import sys, os
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
import unrel.unrel_data as unrel

# Import faster-rcnn modules
# Requries PYTHONPATH to point to faster-rcnn.pytorch-master/lib
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.utils.config import cfg

import pdb

class Model(nn.Module):
	def __init__(self, features=None, classifier=None, **kwargs):
		super(Model, self).__init__()
		self.verbose  = kwargs.get('verbose', False)
		self.roi_mode = kwargs.get('mode', MODE_ALIGN)
		curdir = os.path.realpath(os.path.join(__file__, '..'))
		self._init_features(features or os.path.join(curdir, 'conv.prototxt.statedict.pth'))
		self._init_classifier(classifier or os.path.join(curdir, 'linear.prototxt.statedict.pth'))
		spatial_scale   = kwargs.get('scale', 1.0/32.0)
		self.RoIPooling = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, spatial_scale) # This needs to be the ratio of imdata.shape to the shape of the feature map at the end of the convolutional layers. This is architecture-dependent, not image-dependent (though a pixel here or there can cause some small shift in the true ratio).
		self.RoIAlign   = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, spatial_scale)

	# @arg batch should be a dict with keys 'image' and 'bbs'. The values for these keys should be lists of tensors. (One image's bbs should all be in the same tensor.) Using lists allows the images to be different sizes.
	def forward(self, batch):
		# Extract features
		images_features = [self.features(self._do_cuda(image)) for image in batch['image']]
		# for i, image in enumerate(batch['image']): assert abs(float(image.shape[2]) / float(images_features[i].shape[2]) - 32.0) < 1 or abs(float(image.shape[3]) / float(images_features[i].shape[3]) - 32.0) < 1, (batch['im_id'][i], image.shape, images_features[i].shape, image.shape[2] / images_features[i].shape[2], image.shape[3] / images_features[i].shape[3])
		# Get ROIs
		roisets = [self._rois(im_feat, batch['bbs'][i]) for i, im_feat in enumerate(images_features)]
		# return
		rois = torch.cat(roisets)
		# Get appearance features (Classify ROIs)
		n_rois = rois.shape[0]
		return self.classifier(rois.view(n_rois, -1)) # Rows alternate subj,obj
		
	def _rois(self, features, bbs):
		batch_index = torch.zeros(bbs.shape[0],1)
		tensors = ( self._do_cuda(batch_index), self._do_cuda(bbs).float() )
		boxes_plus = torch.cat(tensors, 1)
		if self.roi_mode == MODE_ALIGN:
			return self.RoIAlign(features, boxes_plus)
		elif self.roi_mode == MODE_POOL:
			return self.RoIPooling(features, boxes_plus)
		else:
			raise Exception('Illegal mode %s' % (self.roi_mode,))

	def _load_state_dict(self, path_or_dict):
		return path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict)

	# Copy tensor to GPU if this model has been copied to the GPU
	def _do_cuda(self, tensor):
		if next(self.features.parameters()).is_cuda and not tensor.is_cuda:
			return tensor.cuda()

	######
	# FC layers
	######
	def _init_classifier(self, path_or_dict=None):
		print('Loading FC layers...')
		state_dict = self._load_state_dict(path_or_dict)
		src_keys = ['fc6.1', 'fc7.1', 'cls_score.1']
		map_keys = ['0', '3', '6']
		def classifier_layers():
			for i in range(3):
				geom = state_dict['%s.weight' % src_keys[i]].shape[::-1]
				yield nn.Linear(*geom)
				if i < 3-1:
					yield nn.ReLU()
					yield nn.Dropout()
		classifier = nn.Sequential(*[x for x in classifier_layers()])
		mapped_dict = {}
		for i,pfx in enumerate(src_keys):
			for kind in ('.weight', '.bias'):
				mapped_dict[map_keys[i]+kind] = state_dict[pfx+kind]
		classifier.load_state_dict(mapped_dict)
		if self.verbose: print(classifier)
		self.classifier = classifier

	######
	# Conv layers
	######
	def _init_features(self, path_or_dict=None):
		print('Loading Conv layers...')
		vgg = torchvision.models.vgg16()
		features = vgg.features
		state_dict = self._load_state_dict(path_or_dict)
		keys = zip(features.state_dict().keys(), state_dict.keys())
		mapped_dict = {vk: state_dict[ck] for vk, ck in keys}
		features.load_state_dict(mapped_dict)
		if self.verbose: print(features)
		self.features = features

MODE_ALIGN = 'align'
MODE_POOL  = 'pool'
MODE_CROP  = 'crop'

MEAN = torch.FloatTensor(list(reversed([123.68, 116.779, 103.939]))).view(1,3,1,1) # Reversed b/c caffe does BGR
BGR_indices = torch.LongTensor((2,1,0))
_transform_caffe_scale = torchvision.transforms.Lambda(lambda ten: ten * 255)
_transform_caffe_BGR = torchvision.transforms.Lambda(lambda ten: ten.index_select(0, BGR_indices))
_transform_subtract_mean = torchvision.transforms.Lambda(lambda ten: ten - MEAN)

TRANSFORM = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	_transform_caffe_scale,
	_transform_caffe_BGR,
	_transform_subtract_mean,
])

if __name__ == '__main__':
	modelpath = 'netmodel.pth'
	if os.path.exists(modelpath):
		model = torch.load(modelpath)
	else:
		model = Model()
		torch.save(model, modelpath)
	model = model.cuda()

	fetcher = unrel.Builder().split('train', 'annotated')
	obj_names = ['<background>'] + fetcher.obj_names

	imid = 3
	imdata = TRANSFORM(fetcher.image(imid)).cuda()
	bbs = [list(ent.bb()) for ent in fetcher.entities(imid)]
	ss = model(imdata, torch.Tensor(bbs).cuda())

	print(ss.argmax(1).cpu().numpy().tolist())
	print([ent.obj_cat for ent in fetcher.entities(imid)])
	print([ent.name() for ent in fetcher.entities(imid)])
