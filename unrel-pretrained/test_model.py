#!/usr/bin/env python
#
# Test that the model we saved from caffe is functional.

import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
import util.matlab
import util.unrel_data as unrel

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

import skimage.io
import numpy as np

######
# FC layers
######

fc_state_dict = torch.load('linear.prototxt.statedict.pth')
classifier_keys = ['fc6.1', 'fc7.1', 'cls_score.1']
map_keys = ['0', '3', '6']

def classifier_layers():
	for i in range(3):
		geom = fc_state_dict['%s.weight' % classifier_keys[i]].shape[::-1]
		yield nn.Linear(*geom)
		if i < 3-1:
			yield nn.ReLU()
			yield nn.Dropout()

classifier = nn.Sequential(*[x for x in classifier_layers()])
modified_dict = {}
for i,pfx in enumerate(classifier_keys):
	for kind in ('.weight', '.bias'):
		modified_dict[map_keys[i]+kind] = fc_state_dict[pfx+kind]
classifier.load_state_dict(modified_dict)
print(classifier)

######
# Conv layers
######

vgg = models.vgg16()
features = vgg.features
conv_state_dict = torch.load('conv.prototxt.statedict.pth')
keys = zip(features.state_dict().keys(), conv_state_dict.keys())
modified_dict = {vk: conv_state_dict[ck] for vk, ck in keys}
features.load_state_dict(modified_dict)
print(features)


######
# Test
######

features = features.cuda()
classifer = classifier.cuda()

features.eval()
classifier.eval()


xform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((224,224)),
	torchvision.transforms.ToTensor()
])
toimg = torchvision.transforms.ToPILImage()

BGR = torch.LongTensor((2,1,0)).cuda()
MEAN = torch.FloatTensor(list(reversed([123.68, 116.779, 103.939]))).cuda().view(3,1,1) # Reversed b/c caffe does BGR

fetcher = unrel.Builder().split('train', 'annotated')
obj_names = ['<background>'] + fetcher.obj_names


def run_bbs(im_id=890, show=False, verbosity=0):
	global cts
	if im_id % 20 == 0: print('running img', im_id)
	entities = fetcher.entities(im_id)
	im_data = fetcher.image(im_id)
	for ent in entities:
		t = xform(ent.crop(im_data)).cuda()
		t *= 255 # Accomodate for the fact that the model comes from caffe
		t = t.index_select(0, BGR) # Accomodate for the fact that the model comes from caffe
		t -= MEAN # Subtract mean
		t.unsqueeze_(0)
		x = features(t)
		y = classifier(x.view(-1))
		cat_id = torch.argmax(y)
		_,ranks = y.sort(0, True)
		rank = (ranks == int(ent.obj_cat)).nonzero()[0,0]
		cts[rank:] += 1 # Increment all counts >= rank, indicating that the corret result is in the top k predictions
		if verbosity >= 1:
			print('TOP5: %2d\tPREDICTION: (#%d) %12s\t GT: (#%d) %12s' % (in_topk, cat_id, obj_names[cat_id], ent.obj_cat, ent.name()))
		if show:
			t.squeeze_(0)
			toimg(t.cpu()).show()

cts = torch.zeros(101).int()
for i in range(1,4000): run_bbs(i)
print('cts', cts)
plt.bar( torch.arange(len(cts)), cts.float() / cts[-1].float(), width=1.0 )
plt.show()
