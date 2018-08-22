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

from unrel_model import Model, TRANSFORM

import pdb

xform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((224,224)),
	torchvision.transforms.ToTensor()
])
toimg = torchvision.transforms.ToPILImage()

BGR = torch.LongTensor((2,1,0)).cuda()
MEAN = torch.FloatTensor(list(reversed([123.68, 116.779, 103.939]))).cuda().view(3,1,1) # Reversed b/c caffe does BGR

fetcher = unrel.Builder().split('train', 'annotated')
obj_names = ['<background>'] + fetcher.obj_names


modelpath = 'netmodel.pth'
if os.path.exists(modelpath):
	model = torch.load(modelpath)
else:
	model = Model()
	torch.save(model, modelpath)
model = model.cuda()

cts = torch.zeros(101).int()
for i in range(1,100):
	imdata = TRANSFORM(fetcher.image(i)).cuda()
	bbs = [list(ent.bb()) for ent in fetcher.entities(i)]
	print('IMID %4d' % i)
	if not fetcher.entities(i): continue
	scoresets = model(imdata, torch.Tensor(bbs).cuda())
	for j, ent in enumerate(fetcher.entities(i)):
		scores = scoresets[j,:]
		cat_id = torch.argmax(scores)
		_,ranks = scores.sort(0, True)
		rank = (ranks == int(ent.obj_cat)).nonzero()[0,0]
		cts[rank:] += 1
		print('RANK: %3d\tPREDICTION: (#%3d) %12s\t GT: (#%3d) %12s' % (rank, cat_id, obj_names[cat_id], ent.obj_cat, ent.name()))

print('cts', cts)
plt.bar( torch.arange(len(cts)), cts.float() / cts[-1].float(), width=1.0 )
plt.show()
