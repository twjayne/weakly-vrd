import torch
import torch.nn as nn


# Classifier 1 takes the base features into its first layer. Its second layer
# takes the output of the first layer, concatentated with the extra features.
class Classifier1(nn.Module):
	# @arg D_in1 is the input dimension of the first linear layer.
	# @arg D_out1 is the output dimension of the first linear layer.
	# @arg D_in2 is the dimension of the extra features. The input dimension of the 2nd linear layer is actually this number plus D_out1.
	# @arg geom is the remaining hidden and output dimensions.
	def __init__(self, D_in1, D_out1, D_in2, *geom):
		super(Classifier1, self).__init__()
		self.stage1 = nn.Linear(D_in1, D_out1)
		linear2 = nn.Linear(D_out1 + D_in2, geom[0])
		stage2_layers = [linear2] + [nn.Linear(geom[i], geom[i+1]) for i in range(len(geom)-1)]
		self.stage2 = nn.Sequential(*stage2_layers)

	def forward(self, x):
		raise Exception('This class is not implemented to be run this way. In order to perform a forward pass, please run h1 = <classifier>.stage1.forward(x1); x2[:range] = h1; out = <classifier>.stage2.forward(x2);')
