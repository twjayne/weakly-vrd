import torch
import torch.nn as nn
import numpy as np
import pdb

BATCHNORM_LAYER = (nn.BatchNorm1d, lambda k, i, o: k(o))

def model_generator(layer_widths, is_batch_gt_1):
    for i in range(1, len(layer_widths)):
        yield nn.Linear(layer_widths[i-1], layer_widths[i])
        if i < len(layer_widths) - 1: # All except the last
            yield nn.Dropout()
            yield nn.BatchNorm1d(layer_widths[i])
            yield nn.ReLU()

# Generate a series of blocks of (nn.Linear [, layer_type...])
# 	E.g. BlockGenerator([nn.Dropout, nn.ReLU], False).generate((1000, 500, 30, 8))
# Instead of passing in a class for each item, you can pass in a tuple where the 1st item is a class and the 2nd item is a lambda. This allows you to pass arguments to the constructor of the class you want.
# 	E.g. BlockGenerator([nn.Dropout, (nn.BatchNorm1d, lambda k, i, o: k(o)), nn.ReLU], False).generate((1000, 500, 30, 8))
class BlockGenerator(object):
	def __init__(self, layer_types):
		self.layer_types = layer_types
	def generate(self, geometry, do_coda=False):
		if not geometry: return
		inouts = zip(geometry[:-1], geometry[1:])
		# pdb.set_trace()
		for i, dimensions in enumerate(inouts):
			yield nn.Linear(*dimensions)
			if i + 2 < len(geometry) or do_coda:
				for item in self.layer_types:
					yield self._extra_layer(item, *dimensions)
	def _extra_layer(self, item, indim, outdim):
		if type(item) is tuple:
			return item[1](item[0], indim, outdim)
		else:
			return item()


class Funnel(nn.Module):
	def __init__(self, geom_basic, geom_scenic, geom_shared, **opts):
		super(Funnel, self).__init__()
		self.use_cuda = opts.get('cuda', True)
		gen = BlockGenerator([nn.Dropout, BATCHNORM_LAYER, nn.ReLU])
		self.basic = nn.Sequential(*[l for l in gen.generate(geom_basic, True)])
		self.scenic = nn.Sequential(*[l for l in gen.generate(geom_scenic, True)])
		self.shared = nn.Sequential(*[l for l in gen.generate(geom_shared)])
	def forward(self, x):
		if self.use_cuda:
			x['basic'] = x['basic'].cuda()
			x['scenic'] = x['scenic'].cuda()
		a = self.basic(x['basic'])
		b = self.scenic(x['scenic'])
		x = torch.cat((a, b), dim=1)
		return self.shared(x)

if __name__ == '__main__':
	f = Funnel([1000, 700], [60, 70], [760, 30, 70])
	print(f)
