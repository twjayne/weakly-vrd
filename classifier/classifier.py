import torch
import torch.nn as nn

def linear_block(geom, **kwargs):
	for i in range(1,len(geom)-1):
		yield nn.Linear(geom[i-1], geom[i])
		yield nn.Dropout()
		if kwargs.get('batchnorm', True): yield nn.BatchNorm1d(geom[i])
		yield nn.ReLU()
	yield nn.Linear(geom[i], geom[i+1])

def sequential(geom, **kwargs):
	return nn.Sequential(*list(linear_block(geom, **kwargs)))

# Test
if __name__ == '__main__':
	print(sequential( [40, 50, 60, 17, 8] ))
