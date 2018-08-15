from . import dataset
import numpy as np
import torch
import util.config as config

class FunnelDataset(dataset.DatasetWithScenicFeatures):
	def __getitem__(self, i):
		return {
			'X': {
				'basic': torch.from_numpy(np.concatenate((self._appearance_features(i), self._spatial_features(i)))).type(config.dtype),
				'scenic': self._scenic_features(i),
			},
			'y': self.pairs['rel_cat'][i].astype(np.long),
		}
