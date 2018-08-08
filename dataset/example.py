import numpy as np
import pdb

class Example(dict):
	def __init__(self, dic, spatial, appearance, **opts):
		for key, val in dic.items():
			self._set(key, val)
		self._set('spatial', spatial)
		self._set('appearance', appearance)
		self.n_klasses = opts.get('K', 70)
		self.dtype = opts.get('dtype', np.float64)

	def _set(self, key, val):
		self.__setattr__(key, val)
		self[key] = val

	def visual_features(self):
		return np.concatenate((self.appearance, self.spatial))

	def triplet(self):
		return (self.sub_cat[0], self.rel_cat[0], self.obj_cat[0])

	# !!! CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices
	def one_hot_label(self):
		arr = np.zeros(self.n_klasses, self.dtype)
		arr[self.rel_cat] = 1
		return arr

class BasicExample(Example):
	def __init__(self, *args):
		super(BasicExample, self).__init__(*args)
		self._set('X', self.visual_features())
		self._set('y', self.rel_cat.astype(np.long))
