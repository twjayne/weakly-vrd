import numpy as np

class Example(dict):
	def __init__(self, dic, spatial, appearance, **opts):
		for key, val in dic.iteritems():
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

	def label(self):
		arr = np.zeros(self.n_klasses, self.dtype)
		arr[self.rel_cat] = 1
		return arr
