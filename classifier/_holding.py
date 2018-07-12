
		if self.supervision == WEAK:
			train_fn = self._train_weak
		elif self.supervision == FULL:
			train_fn = self._train_full
		else:
			raise Exception('Illegal supervision type specified "%s"' % str(self.supervision))
