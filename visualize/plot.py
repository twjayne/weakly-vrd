# Read a list of log files, parse the relevant lines according to a regex,
# then plot one of the fields from said lines.
# Multiple files get plotted to the same plot.

# Usage:
# 	python plot.py ../log/*.log
# E.g.
# 	python plot.py ../log/overfit/noval/N-0\ ep-15\ lr-0\ geom-1000\ *hash*.log

import numpy as np
import matplotlib.pyplot as plt
import sys

train_pattern = r'^\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss\s+(\S+)\s+acc\s+(\S+)'
train_groups = [('epoch', np.int32),
		('batch', np.int32),
		('loss', np.float64),
		('acc', np.float32)]
test_pattern = r'=== TEST ===\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)'
test_groups = [('loss', np.float64),
		('acc', np.float32)]

class Plotter:
	def __init__(self, pattern, groups):
		self.pattern = pattern
		self.groups = groups
	def _line(self, fpath, ax, key):
		dat = np.fromregex(fpath, self.pattern, self.groups)
		x = dat['batch'] if 'batch' in dat.dtype.names else np.arange(len(dat[key]))
		ax.plot(x, dat[key])
	def plot(self, fpaths, key):
		fig, ax = plt.subplots(num='%s %s' % (self.__class__.__name__, key))
		for fpath in fpaths:
			self._line(fpath, ax, key)
		ax.legend(fpaths, prop={'size': 6})

class TestPlotter(Plotter):
	def __init__(self):
		super(TestPlotter, self).__init__(test_pattern, test_groups)


class TrainPlotter(Plotter):
	def __init__(self):
		super(TrainPlotter, self).__init__(train_pattern, train_groups)

TestPlotter().plot(sys.argv[1:], 'acc')
TrainPlotter().plot(sys.argv[1:], 'acc')
plt.show()
