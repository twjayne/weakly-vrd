# Read a list of log files, parse the relevant lines according to a regex,
# then plot one of the fields from said lines.
# Multiple files get plotted to the same plot.

# Usage:
# 	python plot.py ../log/*.log
# E.g.
# 	python plot.py ../log/overfit/noval/N-0\ ep-15\ lr-0\ geom-1000\ *hash*.log

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import glob
import pdb

pattern = re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)', re.MULTILINE)
groups = [('name', np.str_, 16),
	('epoch', np.int32),
	('batch', np.int32),
	('loss', np.float64),
	('acc', np.float32)]


class Plotter(object):
	def __init__(self, key, *fpaths):
		self.fig, self.ax = plt.subplots(num='%s %s' % (self.__class__.__name__, key))
		self.key = key
		self.labels = []
		for fpath in fpaths:
			print('1', fpath)
			self._line(fpath)
		self.ax.legend(self.labels)
	def _line(self, fpath):
		dat = np.fromregex(fpath, pattern, groups)
		srcs = set(dat['name'])
		for src in srcs:
			sub = dat[dat['name'] == src]
			self.ax.plot(sub['batch'], sub[self.key])
			self.labels.append(src)


if len(sys.argv) > 1:
	fpaths = sys.argv[1]
elif len(sys.argv) > 2:
	fpaths = sys.argv[1:]
else:
	glob.glob('log/*.log')
Plotter('acc', fpaths)
plt.show()
