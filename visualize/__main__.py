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
import os
import sys
import glob
from optparse import OptionParser

import pdb




regex = (
	re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)', re.MULTILINE),
	 [('name', np.str_, 16),
	('epoch', np.int32),
	('batch', np.int32),
	('loss', np.float64),
	('acc', np.float32)]
)

rec_regex = (
	re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)\s+rec (\S+)', re.MULTILINE),
	regex[1] + [('rec', np.float32)]
)

class Plotter(object):
	def __init__(self, key, *fpaths):
		self.key = key
		self.labels = []
		if key == 'rec':
			self.regex = rec_regex
		else:
			self.regex = regex
		for fpath in fpaths:
			plt.figure(fpath)
			self.ax = plt.subplot()
			plt.title(fpath)
			self._line(fpath if type(fpath) is str else fpath[0])
			self.ax.legend(self.labels)
	def _line(self, fpath):
		dat = np.fromregex(fpath, *self.regex)
		srcs = set(dat['name'])
		subs = {src: dat[dat['name'] == src] for src in srcs}
		srcs = sorted(subs, key=lambda x: len(subs[x]), reverse=True)
		for src in srcs:
			sub = subs[src]
			self.ax.plot(sub['batch'], sub[self.key])
			self.labels.append(src)



if __name__ == '__main__':
	# Parse command line args
	parser = OptionParser()
	parser.add_option('-f', '--field', dest='field', default='acc')
	parser.add_option('-l', '--last', dest='latest_only', action='store_true', default=False)
	opts, args = parser.parse_args()
	# Get logfiles
	fpaths = glob.glob('log/*.log') if len(args) == 0 else args
	if opts.latest_only:
		latest_file = max(fpaths, key=os.path.getctime)
		fpaths = [latest_file]
	# Run
	Plotter(opts.field, *fpaths)
	plt.show()
