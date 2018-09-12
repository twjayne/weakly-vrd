# Read a list of log files, parse the relevant lines according to a regex,
# then plot one of the fields from said lines.
# Multiple files get plotted to the same plot.

# Usage:
# 	python plot.py ../log/*.log
# E.g.
# 	python plot.py ../log/overfit/noval/N-0\ ep-15\ lr-0\ geom-1000\ *hash*.log
#	find log/ -name out.log -ctime -3 -printf '%C@\t%p\n' | sort -h | cut -f2 | python visualize

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import glob
from optparse import OptionParser
from collections import OrderedDict

import pdb

fileid_regex = re.compile(r'[^/]+/\d{12}')

common_fields = [('name', np.str_, 16),
	('epoch', np.int32),
	('batch', np.int32),
	('loss', np.float64),
	('acc', np.float32)]

REGEXES = [
	(
		re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\) : loss (\S+) : acc (\S+) : R@ (\S+) : R@ (\S+ ): R@ (\S+)', re.MULTILINE),
		common_fields + [('rec', np.float32),('rec2', np.float32),('unrel', np.float32)]
	),
	(
		re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)\s+rec (\S+)', re.MULTILINE),
		common_fields + [('rec', np.float32)]
	),
	(
		re.compile(r'^\s*(.*?)\s*\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss (\S+)\s+acc (\S+)', re.MULTILINE),
		common_fields
	),
]


class Plotter(object):
	def __init__(self, opts, *fpaths):
		self.field = opts.field
		is_multiple_logs = not not opts.split
		if is_multiple_logs:
			print('multi figs')
			label = lambda fpath: fileid_regex.search(fpath).group(0)
			data  = lambda fpath: self.get_ordered_dict(fpath)[opts.split]
			table = {label(fpath): data(fpath) for fpath in fpaths}
			ordered_dict = OrderedDict(sorted(table.items(), key=lambda tup: len(tup[1]), reverse=True))
			self.start_img(self.field)
			self.line(ordered_dict)
		else:
			print('one fig')
			for fpath in fpaths:
				ordered_dict = self.get_ordered_dict(fpath)
				self.start_img(fpath)
				self.line(ordered_dict)

	def get_ordered_dict(self, fpath):
		for regex in REGEXES:
			data = np.fromregex(fpath, *regex)
			if len(data): break
		if data == None: raise Exception('no matches for any regex')
		names = set(data['name']) # TEST, TRAIN, TRAIN_BCH, TRAIN_EP, ...
		# pdb.set_trace()
		table = {name: data[data['name'] == name] for name in names}
		return OrderedDict(sorted(table.items(), key=lambda tup: len(tup[1]), reverse=True))

	def start_img(self, *title):
		plt.figure('-'.join(title))
		self.ax = plt.subplot()
		plt.suptitle(self.field)
		plt.title('-'.join(title), fontsize=8)

	def line(self, ordered_dict):
		for key in ordered_dict:
			data = ordered_dict[key]
			print('N %12s %d' % (key, len(ordered_dict[key])))
			self.ax.plot(data['batch'], data[self.field])
		self.ax.legend(ordered_dict.keys())



if __name__ == '__main__':
	# Parse command line args
	parser = OptionParser()
	parser.add_option('-f', '--field', dest='field', default='acc')
	parser.add_option('-s', '--split', dest='split', default=None)
	parser.add_option('-l', '--last', dest='latest_only', action='store_true', default=False)
	opts, args = parser.parse_args()
	# Get logfiles
	fpaths = args if args else sys.stdin.read().strip().split('\n')
	# Run
	for fpath in fpaths:
		print(fpath)
	print('.......')
	Plotter(opts, *fpaths)
	plt.show()
