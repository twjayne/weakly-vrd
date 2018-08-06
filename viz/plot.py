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

pattern = r'\(ep\s+(\d+):\s+(\d+)/\d+\)\s+loss\s+(\S+)\s+acc\s+(\S+)'
groups = [('epoch', np.int32),
		('batch', np.int32),
		('loss', np.float64),
		('acc', np.float32)]

def plot(key):
	data = [np.fromregex(arg, pattern, groups) for arg in sys.argv[1:]]
	XY = [(item['batch'], item[key]) for item in data]
	XY = [item for sublist in XY for item in sublist]

	fig, ax = plt.subplots(num=key)
	p = ax.plot(*XY)
	ax.legend([str(i) for i in range(len(data))])

	for i, line in enumerate(p):
		print('%10s  %s' % (line.get_color(), sys.argv[i+1]))

def plot1(key):
	plt.figure(num=key)
	for arg in sys.argv[1:]:
		dat = np.fromregex(arg, pattern, groups)
		p = plt.plot(dat['batch'], dat[key])

plot('loss')
plot('acc')

plt.show()
