# Usage e.g.:
# 	util.logging.Logger('mylog.log')
# This will replace sys.stdout with a Logger instance.
# All print statements will go to sys.stdout and a log file.

import logging, sys, os

class Logger(logging.Logger):
	def __init__(self, *fname):
		super(Logger, self).__init__(fname)
		self.terminal = sys.stdout
		fname = os.path.join(*fname)
		logdir = os.path.dirname(fname)
		os.makedirs(logdir, exist_ok=True)
		print('Logging to %s' % fname)
		self.log = open(fname, "w")
		sys.stdout = self

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.terminal.flush()
		self.log.flush()

if __name__ == '__main__':
	sys.stdout = Logger('log/mylog.log')
	print('foo')
