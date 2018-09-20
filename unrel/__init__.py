import sys, os
thisdir = os.path.dirname(__file__)
parentdir = os.path.dirname(thisdir)
if not parentdir in sys.path: sys.path.append(parentdir)
