import scipy.io

def count_features(fpath, key='features', axis=1):
	features = scipy.io.loadmat(fpath)[key]
	return features.shape[axis]

if __name__ == '__main__':
	import sys
	for fpath in sys.argv[1:]:
		print(count_features(fpath))
