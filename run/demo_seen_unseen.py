import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(__file__,'../..')))

from torch.utils.data import DataLoader
print('Importing demo...')
import demo

import dataset.dataset as dataset
import dataset.zeroshot as zeroshot

# Split testset into seen and zeroshot sets
test_sets = zeroshot.Splitter(demo._trainset, demo._testset).split()
# Make dataloaders for new test sets
test_loaders = [DataLoader(data, batch_size=len(data), num_workers=4) for data in test_sets]
# Run solver
demo.solver.train(demo.traindata, *test_loaders)
