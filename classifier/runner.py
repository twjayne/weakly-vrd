import torch
import os, datetime
import classifier.classifier as classifier
import classifier.solver as solver

LEARNING_RATE = os.environ.get('lr', None)
EXTRA_DIMENSIONS = os.environ.get('extra', 500)
N_PREDCIATE_CLASSES = 70

model = classifier.Classifier(1000, 2000, EXTRA_DIMENSIONS, 4096, N_PREDCIATE_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
