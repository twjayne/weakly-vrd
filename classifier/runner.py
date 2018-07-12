import torch
import os, datetime


LEARNING_RATE = os.environ.get('lr', None)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
