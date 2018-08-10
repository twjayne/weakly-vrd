import torch
import shared
import util.loss
import demo
import torch.nn as nn
import torch

demo.solver.num_epochs = 75

scheduler_lambda = lambda epoch: 0.01 * 10**-(demo.opts.lr // 25)
demo.solver.scheduler = torch.optim.lr_scheduler.LambdaLR(demo.optimizer, scheduler_lambda)

trainset = demo._trainset
rs = [x[1] for x in trainset.triplets()]
freq = torch.histc(torch.Tensor(rs), bins=70, min=0, max=69)
wt1 = 1 / freq * 10

demo.solver.loss_fn = nn.CrossEntropyLoss(wt1).double()

demo.train()
