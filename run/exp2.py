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
sampler_weights = [float(1/freq[rs[i]]) for i in range(len(rs))]
sampler = torch.utils.data.WeightedRandomSampler(sampler_weights)
trainloader = torch.utils.data.Dataloader(
	demo._trainset,
	sampler=sampler,
	batch_size=demo.opts.batch_size,
	shuffle=True,
	num_workers=4)

demo.train()
