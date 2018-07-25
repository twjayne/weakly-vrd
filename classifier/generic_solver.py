import torch
import torch.nn as nn
import pdb

class GenericSolver:
	def __init__(self, model, optimizer, **opts):
		self.model = model
		self.optimizer = optimizer
		self.cuda = opts.get('cuda', True)
		self.supervision = opts.get('supervision', WEAK)
		self.verbose = opts.get('verbose', False)
		self.num_epochs = opts.get('num_epochs', 9)
		self.print_every = opts.get('print_every', 20)
		self.test_every = opts.get('test_every', 10)
		self.loss_fn = opts.get('loss', nn.CrossEntropyLoss())
		self.dtype = opts.get('dtype', torch.double)
	
	# @arg trainloader should be a Dataloader
	# @arg testloader should be a Dataloader
	def train(self, trainloader, testloader):
		num_iterations = self.num_epochs * len(trainloader)
		self.loss_history = torch.Tensor(num_iterations)

		if self.cuda: self.model.cuda()

		self.debug()

		iteration_i = 0
		for epoch_i in range(self.num_epochs):
			for batch_i, batch in enumerate(trainloader):
				# pdb.set_trace()
				loss = self._train_step(iteration_i, batch['X'], batch['y'])
				if self.verbose and batch_i % self.print_every == 0:
					print('(%5d/%d) loss %e' % (iteration_i, num_iterations, loss))
				if batch_i % self.test_every == 0:
					for testbatch in testloader:
						loss = self._test(testbatch['X'], testbatch['y'])
						print('=== TEST === loss %e' % (loss,))
				iteration_i += 1
	
	def _train_step(self, iteration, batch_X, batch_Y):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.loss_history[iteration] = self._calc_loss(batch_X, batch_Y)
		loss.backward()
		self.optimizer.step()
		return self.loss_history[iteration]

	def _test(self, test_X, test_Y):
		self.model.eval()
		return self._calc_loss(test_X, test_Y)
	
	def _calc_loss(self, batch_X, batch_Y):
		# if self.cuda:
		if True:
			self.model.cpu()
			batch_X.cpu()
			batch_Y.cpu()
		self.prediction = self.model(batch_X)
		return self.loss_fn( self.prediction, batch_Y.reshape(-1) )

	def debug(self):
		print('%20s %s' % ('optimizer', str(self.optimizer),))
		print('%20s %s' % ('model', str(self.model),))
		print('%20s %s' % ('cuda', self.cuda,))
		print('%20s %s' % ('supervision', self.supervision,))
		print('%20s %s' % ('verbose', self.verbose,))
		print('%20s %d' % ('num_epochs', self.num_epochs,))
		print('%20s %d' % ('print_every', self.print_every,))
		print('%20s %d' % ('test_every', self.test_every,))
		print('%20s %s' % ('loss_fn', str(self.loss_fn),))
		print('%20s %s' % ('dtype', self.dtype,))


WEAK = 'weak'
FULL = 'full'
