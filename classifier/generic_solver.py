
class GenericSolver:
	def __init__(self, model, optimizer, **opts):
		self.model = model
		self.optimizer = optimizer
		self.cuda = opts.get('cuda', True)
		self.supervision = opts.get('supervision', WEAK)
		self.verbose = opts.get('verbose', False)
		self.print_every = opts.get('print_every', 1000)
	
	# @arg traindata should be a Dataloader
	# @arg testdata should be a tuple of (X,Y)
	def train(self, traindata, testdata):
		num_iterations = self.num_epochs * len(self.traindata)
		self.loss_history = torch.Tensor(num_iterations)

		if self.cuda: self.model.cuda()
		self.model.train()

		iteration_i = 0
		for epoch_i in range(self.num_epochs):
			for batch_i, batch in enumerate(traindata):
				loss = self._train_step(iteration_i, batch)
				if self.verbose and i % self.print_every == 0:
					print('(%5d/%d) loss %e' % (i, num_iterations, loss))
				if i % self.test_every == 0:
					loss = self._test(testdata)
					print('=== TEST === loss %e' % (loss,))
				iteration_i += 1
	
	def _train_step(self, iteration, batch):
		self.model.train()
		# Take step
		self.loss_history[iteration] = self._calc_loss(batch_X, batch_Y)
		self.optimizer.zero_grad()
		self.loss_history[iteration].backward()
		optimizer.step()
		return self.loss_history[iteration]

	def _test(self):
		self.model.eval()
		test_X, test_Y = self.testdata
		return self._calc_loss(test_X, test_Y)
	
	def _calc_loss(self, batch_X, batch_Y):
		self.prediction = self.model.forward(batch_X)
		return self.loss_fn(prediction, batch_Y)


WEAK = 'weak'
FULL = 'full'
