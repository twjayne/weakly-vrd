import torch
import torch.nn as nn
import time
import os
import pdb

from RecallEvaluator import RecallEvaluator

class GenericSolver:
	def __init__(self, model, optimizer, **opts):
		opts = self.opts = {k: v for k, v in opts.items() if v is not None}
		self.model       = model
		self.optimizer   = optimizer
		self.outdir      = opts.get('outdir', None)
		self.scheduler   = opts.get('scheduler', None)
		self.cuda        = opts.get('cuda', True)
		self.supervision = opts.get('supervision', WEAK)
		self.verbose     = opts.get('verbose', False)
		self.num_epochs  = opts.get('num_epochs', 9)
		self.print_every = opts.get('print_every', 20)
		self.test_every  = opts.get('test_every', 80)
		self.loss_fn     = opts.get('loss', nn.CrossEntropyLoss())
		self.dtype       = opts.get('dtype', torch.double)
		self.save_every  = opts.get('save_every', None)
		self.save_end    = opts.get('save_end', False)
		self.recalls     = {}
	
	# @arg trainloader should be a Dataloader
	# @arg testloaders should be a Dataloaders
	def train(self, trainloader, testloader, *additional_testloaders):
		self.num_iterations = self.num_epochs * len(trainloader)
		self.loss_history = torch.Tensor(self.num_iterations)
		# initialize evaluator
		self.evaluator = RecallEvaluator('/home/SSD2/tyler-data/unrel/data',"/home/tylerjan/code/vrd/unrel","/home/tylerjan/code/vrd/unrel/scores")

		if self.cuda:
			self.model.cuda()
			self.loss_fn.cuda()

		self.debug()
		print('%20s %s' % ('num_epochs', self.num_epochs,))
		print('%20s %s' % ('num_batches', len(trainloader),))
		print('%20s %s' % ('batch_size', trainloader.batch_size,))

		self.best_acc = 0
		self.iteration = 0
		for self.epoch in range(self.num_epochs):
			for batch_i, batch in enumerate(trainloader):
				# Train
				tic = time.time()
				loss = self._train_step(batch['X'], batch['y'])
				toc = (time.time() - tic) / len(batch['y'])
				if self.verbose and batch_i % self.print_every == 0:
					self._print(loss, 'TRAIN')
				if self.scheduler and testloader is None:
					self.scheduler.step(loss)
				# Test
				if self.iteration % self.test_every == 0:
					if testloader: self._test_primary(testloader)
					# Test extra testsets
					if additional_testloaders:
						for additional in additional_testloaders: self._test(additional)
				# Save model
				if self.save_every and self.iteration % self.save_every == 0:
					self.save_checkpoint('iter-%d-acc-%f.pth.tar' % (self.iteration, self.acc))
				self.iteration += 1
	
	def _train_step(self, batch_X, batch_Y):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self._calc_loss(batch_X, batch_Y)
		loss.backward()
		self.loss_history[self.iteration] = float(loss.data)
		self.optimizer.step()
		return loss

	def _test_primary(self, testloader):
		loss = self._test(testloader)
		if self.scheduler: self.scheduler.step(loss)
		if self.opts.get('save_best', False) and self.acc > self.best_acc:
			self.save_checkpoint('best.pth')
			self.best_acc = self.acc

	def _test(self, testloader):
		self.model.eval()
		for testbatch in testloader:
			loss = self._calc_loss(testbatch['X'], testbatch['y'])
			# self.recalls = self._calc_recall(self.model)
			self._print(loss, testloader.dataset.name or 'TEST')
		return loss

	def _calc_loss(self, batch_X, batch_Y):
		if self.cuda:
			if type(batch_X) is torch.Tensor: batch_X = batch_X.cuda()
			if type(batch_Y) is torch.Tensor: batch_Y = batch_Y.cuda()
		self.prediction = self.model(batch_X)
		correct_predictions = torch.sum( torch.argmax(self.prediction, 1) == batch_Y.transpose(1,0) ).item()
		self.acc = correct_predictions / float(batch_Y.shape[0])
		return self.loss_fn( self.prediction, batch_Y.reshape(-1) )

	def _calc_recall(self):
		recalls = self.evaluator.recall_from_matlab(self.model)
		print(f"recalls: {recalls}")
		return recalls

	def save_checkpoint(self, filename='checkpoint.pth'):
		print(' --- saving checkpoint --- ')
		torch.save({
			'epoch': self.epoch,
			'iteration': self.iteration,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			}, os.path.join(self.outdir or '', filename))

	def _print(self, loss, dataname='TRAIN'):
		print('%12s (ep %3d: %5d/%d) loss %e\tacc %.3f' % (dataname, self.epoch, self.iteration, self.num_iterations, loss, self.acc))

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
