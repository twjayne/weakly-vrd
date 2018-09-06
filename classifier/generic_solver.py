import torch
import torch.nn as nn
import time
import datetime
import os
import sys
import pdb

from .loss_calculator import LossCalculator

class GenericSolver:
	def __init__(self, model, optimizer, **opts):
		opts = self.opts = {k: v for k, v in opts.items() if v is not None}
		self.model           = model
		self.optimizer       = optimizer
		self.outdir          = opts.get('outdir', None)
		self.scheduler       = opts.get('scheduler', None)
		self.cuda            = opts.get('cuda', True)
		self.supervision     = opts.get('supervision', WEAK)
		self.verbose         = opts.get('verbose', False)
		self.num_epochs      = opts.get('num_epochs', 9)
		self.print_every     = opts.get('print_every', 20)
		self.test_every      = opts.get('test_every', 80)
		self.dtype           = opts.get('dtype', torch.double)
		self.save_every      = opts.get('save_every', None)
		self.save_best       = opts.get('save_best', True)
		self.save_end        = opts.get('save_end', False)
		self.recall_every    = opts.get('recall_every', None)
		self.train_loss      = opts.get('train_loss', LossCalculator(self.model))
		self.test_loss       = opts.get('test_loss', LossCalculator(self.model))

	def init_train(self, trainloader):
		self.num_iterations = self.num_epochs * len(trainloader)
		self.loss_history = torch.Tensor(self.num_iterations)

		if self.cuda:
			self.model.cuda()

		self.debug()
		print('%20s %s' % ('num_epochs', self.num_epochs,))
		print('%20s %s' % ('num_batches', len(trainloader),))
		print('%20s %s' % ('batch_size', trainloader.batch_size,))

		self.best_val = 0
		self.iteration = 0
	
	# @arg trainloader should be a Dataloader
	# @arg testloaders should be a Dataloaders
	def train(self, trainloader, *testloaders):
		# Init dataloaders
		self.init_train(trainloader)
		testloader = testloaders[0] if len(testloaders) else None
		additional_testloaders = testloaders[1:]
		# Iterate
		for self.epoch in range(self.num_epochs):
			tic = time.time()
			for batch_i, batch in enumerate(trainloader):
				self.iteration += 1
				# Train
				self._train_step(batch)
				# Test
				if self.iteration % self.test_every == 0:
					if testloader: self._test(testloader, True)
					for additional in additional_testloaders: self._test(additional, False)
				# Save model
				if self.save_every and self.iteration % self.save_every == 0:
					self.save_checkpoint('iter-%d-acc-%f.pth.tar' % (self.iteration, self.acc))
			toc = (time.time() - tic)
			self._print('TRAIN_EP', self.train_loss.end_epoch())
			print('EP tic toc (%f) %s' % (toc, str(datetime.timedelta(seconds=toc))))
	
	def _train_step(self, batch):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.train_loss(batch)
		loss.backward()
		if self.verbose and self.iteration % self.print_every == 0:
			self._print('TRAIN_BCH', train_loss.batch_stats.compute())
		if self.scheduler and testloader is None:
			self.scheduler.step(loss.item())
		self.loss_history[self.iteration] = float(loss.data)
		self.optimizer.step()

	def _test(self, testloader, is_primary):
		self.model.eval()
		stats = self.test_loss(testloader)
		self._print(testloader.dataset.name or 'TEST', stats)
		if is_primary:
			if self.scheduler: self.scheduler.step(stats.loss)
			if self.save_best and stats.unrel_recall > self.best_val:
				self.save_checkpoint('best.pth')
				self.best_val = stats.unrel_recall

	def _calc_recall_matlab(self):
		recalls = self.evaluator.recall_from_matlab(self.model)
		return recalls

	def save_checkpoint(self, filename='checkpoint.pth'):
		print(' --- saving checkpoint --- ')
		torch.save({
			'epoch': self.epoch,
			'iteration': self.iteration,
			'model_type': str(self.model.__class__),
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'optimizer_type': str(type(self.optimizer)),
			}, os.path.join(self.outdir or '', filename))
		name, ext = os.path.splitext(filename)
		torch.save({'model': self.model},
			os.path.join(self.outdir or '', name+'whole'+ext))

	def _print(self, dataname, loss_calculator):
		sys.stdout.write('%12s (ep %3d: %5d/%d)' % (dataname, self.epoch, self.iteration, self.num_iterations))
		sys.stdout.write(' : loss %e : acc %.3f' % (loss_calculator.loss, loss_calculator.acc))
		for tensor in (loss_calculator.rec, loss_calculator.rec2, loss_calculator.unrel_recall):
			sys.stdout.write(' : R@')
			for rec in tensor: sys.stdout.write(' %.3f' % rec)
		print()

	def debug(self):
		print('%20s %s' % ('optimizer', str(self.optimizer),))
		print('%20s %s' % ('scheduler', str(self.scheduler.state_dict() if self.scheduler else None),))
		print('%20s %s' % ('model', str(self.model),))
		print('%20s %s' % ('cuda', self.cuda,))
		print('%20s %s' % ('supervision', self.supervision,))
		print('%20s %s' % ('verbose', self.verbose,))
		print('%20s %d' % ('num_epochs', self.num_epochs,))
		print('%20s %d' % ('print_every', self.print_every,))
		print('%20s %d' % ('test_every', self.test_every,))
		print('%20s %s' % ('dtype', self.dtype,))


WEAK = 'weak'
FULL = 'full'
