import torch
import torch.nn as nn
import time
import os
import pdb

from .loss_calculator import LossCalculator
from RecallEvaluator import RecallEvaluator

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
		self.save_end        = opts.get('save_end', False)
		self.recall_every    = opts.get('recall_every', None)
		self.loss_calculator = opts.get('loss_calculator', LossCalculator(self.model))

	def init_train(self, trainloader):
		self.num_iterations = self.num_epochs * len(trainloader)
		self.loss_history = torch.Tensor(self.num_iterations)

		# initialize evaluator
		if self.recall_every:
			self.evaluator = RecallEvaluator()

		if self.cuda:
			self.model.cuda()

		self.debug()
		print('%20s %s' % ('num_epochs', self.num_epochs,))
		print('%20s %s' % ('num_batches', len(trainloader),))
		print('%20s %s' % ('batch_size', trainloader.batch_size,))

		self.best_acc = 0
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
			for batch_i, batch in enumerate(trainloader):
				# Train
				tic = time.time()
				loss = self._train_step(batch)
				toc = (time.time() - tic) / (len(batch['y']) if 'y' in batch else batch['N'])
				if self.verbose and batch_i % self.print_every == 0:
					self._print(loss, 'TRAIN')
				if self.scheduler and testloader is None:
					self.scheduler.step(loss.item())
				# Test
				if self.iteration % self.test_every == 0:
					if testloader: self._test(testloader, True)
					for additional in additional_testloaders: self._test(additional, False)
				# Calc recall (external) todo: replace with translation
				if self.recall_every and self.iteration % self.recall_every == 0:
					recalls = self._calc_recall()
					print('RECALL\t%s' % (' '.join(recalls.values(),)))
				# Save model
				if self.save_every and self.iteration % self.save_every == 0:
					self.save_checkpoint('iter-%d-acc-%f.pth.tar' % (self.iteration, self.acc))
				self.iteration += 1
	
	def _train_step(self, batch):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.loss_calculator.calc(batch)
		loss.backward()
		self.loss_history[self.iteration] = float(loss.data)
		self.optimizer.step()
		return loss

	def _test(self, testloader, is_primary, do_recall=True):
		self.model.eval()
		loss = self.loss_calculator.calc(testloader, do_recall)
		extra_strings = []
		if do_recall: extra_strings.append('rec %.2f %.2f' % self.loss_calculator.recall(50), self.loss_calculator.recall(100))
		self._print(loss, testloader.dataset.name or 'TEST', *extra_strings)
		if self.scheduler: self.scheduler.step(loss.item())
		if self.opts.get('save_best', False) and self.acc > self.best_acc:
			self.save_checkpoint('best.pth')
			self.best_acc = self.acc

	def _calc_recall(self, testloader):
		recalls = self.evaluator.evaluate_recall(self.model, self.supervision, False) # False for language scroes (not using language scores, yet)
		return recalls

	def _calc_recall_matlab(self):
		recalls = self.evaluator.recall_from_matlab(self.model)
		return recalls

	def save_checkpoint(self, filename='checkpoint.pth'):
		print(' --- saving checkpoint --- ')
		torch.save({
			'epoch': self.epoch,
			'iteration': self.iteration,
			'model_type': str(type(self.model)),
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'optimizer_type': str(type(self.optimizer)),
			}, os.path.join(self.outdir or '', filename))

	def _print(self, loss, dataname='TRAIN', *extra_strings):
		text = '%12s (ep %3d: %5d/%d) loss %e\tacc %.3f' % (dataname, self.epoch, self.iteration, self.num_iterations, loss, self.loss_calculator.acc)
		for string in extra_strings: text += '\t%s' % string
		print(text)

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
