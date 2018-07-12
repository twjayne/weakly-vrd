import generic_solver

class Solver(generic_solver.GenericSolver):
	
	def _train_step(self, iteration, batch):
		self.model.train()
		# Allocate variables to hold concatenated inputs (and then feed into the model layers)
		N = len(batch[LBL_PREDICATE])
		if 'stage2_X' not in dir(self):
			F = model.stage2[0].in_features
			self.stage2_X = torch.autograd.Variable(torch.Tensor( N, F ))
		if 'stage1_X' not in dir(self):
			F = model.stage1[0].in_features
			self.stage1_X = torch.autograd.Variable(torch.Tensor( N, F ))
		# Forward pass
		self.stage1_X.data = torch.cat((batch[LBL_APPEARANCE], batch[LBL_SPATIAL]), dim=1)
		h1 = self.model.block1.forward(self.stage1_X)
		self.stage2_X.data = torch.cat((h1, batch[LBL_STAGE2]), dim=1)
		prediction = self.model.block2.forward(self.stage2_X)
		loss = self.loss_fn(prediction, batch[LBL_PREDICATE])
		# Take step
		self.loss_history[iteration] = loss
		self.optimizer.zero_grad()
		self.loss_history[iteration].backward()
		optimizer.step()
		return self.loss_history[iteration]

LBL_PREDICATE = 'predicate'
LBL_APPEARANCE = 'spatial'
LBL_SPATIAL = 'spatial'
LBL_STAGE2 = 'extra'
