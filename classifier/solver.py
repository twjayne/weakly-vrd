import generic_solver

class Solver(generic_solver.GenericSolver):
	
	def _train_step(self, iteration, batch):
		self.model.train()
		# Split up batch
		batch_X1 = 
		batch_X2 = 
		batch_Y  = 
		# Forward pass
		h1 = self.model.block1.forward(batch_X1)
		batch_X2[:len(h1)] = h1
		prediction = self.model.block2.forward(batch_X2)
		loss = self.loss_fn(prediction, batch_Y)
		# Take step
		self.loss_history[iteration] = loss
		self.optimizer.zero_grad()
		self.loss_history[iteration].backward()
		optimizer.step()
		return self.loss_history[iteration]
