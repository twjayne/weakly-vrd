import torch
import shared
import util.loss
import demo


# experiment
# 	1/freq
# 	25 ep 0.01
# 	25 ep 0.001
# 	25 ep 0.0001

# experiment
# 	1/freq x 10
# 	25 ep 0.01
# 	25 ep 0.001
# 	25 ep 0.0001

# experiment
# 	sample by class

demo.solver.num_epochs = 75

scheduler_lambda = lambda epoch: 0.01 * 10**-(demo.opts.lr // 25)
demo.solver.scheduler = torch.optim.lr_scheduler.LambdaLR(demo.optimizer, scheduler_lambda)

demo.solver.loss_fn = util.loss.weighted_crossentropy_loss(demo.trainloader.dataset)
demo.train()
