import torch
import numpy as np
from torch.optim import Optimizer

from ..config import TrainOptimizerConfig


class ScheduledOptim(Optimizer):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model: torch.nn.Module, train_optimizer_config: TrainOptimizerConfig, init_lr, current_step):
        super(ScheduledOptim, self).__init__(model.parameters(), {})
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_optimizer_config.betas,
            eps=train_optimizer_config.eps,
            weight_decay=train_optimizer_config.weight_decay,
        )
        self.n_warmup_steps = train_optimizer_config.warm_up_step
        self.anneal_steps = train_optimizer_config.anneal_steps
        self.anneal_rate = train_optimizer_config.anneal_rate
        self.current_step = current_step if current_step else 0
        self.init_lr = init_lr

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        self._optimizer.zero_grad(set_to_none)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

# if __name__ == "__main__":
#     optim = ScheduledOptim(
#         torch.nn.Linear(10, 1),
#         train_config={
#             "optimizer" : {
#                 "betas": [0.9, 0.98],
#                 "eps": 0.000000001,
#                 "weight_decay": 0.0,
#                 "warm_up_step": 4000,
#                 "anneal_steps": [300000, 400000, 500000],
#                 "anneal_rate": 0.3,
#             }
#         },
#         model_config = {
#             "transformer": {
#                 "encoder_hidden": 256,
#             }
#         }, 
#         current_step=0)
#     a = optim.state_dict()
#     print(a)