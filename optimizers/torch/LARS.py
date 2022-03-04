import torch
from torch.optim import Optimizer

class LARS(Optimizer):

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        defaults = dict(lr=lr, beta=beta, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1 = group['beta']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad.add_(group['weight_decay'], p.data))
                step_size = group['lr']

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                exp_avg_norm = exp_avg.pow(2).sum().sqrt()

                if weight_norm == 0 or exp_avg_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / exp_avg_norm

                state['weight_norm'] = weight_norm
                state['trust_ratio'] = trust_ratio
                state['exp_avg_norm'] = exp_avg_norm

                p.data.add_(-step_size * trust_ratio, exp_avg)

        return loss