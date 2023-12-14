"""
Adapted from:
https://github.com/tadeephuy/GradientReversal

Citation:
@misc{ganin2015unsupervised,
      title={Unsupervised Domain Adaptation by Backpropagation},
      author={Yaroslav Ganin and Victor Lempitsky},
      year={2015},
      eprint={1409.7495},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
"""

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x, alpha):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply


class GradientReversalModule(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
