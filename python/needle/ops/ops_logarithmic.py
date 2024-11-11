from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z = Tensor(Z, dtype='float32')
        x = logsumexp(z, axes=1)
        return (z - reshape(x, (x.shape[0], 1))).numpy()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        return out_grad - out_grad * grad_of_logsumexp
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
                     
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        print(f"Z:{Z.shape}")
        print(f"axes:{self.axes}")
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        print(f"max_Z:{max_Z.shape}")
        x = array_api.exp(Z - max_Z)
        x = array_api.sum(x, axis=self.axes)
        x = array_api.log(x) + array_api.max(Z, axis=self.axes, keepdims=False)
        return x
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes is None:   # edge case: if axes is None
            return out_grad * exp(Z - node)
        else:
            original_axes = array_api.max(Z.numpy(), axis=self.axes, keepdims=True).shape
            out = reshape(out_grad, original_axes)
            return out * exp(Z - reshape(node, original_axes))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

