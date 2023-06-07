from mindspore import nn, Tensor, context, ops
import mindspore as ms
import numpy as np
# graph mode, ms 2.0.0rc1
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
#context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        pass
    def construct(self, x):
        mask = (x == 0)
        indexes = []
        for i in range(x.shape[0]):
            tx = x[i]
            index = (tx == 0).nonzero().squeeze()[1:] # do not include the first zero index
            index = ops.stack([ops.ones(index.shape, dtype=index.dtype)*i, index], axis=-1)
            indexes.append(index)
        indexes = ops.concat(indexes, axis=0)
        out = ops.TensorScatterUpdate()(x, indexes, ops.ones(indexes.shape[0], x.dtype)*-10) # works in pynative mode; under graph mode, ones_op input1(shape) float64 error
        return out
# class Net(nn.Cell):
#     def __init__(self):
#         super().__init__()
#         pass
#     def construct(self, x):
#         mask = (x != 0)
#         for i in range(x.shape[0]):
#             tx = x[i]
#             index = (tx == 0).nonzero().squeeze()[0]
#             mask[i, index] = True
#         x[~mask] = -10
#         return x
input_x = Tensor(np.array([[10, 10, 10, 0, 0, 0],
                           [3, 3, 0, 0, 0, 0]]), ms.float32)

# in each row, find the index of the first zero value, and replace the elements from index+1 by -10
network = Net()
y = network(input_x)
print(y)
"""
[[10, 10, 10, 0, -10, -10],
[3, 3, 0, -10, -10, -10]]
"""