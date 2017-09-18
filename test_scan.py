import theano
from theano import tensor as T
import numpy as np

k = 2
A = T.vector("A")

# Symbolic description of the result
class Func:
    def __init__(self, k):
        self.k = k
    def __call__(self, x, A):
        for i in xrange(self.k):
            x = x * A
        return x

func = Func(k)

result, updates = theano.scan(fn=func,
                              outputs_info=T.ones_like(A),
                              non_sequences=[A],
                              n_steps=1)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A], outputs=final_result, updates=updates)

print power(range(10))