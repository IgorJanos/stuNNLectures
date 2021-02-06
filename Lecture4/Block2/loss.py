#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np


#------------------------------------------------------------------------------
#   LossFunction
#------------------------------------------------------------------------------
class LossFunction:
    def __init__(self):
        pass

    def __call__(self, a, y):
        pass

    def derivative(self, a, y):
        pass


#------------------------------------------------------------------------------
#   BinaryCrossEntropyLossFunction
#------------------------------------------------------------------------------
class BinaryCrossEntropyLossFunction(LossFunction):
    def __call__(self, a, y):
        return -( np.multiply(y, np.log(a)) + np.multiply((1-y), np.log(1-a)) )

    def derivative(self, a, y):
        return -np.divide(y, a) + np.divide((1-y), (1-a))


MAP_LOSS_FUNCTIONS = {
    "bce": BinaryCrossEntropyLossFunction
}

def CreateLossFunction(kind):
    if (kind in MAP_LOSS_FUNCTIONS):
        return MAP_LOSS_FUNCTIONS[kind]()
    raise ValueError(kind, "Unknown loss function {}".format(kind))
