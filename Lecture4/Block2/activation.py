#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np


#------------------------------------------------------------------------------
#   ActivationFunction class
#------------------------------------------------------------------------------
class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, z):
        pass

    def derivative(self, z):
        pass


#------------------------------------------------------------------------------
#   LinearActivationFunction class
#------------------------------------------------------------------------------
class LinearActivationFunction(ActivationFunction):
    def __call__(self, z):
        return z

    def derivative(self, z):
        return np.ones_like(z)


#------------------------------------------------------------------------------
#   ReLUActivationFunction class
#------------------------------------------------------------------------------
class ReLUActivationFunction(ActivationFunction):
    def __call__(self, z):
        return np.maximum(z, 0)

    def derivative(self, z):
        return (z > 0).astype(float)


#------------------------------------------------------------------------------
#   SigmoidActivationFunction class
#------------------------------------------------------------------------------
class SigmoidActivationFunction(ActivationFunction):
    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        a = self(z)
        return np.multiply(a, 1-a)



MAP_ACTIVATION_FUNCTIONS = {
    "linear": LinearActivationFunction,
    "relu": ReLUActivationFunction,
    "sigmoid": SigmoidActivationFunction
}

# Vytvarame funkcie podla mena. Stazujeme sa, ak narazime na neznamy typ funkcie.
def CreateActivationFunction(kind):
    if (kind in MAP_ACTIVATION_FUNCTIONS):
        return MAP_ACTIVATION_FUNCTIONS[kind]()
    raise ValueError(kind, "Unknown activation function {}".format(kind))
