#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np


#------------------------------------------------------------------------------
#   Optimizer class
#------------------------------------------------------------------------------
class Optimizer:
    def __init__(self):
        pass

    def backward(self, optimizerContext, dW, db):
        # Pri spatnom prechode moze optimizer vykonat svoju pracu nad gradientami dW, db
        pass

    def update(self, optimizerContext, W, b):
        # Uprava vah - jeden krok zostupu
        pass


#------------------------------------------------------------------------------
#   GradientDescent class
#------------------------------------------------------------------------------
class GradientDescent(Optimizer):
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def backward(self, optimizerContext, dW, db):
        # Odkladame aktualne hodnoty dW a db do noveho kontextu
        return (dW, db)

    def update(self, optimizerContext, W, b):
        # Pouzijeme gradienty z kontextu
        dW, db = optimizerContext

        # Jeden krok zostupu
        W = W - self.learningRate*dW
        b = b - self.reatningRate*db
        return W, b
