#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np



#------------------------------------------------------------------------------
#   Initializer baseclass
#------------------------------------------------------------------------------
class Initializer:
    def __init__(self):
        pass

    def __call__(self, nin, nout):
        pass


#------------------------------------------------------------------------------
#   Normal class
#------------------------------------------------------------------------------
class Normal(Initializer):
    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, nin, nout):
        # Spravime nahodny tensor s rozmermi (Nout x Nin)
        t = np.random.randn(nout, nin)

        # Posunieme Mean a deviation
        t = (t * self.stddev) + self.mean
        return t


#------------------------------------------------------------------------------
#   XavierNormal class
#------------------------------------------------------------------------------
class XavierNormal(Initializer):
    def __super__(self):
        pass

    def __call__(self, nin, nout):

        # Spocitame pozadovanu varianciu
        variance = 2.0 / (nin + nout)

        # Spravime nahodny tensor s rozmermi (Nout x Nin) a prenasobime scale factorom
        t = np.random.randn(nout, nin) * np.sqrt(variance)
        return t


#------------------------------------------------------------------------------
#   HeNormal class
#------------------------------------------------------------------------------
class HeNormal(Initializer):
    def __super__(self):
        pass

    def __call__(self, nin, nout):

        # Spocitame pozadovanu varianciu
        variance = 4.0 / (nin + nout)

        # Spravime nahodny tensor s rozmermi (Nout x Nin) a prenasobime scale factorom
        t = np.random.randn(nout, nin) * np.sqrt(variance)
        return t
