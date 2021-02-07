#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np

from activation import CreateActivationFunction


#------------------------------------------------------------------------------
#   Layer class
#------------------------------------------------------------------------------
class Layer:
    def __init__(self, act='linear'):
        # Default shape
        self.shape = (0, 0)
        # Kazda vrstva by mala mat svoju aktivacnu funkciu. Vieme ju vytvarat podla mena.
        self.activation = CreateActivationFunction(act)

    def initialize(self, prevLayer):
        # 1. Inicializacia parametrov - vahy, bias
        #    Mame pristup k predchadzajucej vrstve, aby sme vedeli zistit jej shape
        pass

    def forward(self, x):
        # 2. Priamy prechod
        #    Na vstupe mame aktivaciu predchadzajucej vrstvy
        #    Vystupom je aktivacia a cache
        pass

    def backward(self, da, aPrev, cache, optimizer):
        # 3. Spatny prechod
        #    Na vstupe mame:
        #        da        = dL/da aktualnej vrstvy
        #        aPrev     = aktivacia predchadzajucej vrstvy
        #        cache     = medzivysledky z priameho prechodu
        #        optimizer = instancia optimizera, s ktorym spolupracujeme
        pass

    def update(self, optimizer):
        # 4. Zostup podla gradientu
        #    Na vstupe mame:
        #        optimizer = instancia optimizera, s ktorym zostup vykonavame
        pass


#------------------------------------------------------------------------------
#   Input class
#------------------------------------------------------------------------------
class Input(Layer):
    def __init__(self, nx):
        super().__init__(act='linear')

        # Vstupny tvar
        self.nx = nx

    def initialize(self, prevLayer):
        self.shape = (self.nx, 1)

    def forward(self, x):
        return x, None

    def backward(self, da, aPrev, cache, optimizer):
        return None, None


#------------------------------------------------------------------------------
#   Dense class
#------------------------------------------------------------------------------
class Dense(Layer):
    def __init__(self, nunits, act='linear'):
        super().__init__(act)        
        self.nunits = nunits             # Pocet neuronov vrstvy
        self.W = None                    # Maticu vah zatial nemame
        self.b = None                    # Bias zatial nemame
        self.optimizerContext = None     # Kontext zatial nemame

    def initialize(self, prevLayer):
        # 1. Inicializacia

        # Najskor spocitame nas shape
        self.shape = (self.nunits, 1)
        self.optimizerContext = None

        # Potom ziskame shape z predchadzajucej vrstvy, aby sme vedeli 
        # urcit rozmer pre maticu vah
        pnx, _ = prevLayer.shape
        nx, _ = self.shape

        # Inicializujeme vahy a bias
        self.W = np.random.randn(nx, pnx)
        self.b = np.zeros((nx, 1), dtype=float)

    def forward(self, x):
        # 2. Priamy prechod
        z = np.matmul(self.W, x) + self.b     # z = Wx + b
        a = self.activation(z)                # a = activation(z)

        # Vratime aktivaciu a cache
        return a, (self.W, self.b, z)

    def backward(self, da, aPrev, cache, optimizer):
        # 3. Spatny prechod

        # Rozbalime medzivysledky z cache
        W, b, z = cache

        # Vektorizacia - potrebujeme vediet pocet m
        _, m = da.shape

        # Spocitame spatny prechod
        dz = np.multiply(da, self.activation.derivative(z))
        dW = (1.0/m) * np.matmul(dz, aPrev.T)
        db = (1.0/m) * np.sum(dz, axis=1, keepdims=True)

        # Vysledok - da pre predchadzajucu vrstvu 
        daPrev = np.matmul(W.T, dz)

        # Spolupraca s optimizerom - ukazeme mu gradienty dW, db
        # a dostaneme novy kontext
        self.optimizerContext = optimizer.backward(self.optimizerContext, dW, db)
        return daPrev

    def update(self, optimizer):
        # 4. Uprava parametrov - spolupraca s optimizerom
        self.W, self.b = optimizer.update(self.optimizerContext, self.W, self.b)


