#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------
import numpy as np

from backstage.utils import MakeBatches

#------------------------------------------------------------------------------
#   Model class
#------------------------------------------------------------------------------
class Model:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optimizer = None

    def initialize(self, loss, optimizer):
        # Nase vlastne si odlozime
        self.loss = loss
        self.optimizer = optimizer

        # 1. Inicializujeme vsetky vrstvy
        prevLayer = None
        for l in self.layers:
            l.initialize(prevLayer)
            prevLayer = l  

    def forward(self, x):
        # 2. Priamy prechod - predikcia
        a = x
        for l in self.layers:
            a, _ = l.forward(a, False)      # Neucime !
        return a

    def __call__(self, x):
        # Len 2 triedy nas tu zaujimaju
        a = self.forward(x)
        a = (a > 0.5).astype(float)
        return a

    def _trainSingleStep(self, x, y):
        #
        #   Jeden krok ucenia:
        #           x = minibatch z trenovacej mnoziny
        #           y = minibatch label
        #
        #   Vraciame:
        #           loss = vektor strat pre davku (pozor! toto nie je cost!)
        #

        #   1. Priamy prechod - vypocitame hodnotu loss
        #   2. Spatny prechod - vypocitame hodnoty gradientov (vnutorne data vrstvy/optimizer kontextu)
        #   3. Uprava parametrov - zostup podla gradientu

        #   1. Priamy prechod
        a = x
        forwardCache = []
        for l in self.layers:
            aNext, layerCache = l.forward(a, True)      # isTraining = True
            
            # Odkladame si (Vrstvu, cache data z vrstvy - (W, b, z), a aktivaciu z predchadzajucej vrstvy)
            forwardCache.append((l, layerCache, a))
            a = aNext

        # Spocitame hodnotu straty - vektor pre kazdu vzorku z minibatchu
        # a spocitame derivaciu stratovej funkcie podla poslednej aktivacie.
        loss = self.loss(a, y)
        da = self.loss.derivative(a, y)

        #   2. Spatny prechod - v opacnom poradi od konca k zaciatku
        for l, layerCache, aPrev in reversed(forwardCache):
            daPrev = l.backward(da, aPrev, layerCache, self.optimizer)
            da = daPrev

        #   3. Update parametrov
        for l in self.layers:
            l.update(self.optimizer)

        return loss

    def train(self, X, Y, epochs, batchSize, devX=None, devY=None, verboseInterval=1000):

        # 3. Ucenie - budeme vykonavat iteracie (epochy) cez nasu trenovaciu mnozinu.

        # Nasekame trenovaciu mnozinu
        _, m = X.shape
        data = MakeBatches((X, Y), batchSize)

        # Budeme vykonavat aj validaciu ?
        doValidation = True
        if (devX is None or devY is None):
            doValidation = False

        # Dictionary pre vysledky
        results = {
            "epochs": [],
            "loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):

            # Akumulator pre training loss
            trainLossAcc = 0.0
            for batchX, batchY in data:
                loss = self._trainSingleStep(batchX, batchY)               
                trainLossAcc += np.sum(loss)

            # Odkladame si vysledok za tuto epochu
            results["epochs"].append(epoch)
            results["loss"].append((1.0 / m) * trainLossAcc)       # Toto je Cost J !

            # Validacia ?
            if (doValidation):
                yhat = self.forward(devX)
                _, devm = yhat.shape
                valLoss = self.loss(yhat, devY)
                results["val_loss"].append((1.0 / devm) * np.sum(valLoss))       # Aj toto je Cost J !
        
            # A este raz za 1000 epoch napiseme aktualny stav
            if (epoch % verboseInterval == 0):
                if (doValidation):
                    print('Epoch {0}:  Loss = {1:.7f}   Val_Loss = {2:.7f}'.format(
                        epoch, results["loss"][epoch], results["val_loss"][epoch])
                    )
                else:
                    print('Epoch {0}:  Loss = {1:.7f}'.format(epoch, results["loss"][epoch]))

        # Final
        print('Training complete.')
        if (doValidation):
            print('Epoch {0}:  Loss = {1:.7f}   Val_Loss = {2:.7f}'.format(
                epoch, results["loss"][epoch], results["val_loss"][epoch])
            )
        else:
            print('Epoch {0}:  Loss = {1:.7f}'.format(epoch, results["loss"][epoch]))

        return results


    