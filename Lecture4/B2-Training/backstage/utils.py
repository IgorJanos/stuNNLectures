#------------------------------------------------------------------------------
#
#   FIIT - Neuronove Siete - Prednaska 4
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt



def dataset_Circles(m=10, radius=0.7, noise=0.0, verbose=False):

    # Hodnoty X budu v intervale <-1; 1>
    X = (np.random.rand(2, m) * 2.0) - 1.0
    if (verbose): print('X: \n', X, '\n')

    # Element-wise nasobenie nahodnym sumom
    N = (np.random.rand(2, m)-0.5) * noise
    if (verbose): print('N: \n', N, '\n')
    Xnoise = X + N
    if (verbose): print('Xnoise: \n', Xnoise, '\n')

    # Spocitame polomer
    # Element-wise druha mocnina
    XSquare = Xnoise ** 2
    if (verbose): print('XSquare: \n', XSquare, '\n')

    # Spocitame podla prvej osi. Ziskame (1, m) array.
    RSquare = np.sum(XSquare, axis=0, keepdims=True)
    if (verbose): print('RSquare: \n', RSquare, '\n')
    R = np.sqrt(RSquare)
    if (verbose): print('R: \n', R, '\n')

    # Y bude 1, ak je polomer vacsi ako argument radius
    Y = (R > radius).astype(float)
    if (verbose): print('Y: \n', Y, '\n')

    # Vratime X, Y
    return X, Y


def dataset_Flower(m=10, noise=0.0):
    # Inicializujeme matice
    X = np.zeros((m, 2), dtype='float')
    Y = np.zeros((m, 1), dtype='float')

    a = 1.0
    pi = 3.141592654
    M = int(m/2)

    for j in range(2):
        ix = range(M*j, M*(j+1))
        t = np.linspace(j*pi, (j+1)*pi, M) + np.random.randn(M)*noise
        r = a*np.sin(4*t) + np.random.randn(M)*noise
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y


def draw_Scatter(X, Y):

    # 9 x 9 palcovy obrazok s linearnou mierkou
    plt.figure(figsize=(9, 9))
    plt.xscale('linear')
    plt.yscale('linear')

    # Nakreslime scatter graf
    plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.RdBu)
    plt.show()
    plt.close()


def draw_DecisionBoundary(X, Y, model, size=6):

    # Najdeme hranice, pre ktore budeme skumat predikciu
    pad = 0.5
    x1_Min, x1_Max = X[0,:].min()-pad, X[0,:].max()+pad
    x2_Min, x2_Max = X[1,:].min()-pad, X[1,:].max()+pad

    # Spravime mriezku dvojic - vzorkujeme cely interval <MIN; MAX> s granularitou h
    h = 0.01  
    x1_Grid, x2_Grid = np.meshgrid(
        np.arange(x1_Min, x1_Max, h),
        np.arange(x2_Min, x2_Max, h)
    )

    # Usporiadame si mriezku hodnot do rovnakeho tvaru ako ma X
    XX = np.c_[x1_Grid.ravel(), x2_Grid.ravel()].T

    # Vypocitame predikciu pomocou modelu na vsetky hodnoty mriezky
    YHat = model(XX)

    # A usporiadame si vysledok tak, aby sme ho mohli podhodit PyPlotu
    YHat = YHat.reshape(x1_Grid.shape)

    # Najskor nakreslime contour graf - vysledky skumania pre mriezku
    plt.figure(figsize=(size, size))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.contourf(x1_Grid, x2_Grid, YHat, cmap=plt.cm.RdYlBu)

    # Potom este pridame scatter graf pre X, Y
    plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.RdBu)
    
    plt.show()
    plt.close()


def MakeBatches(dataset, batchSize):
    # Set obsahuje 2 mnoziny - X, Y
    X, Y = dataset

    # Zistime celkovy pocet vzoriek
    nx, m = X.shape
    ny, _ = Y.shape

    # Vysledny zoznam
    result = []

    # Ak je batchSize = 0, berieme celu mnozinu
    if (batchSize <= 0):
        batchSize = m

    # Celkovy pocet davok sa zaokruhluje nahor
    steps = int(np.ceil(m / batchSize))
    for i in range(steps):

        # Spocitame hranice rezu
        mStart = i * batchSize
        mEnd = min(mStart + batchSize, m)

        # Vyberame data pre aktualny rez - chceme dodrzat rank
        minibatchX = X[:,mStart:mEnd]
        minibatchY = Y[:,mStart:mEnd]

        assert(len(minibatchX.shape) == 2)
        assert(len(minibatchY.shape) == 2)

        # Pridame novu dvojicu do vysledneho zoznamu
        result.append((minibatchX, minibatchY))

    return result

def draw_TrainingResults(results, size=6):

    # Inicializacia - logaritmicka mierka na Y osi
    plt.figure(figsize=(size, size*3/4))
    plt.xlabel('Epoch')
    plt.xscale('linear')
    plt.yscale('log')

    # Dva grafy - training loss, validation loss
    plt.plot(results["epochs"], results["loss"], 'b', label='loss')
    plt.plot(results["epochs"], results["val_loss"], 'r', label='val_loss')
    
    plt.legend()
    plt.show()
    plt.close()

