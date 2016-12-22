#!/usr/bin/python
#prueba de red LSTM
#genera grafica de las senales de entrada y la respuesta de la redes
#entrada senoidal
from __future__ import division
import numpy as np

from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
import matplotlib.pyplot as plt

plt.close('all')     #cierra figuras anteriores

#construct target signal:
T = 1               #periodo de la senal
Nyq = 20            #minimo 2 por el teorema de Nyquist
Ts = T/Nyq           #periodo de muestreo
f = 1/T             #frecuencia de la senal
fs = 1/Ts           #frecuencia de periodo
A = 10              #amplitud
Tiempo = 5         #tiempo total de muestreo
#NN input signal:
t0 = np.arange(0,Tiempo,Ts)   #genera un vector de n hasta N, con incrementos de i (n,N,i)
#valor en el instante t0
#np.sin(Wn*t0)  Wn=2*pi*f   t0=instante de tiempo
data = A*np.cos(2*np.pi*f*t0)     #senal de entrada a la red
print 'numero de datos de entrenamiento %i'%len(data)

net = buildNetwork(1, 15, 1,hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

ds = SequentialDataSet(1, 1)
for sample, next_sample in zip(data, cycle(data[1:])):
    ds.addSample(sample, next_sample)



trainer = RPropMinusTrainer(net, dataset=ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 100
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    #print("\r epoch {}/{}".format(epoch, EPOCHS))
    stdout.flush()

print "final error =", train_errors[-1]

y0 = []#muestra
y1 = []#red
y2 = []#objetivo

for sample, target in ds.getSequenceIterator(0):
    y0.append(sample)
    y1.append(net.activate(sample))
    y2.append(target)
    #print("               sample = %4.1f" % sample)
    #print("predicted next sample = %4.1f" % net.activate(sample))
    #print("   actual next sample = %4.1f" % target)

fsize=8
t0 = np.arange(0,len(data),1)
fig1 = plt.figure(1)
plt.plot(t0, y1, 'ro',label='original')
plt.plot(t0, y2, 'k',label='red')
plt.xlabel('Time',fontsize=fsize)
plt.ylabel('Amplitude',fontsize=fsize)
plt.grid()
plt.title('Target range = [0,%0.1f]'%len(data),fontsize=fsize)
plt.xlim(1.2*np.min(t0),1.2*np.max(t0))
plt.ylim(1.2*np.min(y1),1.2*np.max(y1))

fig1name = './prueba04_b_fig1.png'
print 'Saving Fig. 1 to:', fig1name
fig1.savefig(fig1name, bbox_inches='tight')

fig2 = plt.figure(2)
plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('epoch')
plt.ylabel('error')

fig2name = './prueba04_b_fig2.png'
print 'Saving Fig. 2 to:', fig2name
fig2.savefig(fig2name, bbox_inches='tight')

plt.show()

