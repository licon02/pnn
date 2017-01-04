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
from mpi4py import MPI
import sys

plt.close('all')     #cierra figuras anteriores

#construct target signal:
T = 1               #periodo de la senal
Nyq = 40            #minimo 2 por el teorema de Nyquist
Ts = T/Nyq           #periodo de muestreo
f = 1/T             #frecuencia de la senal
fs = 1/Ts           #frecuencia de periodo
A = 1              #amplitud
Tiempo = 1        #tiempo total de muestreo
#NN input signal:
t0 = np.arange(0,Tiempo,Ts)   #genera un vector de n hasta N, con incrementos de i (n,N,i)
#valor en el instante t0
#np.sin(Wn*t0)  Wn=2*pi*f   t0=instante de tiempo
data = A*np.sin(2*np.pi*f*t0)     #senal de entrada a la red


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
print ''
sys.stdout.write("Hello, World! I am process %d of %d on %s.\n"% (rank, size, name))
comm = MPI.COMM_WORLD

print 'numero de datos de entrenamiento %i'%len(data)

net = buildNetwork(1, 25, 1,hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

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

L = len(data)
out = np.zeros(L)
aux3 = np.zeros(L)
aux4 = np.zeros(L)

for sample, target in ds.getSequenceIterator(0):
    y0.append(sample)
    y1.append(net.activate(sample))
    y2.append(target)
    #print("               sample = %4.1f" % sample)
    #print("predicted next sample = %4.1f" % net.activate(sample))
    #print("   actual next sample = %4.1f" % target)

for c in range(len(y1)):
    out[c] = y1[c]
aux3 = out
aux4 = out
print out

# Draw all charts
if name == 'Pi03':
    aux3 = out
    comm.Send(aux3, dest = 0)
if name == 'Pi04':
    aux4 = out
    comm.Send(aux4, dest = 0)
if name == 'Pi01':
    fsize=8
    fig1 = plt.figure()
    plt.plot(t0, y1, 'ro',label='original')
    plt.plot(t0, y2, 'k',label='red')
    plt.xlabel('Time',fontsize=fsize)
    plt.ylabel('Amplitude',fontsize=fsize)
    plt.grid()
    plt.title('Target range = [0,%0.1f]'%len(data),fontsize=fsize)
    plt.xlim(1.2*np.min(t0),1.2*np.max(t0))
    plt.ylim(1.2*np.min(y1),1.2*np.max(y1))
    if size >=2:
        comm.Recv(aux3,source = 1)
        plt.figure()
        plt.plot(t0, y1, 'ro',label='original')
        plt.plot(t0, aux3, 'k',label='red')
        plt.xlabel('Time',fontsize=fsize)
        plt.ylabel('Amplitude',fontsize=fsize)
        plt.grid()
        plt.title('Target range = [0,%0.1f]'%len(data),fontsize=fsize)
        plt.xlim(1.2*np.min(t0),1.2*np.max(t0))
        plt.ylim(1.2*np.min(y1),1.2*np.max(y1))
    if size >= 3:
        comm.Recv(aux4,source = 2)
        plt.figure()
        plt.plot(t0, y1, 'ro',label='original')
        plt.plot(t0, aux4, 'k',label='red')
        plt.xlabel('Time',fontsize=fsize)
        plt.ylabel('Amplitude',fontsize=fsize)
        plt.grid()
        plt.title('Target range = [0,%0.1f]'%len(data),fontsize=fsize)
        plt.xlim(1.2*np.min(t0),1.2*np.max(t0))
        plt.ylim(1.2*np.min(y1),1.2*np.max(y1))
        


plt.show()



    



