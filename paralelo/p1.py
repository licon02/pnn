#!/usr/bin/python
#entrenamiento mediante rprop por numero de aproximaciones
#entrenamiento standard y paralelo
#mezcla prueba0 y prueba 03

from __future__ import division
import numpy as np
import pylab as pl
from pybrain.structure.modules import LSTMLayer
import pybrain.tools.shortcuts as pybrain_tools
import pybrain.datasets
import pybrain.supervised.trainers.rprop as pybrain_rprop
import multiprocessing
import timeit

from mpi4py import MPI
import sys




pl.close('all')

def init_sin_dataset():#regresa un vector de una entrada, una salida
    #construct target signal:
    T = 1               #periodo de la senal
    Nyq = 40            #minimo 2 por el teorema de Nyquist
    Ts = T/Nyq           #periodo de muestreo
    f = 1/T             #frecuencia de la senal
    fs = 1/Ts           #frecuencia de periodo
    A = 1              #amplitud
    Tiempo = 1         #tiempo total de muestreo
    #NN input signal:
    t0 = np.arange(0,Tiempo+Ts,Ts)   #genera un vector de n hasta N, con incrementos de i (n,N,i)
    L = len(t0)                 #valor aleatorio 
    print 'numero de datos %i' % L
    #valor en el instante t0
    #np.sin(Wn*t0)   Wn=2*pi*f*t0
    x0 = A*np.cos(2*np.pi*f*t0)     #senal de entrada a la red
    dataset = pybrain.datasets.SupervisedDataSet(1, 1)#1 entradas,1 salida
    for i in range(L):
        dataset.addSample(t0[i], x0[i])#muestra n
    return dataset

def chart_original_output(entrada,salida,out):#graficar dataset
    #print net.params
    fig = pl.figure()
    fsize=8
    pl.plot(entrada,salida,'r.-',label='input')
    pl.plot(entrada,out,'bx-',label='predicted')
    pl.xlabel('Time',fontsize=fsize)
    pl.ylabel('Amplitude',fontsize=fsize)
    pl.xlim(np.min(entrada),1.2*np.max(entrada))
    pl.ylim(1.2*np.min(salida),1.2*np.max(salida))
    pl.grid()
    pl.legend(loc='lower right',ncol=2,fontsize=fsize)
    pl.title('Target range = [0,1]',fontsize=fsize)
      

epochs = 2500#numero de iteraciones de la red

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
sys.stdout.write("Hello, World! I am process %d of %d on %s.\n"% (rank, size, name))
comm = MPI.COMM_WORLD
np.random.seed(0)
net = pybrain_tools.buildNetwork(1, 40, 1)#ajusta la red a i entradas,j capas ocultas,k salidas
net.randomize()#ajusta aleatoriamente los parametros de la red
print 'entrenando red standard'
data=init_sin_dataset()
trainer = pybrain_rprop.RPropMinusTrainer(net, dataset=data)#red, datos de entrenamiento
trainer.trainEpochs(epochs)#numero de iteraciones

entrada = data['input']
salida = data['target']
L = len(entrada)
out = np.zeros(L)
aux3 = np.zeros(L)
aux4 = np.zeros(L)
for c in range(L):
    out[c] = net.activate([entrada[c]])
#chart_original_output(data,net)#graficar dataset
# Draw all charts
print out
if name == 'Pi03':
    aux3 = out
    comm.Send(aux3, dest = 0)
if name == 'Pi04':
    aux4 = out
    comm.Send(aux4, dest = 0)
if name == 'Pi01':
    chart_original_output(entrada,salida,out)#graficar dataset
    if size >=2:
        comm.Recv(aux3,source = 1)
        chart_original_output(entrada,salida,aux3)#graficar dataset
    if size >= 3:
        comm.Recv(aux4,source = 2)
        chart_original_output(entrada,salida,aux4)#graficar dataset
    
pl.show()




