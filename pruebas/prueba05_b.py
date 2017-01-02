#!/usr/bin/python
#entrenamiento mediante backpropagation por numero de aproximaciones
#entrenamiento standard y paralelo
#mezcla prueba0 y prueba 03

from __future__ import division
import numpy as np
import pylab as pl

import pybrain.datasets

import pybrain.tools.shortcuts as pybrain_tools
import pybrain.supervised.trainers.rprop as pybrain_rprop

import multiprocessing
import timeit

from pybrain.structure import LinearLayer,SoftmaxLayer,TanhLayer,SigmoidLayer

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
    #print 'numero de datos %i' % L
    #valor en el instante t0
    #np.sin(Wn*t0)   Wn=2*pi*f*t0
    x0 = A*np.sin(2*np.pi*f*t0)     #senal de entrada a la red
    dataset = pybrain.datasets.SupervisedDataSet(1, 1)#1 entradas,1 salida
    for i in range(L):
        dataset.addSample(t0[i], x0[i])#muestra n
    return dataset

def chart_original_output(dataset,net):#graficar dataset
    entrada = dataset['input']
    salida = dataset['target']
    L = len(entrada)
    red = np.zeros(L)
    for c in range(L):
        red[c] = net.activate([entrada[c]])
    fig = pl.figure()
    fsize=8
    pl.plot(entrada,salida,'r.-',label='input')
    pl.plot(entrada,red,'bx-',label='predicted')
    pl.xlabel('Time',fontsize=fsize)
    pl.ylabel('Amplitude',fontsize=fsize)
    pl.xlim(np.min(entrada),1.2*np.max(entrada))
    pl.ylim(1.2*np.min(salida),1.2*np.max(salida))
    pl.grid()
    pl.legend(loc='lower right',ncol=2,fontsize=fsize)
    pl.title('Target range = [0,1]',fontsize=fsize)


def standard_train(epochs = 50):#50 aproximaciones por default
    np.random.seed(0)
    net = pybrain_tools.buildNetwork(1, 20, 1, hiddenclass=TanhLayer, outclass=LinearLayer, bias = False)
    #ajusta la red a i entradas,j capas ocultas,k salidas, fa tanh capa oculta, fa softmax capa salida 
    #fa = funcion de activacion, TanhLayer o SigmoidLayer
    net.randomize()#ajusta aleatoriamente los parametros de la red
    print 'entrenando red standard '
    #print 'entrenando red standard, {}'.format(net)
    data = init_sin_dataset()
    trainer = pybrain_rprop.BackpropTrainer(net, dataset=data)#red, datos de entrenamiento
    trainer.trainEpochs(epochs)#numero de iteraciones
    chart_original_output(data,net)#graficar dataset


def multithreaded_train(epochs=50,threads=2):#50 aproximaciones,8 hilos por default
    nets = []#vector de redes
    trainers = []#vector de entrenaientos
    processes = []#procesos
    data = init_sin_dataset()#valores de entrenamiento
    
    for n in range(threads):
        nets.append(pybrain_tools.buildNetwork(1, 20, 1))#ajusta la red n a 1 entradas,2 capas ocultas,1 salida
        nets[n].randomize()#ajusta aleatoriamente los parametros de la red n
        print 'entrenando red multihilo {} '.format(n)
        #print 'entrenando red multihilo {} {}'.format(n,nets[n])
        trainers.append(pybrain_rprop.BackpropTrainer(nets[n], dataset=data))#red n, datos de entrenamiento
        processes.append(multiprocessing.Process(target=trainers[n].trainEpochs(epochs)))#numero de iteraciones para red n
        processes[n].start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Draw all charts
    for n in range(threads):
        chart_original_output(data,nets[n])#graficar dataset
    


if __name__ == '__main__':
    epochs = 1500#numero de iteraciones de la red
    threads = 4#numero de hilos/procesos
    iterations_standard = 4#numero de veces que se repite la funcion en timeit
    iterations_multi = 1#numero de veces que se repite la funcion en timeit
    #''' standard 4 iteracion en timeit
    t1 = timeit.timeit("standard_train({})".format(epochs),
                       setup="from __main__ import standard_train",
                       number=iterations_standard)
    #''' multihilo 1 iteracion en timeit
    tn = timeit.timeit("multithreaded_train({},{})".format(epochs,threads),
                       setup="from __main__ import multithreaded_train",
                       number=iterations_multi)
    print "Execution time for single threaded training: {} seconds.".format(t1)
    print "Execution time for {} multi threaded training: {} seconds.".format(threads,tn)
    pl.show()





