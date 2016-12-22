#!/usr/bin/python
#prueba para probar tiempo de ejecucion de hilos

import pybrain.tools.shortcuts as pybrain_tools
import pybrain.datasets
import pybrain.supervised.trainers.rprop as pybrain_rprop
import multiprocessing
import timeit


def init_XOR_dataset():
    dataset = pybrain.datasets.SupervisedDataSet(2, 1)#2 entradas,1 salida
    dataset.addSample([0, 0], [0])#muestra 1
    dataset.addSample([0, 1], [1])#muestra 2
    dataset.addSample([1, 0], [1])#muestra 3
    dataset.addSample([1, 1], [0])#muestra 4
    return dataset


def standard_train():
    net = pybrain_tools.buildNetwork(2, 2, 1)#ajusta la red a 2 entradas,2 capas ocultas,1 salida
    net.randomize()#ajusta aleatoriamente los parametros de la red
    trainer = pybrain_rprop.RPropMinusTrainer(net, dataset=init_XOR_dataset())#red, datos de entrenamiento
    trainer.trainEpochs(50)#numero de iteraciones


def multithreaded_train(threads=8):
    nets = []#vector de redes
    trainers = []#vector de entrenaientos
    processes = []#procesos
    data = init_XOR_dataset()#valores de entrenamiento

    for n in range(threads):
        nets.append(pybrain_tools.buildNetwork(2, 2, 1))#ajusta la red n a 2 entradas,2 capas ocultas,1 salida
        nets[n].randomize()#ajusta aleatoriamente los parametros de la red n
        trainers.append(pybrain_rprop.RPropMinusTrainer(nets[n], dataset=data))#red n, datos de entrenamiento
        processes.append(multiprocessing.Process(target=trainers[n].trainEpochs(50)))#numero de iteraciones para red n
        processes[n].start()

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    threads = 4
    iterations = 16

    t1 = timeit.timeit("standard_train()",
                       setup="from __main__ import standard_train",
                       number=iterations)
    tn = timeit.timeit("multithreaded_train({})".format(threads),
                       setup="from __main__ import multithreaded_train",
                       number=iterations)

    print "Execution time for single threaded training: {} seconds.".format(t1)
    print "Execution time for {} multi threaded training: {} seconds.".format(threads,tn)
