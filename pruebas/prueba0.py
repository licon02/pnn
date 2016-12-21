import pybrain.tools.shortcuts as pybrain_tools
import pybrain.datasets
import pybrain.supervised.trainers.rprop as pybrain_rprop
import multiprocessing
import timeit


def init_XOR_dataset():
    dataset = pybrain.datasets.SupervisedDataSet(2, 1)
    dataset.addSample([0, 0], [0])
    dataset.addSample([0, 1], [1])
    dataset.addSample([1, 0], [1])
    dataset.addSample([1, 1], [0])
    return dataset


def standard_train():
    net = pybrain_tools.buildNetwork(2, 2, 1)
    net.randomize()
    trainer = pybrain_rprop.RPropMinusTrainer(net, dataset=init_XOR_dataset())
    trainer.trainEpochs(50)


def multithreaded_train(threads=8):
    nets = []
    trainers = []
    processes = []
    data = init_XOR_dataset()

    for n in range(threads):
        nets.append(pybrain_tools.buildNetwork(2, 2, 1))
        nets[n].randomize()
        trainers.append(pybrain_rprop.RPropMinusTrainer(nets[n], dataset=data))
        processes.append(multiprocessing.Process(target=trainers[n].trainEpochs(50)))
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
    print "Execution time for multi threaded training: {} seconds.".format(tn)
