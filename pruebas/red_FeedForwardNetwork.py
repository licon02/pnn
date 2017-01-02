#!/usr/bin/python
#descripcion y ensamble de la red FeedForwardNetwork paso a paso

from pybrain.structure import FeedForwardNetwork
#estructuras FeedForwardNetwork and RecurrentNetwork

from pybrain.structure import LinearLayer, SigmoidLayer
#modulos BiasUnit,GaussianLayer, LinearLayer, LSTMLayer, MDLSTMLayer, SigmoidLayer, SoftmaxLayer, StateDependentLayer, TanhLayer

from pybrain.structure import FullConnection
#conectores

n = FeedForwardNetwork(name = 'red de prueba1')#n = red
inLayer = LinearLayer(2, name = 'entradas')#numero de entradas
hiddenLayer = SigmoidLayer(3, name = 'ocultas')#numero de capas ocultas
outLayer = LinearLayer(1, name = 'salidas')#numero de salidas

n.addInputModule(inLayer)#agrega el modulo de 'entrada' a la red
n.addModule(hiddenLayer)#agrega el modulo de 'oculta' a la red
n.addOutputModule(outLayer)#agrega el modulo de 'salida' a la red

#flujo de informacion entre las capas
in_to_hidden = FullConnection(inLayer, hiddenLayer)#conecta explicitamente la entrada con las capas ocultas
hidden_to_out = FullConnection(hiddenLayer, outLayer)#conecta explicitamente la capa oculta con la capa de salida

n.addConnection(in_to_hidden)#agrega el modulo de conexion de capa 'entrada' a la red
n.addConnection(hidden_to_out)#agrega el modulo de conexion de capa 'salida' a la red

n.sortModules()#ordena la red

#visualizar valores (pesos)
print n#visualizar estructura de la red
print ''
print in_to_hidden.params#capa entrada a capa oculta
print hidden_to_out.params#capa oculta a capa salida
print ''
print n.params#todos los pesos
print ''
print n.activate([1, 2])#prenguntar a la red la salida para cierta entrada
#n.reset()#limpiar la red


