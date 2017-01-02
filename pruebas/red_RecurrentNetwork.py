#!/usr/bin/python
#descripcion y ensamble de la red RecurrentNetwork paso a paso

from pybrain.structure import RecurrentNetwork
#estructuras FeedForwardNetwork and RecurrentNetwork

from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
#modulos BiasUnit,GaussianLayer, LinearLayer, LSTMLayer, MDLSTMLayer, SigmoidLayer, SoftmaxLayer, StateDependentLayer, TanhLayer

from pybrain.structure import FullConnection
#conectores

n = RecurrentNetwork(name = 'red de prueba1')#n = red

n.addInputModule(LinearLayer(2, name = 'entradas'))#agrega el modulo de 'entrada' a la red con 2 entradas
n.addModule(SigmoidLayer(3, name = 'ocultas'))#agrega el modulo de 'oculta' a la red con 3 capas ocultas
#n.addModule(TanhLayer(3, name = 'ocultas'))#agrega el modulo de 'oculta' a la red con 3 capas ocultas
n.addOutputModule(LinearLayer(1, name = 'salidas'))#agrega el modulo de 'salida' a la red con 1 salida

#flujo de informacion entre las capas
n.addConnection(FullConnection(n['entradas'],n['ocultas'], name = 'con1'))#agrega y conecta el modulo de conexion de capa 'entrada' a la red con las capas ocultas
n.addConnection(FullConnection(n['ocultas'],n['salidas'], name = 'con2'))#agrega y conecta el modulo de conexion de capa 'salida' a la red con las capas ocultas

n.addRecurrentConnection(FullConnection(n['ocultas'], n['ocultas'], name='con3'))#metodo adicional

n.sortModules()#ordena la red

#visualizar valores (pesos)
print n#visualizar estructura de la red
print ''
print n.params#todos los pesos
print ''
print n.activate([1, 2])#prenguntar a la red la salida para cierta entrada
#n.reset()#limpiar la red


