#!/usr/bin/python
#prueba para graficar prediccion de senal senoidal
#prediccion muy erronea

from __future__ import division
import numpy as np
import pylab as pl

from pybrain.structure import TanhLayer, LinearLayer, SigmoidLayer #SoftmaxLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet 
from pybrain.supervised.trainers import BackpropTrainer 
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection 

np.random.seed(0)
pl.close('all')

#create NN structure:
net = FeedForwardNetwork()
inLayer = LinearLayer(1)
hiddenLayer = TanhLayer(100)
#hiddenLayer = SigmoidLayer(50)
outLayer = LinearLayer(1)

#add classes of layers to network, specify IO:
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

#specify how neurons are to be connected:
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

#add connections to network:
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

#perform internal initialisation:
net.sortModules()

#construct target signal:
T = 1
Ts = T/10
f = 1/T
fs = 1/Ts

#NN input signal:
t0 = np.arange(0,10*T,Ts)
L = len(t0)

#NN target signal:
x0 = 10*np.cos(2*np.pi*f*t0) + 10 + np.random.randn(L)

#normalise input signal:
t = t0/np.max(t0)

#normalise target signal to fit in range [0,1] (min) or [-1,1] (mean):
dcx = np.min(x0) #np.min(x0) #np.mean(x0) 
x = x0-dcx
sclf = np.max(np.abs(x))
x /= sclf

#add samples and train NN:
ds = SupervisedDataSet(1, 1)
for c in range(L):
 ds.addSample(t[c], x[c])

trainer = BackpropTrainer(net, ds, learningrate=0.001, momentum=0.01)

for c in range(100):
 e1 = trainer.train()
 print 'Epoch %d Error: %f'%(c,e1)

y=np.zeros(L)
for c in range(L):
 #y[c] = net.activate([x[c]])
 y[c] = net.activate([t[c]])

yout = y*sclf  
yout = yout + dcx

fig1 = pl.figure(1)
pl.ion()

fsize=8
pl.subplot(211)
pl.plot(t0,x0,'r.-',label='input')
pl.plot(t0,yout,'bx-',label='predicted')
pl.xlabel('Time',fontsize=fsize)
pl.ylabel('Amplitude',fontsize=fsize)
pl.grid()
pl.legend(loc='lower right',ncol=2,fontsize=fsize)
pl.title('Target range = [0,1]',fontsize=fsize)

fig1name = './prueba01.png'
print 'Saving Fig. 1 to:', fig1name
fig1.savefig(fig1name, bbox_inches='tight')

print 'end'
