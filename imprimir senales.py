#!/usr/bin/python
#prueba 03
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)   #generar semilla para numeros randomize
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
L = len(t0)                 #valor aleatorio 

#NN target signal:
#valor en el instante t0
#np.sin(Wn*t0)   Wn=2*pi*f*t0
x0 = A*np.sin(2*np.pi*f*t0)     #senal de entrada a la red

#senal con ruido
yout = A*np.sin(2*np.pi*f*t0) + np.random.randn(L)

fig1 = plt.figure(1)

fsize=8
plt.subplot(211)
plt.plot(t0,x0,'r.-',label='input')
plt.xlabel('Time',fontsize=fsize)
plt.ylabel('Amplitude',fontsize=fsize)
plt.grid()
plt.legend(loc='lower right',ncol=2,fontsize=fsize)
plt.title('Target range = [0,%0.1f]'%Tiempo,fontsize=fsize)
plt.xlim(0,Tiempo)
plt.ylim(1.5*np.min(x0),1.5*np.max(x0))

plt.subplot(212)
plt.plot(t0,yout,'bx-',label='predicted')
plt.xlabel('Time',fontsize=fsize)
plt.ylabel('Amplitude',fontsize=fsize)
plt.grid()
plt.legend(loc='lower right',ncol=2,fontsize=fsize)
plt.title('Target range = [0,%0.1f]'%Tiempo,fontsize=fsize)
plt.xlim(0,Tiempo)
plt.ylim(1.5*np.min(yout),1.5*np.max(yout))


plt.show()

fig1name = './sin_min.png'
print 'Saving Fig. 1 to:', fig1name
fig1.savefig(fig1name, bbox_inches='tight')


