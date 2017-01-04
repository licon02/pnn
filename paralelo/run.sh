#!/bin/sh
clear
scp -r p1.py 192.168.0.102:/home/pi
scp -r p1.py 192.168.0.103:/home/pi
mpiexec -f machinefile -n 3 python p1.py 
