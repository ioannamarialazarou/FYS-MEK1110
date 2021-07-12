### oblig 2

import numpy as np
import matplotlib.pyplot as plt

#Constants
g = 9.81      #m/s^2
m = 0.1       #kg
L0 = 1.0      #m
k = 200.0     #N/m
θ = np.radians(30.0) #radians

dt = 0.01     #time step - question i)
# dt = 0.1    #time step - question i) (checking different dt)
# dt = 0.001  #time step - question j)
T = 10.0      #time

n = int(round(T/dt))
t = np.zeros((n,1),float)
r = np.zeros((n,2),float)
v = np.zeros((n,2),float)
a = np.zeros((n,2),float)

v[0] = np.array([0,0])
r[0] = np.array([L0*np.sin(θ),-L0*np.cos(θ)])

for i in range(n-1):
    lr = np.sqrt(r[i,0]**2 + r[i,1]**2)
    a[i,:] = np.array([(-k*(1-L0/lr)*r[i,0])/m,-g-(k*(1-L0/lr)*r[i,1])/m])
    v[i+1,:] = v[i,:] + a[i,:]*dt
    r[i+1,:] = r[i,:] + v[i+1,:]*dt
    t[i+1] = t[i] + dt

#plots
#main
plt.plot(r[:,0],r[:,1])
plt.title('Posisjon av ballen')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

#alternative plot
plt.plot(t,r[:])
plt.axis([0,5,-1.5,1.5])
plt.xlabel('tid [s]')
plt.ylabel('posisjon [m]')
plt.legend(['x','y'])
plt.title('Posisjon av ballen')
plt.show()

"""
(plots)
"""
