### Oblig 3

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#e)
l = 0.5  #m
k = 500  #N/m
h = 0.3  #m

x = np.linspace(-0.75, 0.75)
r = np.sqrt(x**2 + h**2)
F = lambda x: -k*x*(1-(l/r))

plt.plot(x,F(x))
plt.xlabel('x [m]')
plt.ylabel('$F_x$ [N]')
# plt.savefig('e.pdf')
plt.show()

#f)
l = 0.5  #m
k = 500  #N/m
h = 0.3  #m
m = 5.   #kg

time = 10     # s
dt   = 1./100 # time steps
n    = int(time/dt)

t = np.zeros(n,float); x = np.zeros(n,float);
v = np.zeros(n,float); a = np.zeros(n,float);

x[0] = 0.6
# x[0] = 0.65 #for task f)
v[0] = 0
a[0] = 0

for i in range(n-1):
    r = sqrt(x[i]**2 + h**2)
    a[i+1] = -(k/m)*x[i]*(1 - l/r)
    v[i+1] = v[i] + dt*(a[i+1])
    x[i+1] = x[i] + dt*(v[i+1])
    t[i+1] = t[i] + dt

plt.subplot(2,1,1)
plt.plot(t,x)
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.title('Posisjon - tid$\in$(0,10)')

plt.subplot(2,1,2)
plt.plot(t,v)
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.title('Hastighet - tid$\in$(0,10)')

plt.tight_layout()
# plt.savefig('f.pdf')
plt.show()

#k)
l = 0.5   #m
k = 500   #N/m
h = 0.3   #m
m = 5.    #kg
g = 9.81  #m/s^2
md = 0.05

time = 10     # s
dt   = 1./100 # time steps
n    = int(time/dt)

t = np.zeros(n,float); x = np.zeros(n,float);
v = np.zeros(n,float); a = np.zeros(n,float)

x[0] = 0.75
v[0] = 0
a[0] = 0

for i in range(n-1):
    Fd = -md*m*g*np.sign(v[i])  #friksjon

    r = sqrt(x[i]**2 + h**2)
    a[i+1] = -(k/m)*x[i]*(1 - l/r) + Fd/m
    v[i+1] = v[i] + dt*(a[i+1])
    x[i+1] = x[i] + dt*(v[i+1])
    t[i+1] = t[i] + dt

plt.subplot(2,1,1)
plt.plot(t,x)
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.title('Posisjon - tid')

plt.subplot(2,1,2)
plt.plot(t,v)
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.title('Hastighet - tid')

plt.tight_layout()
# plt.savefig('k.pdf')
plt.show()


#l)
l = 0.5   #m
k = 500   #N/m
h = 0.3   #m
m = 5.    #kg
g = 9.81  #m/s^2
md = 0.05

time = 10     # s
dt   = 1./100 # time steps
n    = int(time/dt)

t = np.zeros(n,float); x = np.zeros(n,float);
v = np.zeros(n,float); a = np.zeros(n,float);
K = np.zeros(n,float)

x[0] = 0.75
v[0] = 0
a[0] = 0
K[0] = 0

for i in range(n-1):
    Fd = -md*m*g*np.sign(v[i])  #friksjon

    r = sqrt(x[i]**2 + h**2)
    a[i+1] = -(k/m)*x[i]*(1 - l/r) + Fd/m
    v[i+1] = v[i] + dt*(a[i+1])
    x[i+1] = x[i] + dt*(v[i+1])
    t[i+1] = t[i] + dt


    K[i] = 0.5*m*v[i]**2

plt.plot(x,K)
plt.ylabel('K [J]')
plt.xlabel('x [m]')
plt. title('Kinetisk energi - posisjon')
# plt.savefig('l.pdf')
plt.show()


#kinetic energy during 10 sec.
plt.plot(t,K)
plt.show()

"""
(plots)
"""
