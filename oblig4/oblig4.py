### oblig 4

#h)
import numpy as np
import matplotlib.pyplot as plt

#Startverdier
U0 = 150
m = 23
x0 = 2
alpha = 39.48

time = 10
dt = 0.01
n = int(time/dt)

x = np.zeros(n)
v = np.zeros(n)
a = np.zeros(n)
t = np.linspace(0, time, n)

# #i)
# v[0] = 8.
# x[0] = -5.

# #j)
# v[0] = 10.
# x[0] = -5.

U = lambda x: U0 if np.abs(x) >= x0 else U0*(np.abs(x)/x0)

F = lambda x : U0/x0 if (x>-x0 and x<0) else(-U0/x0 if (x<x0 and x>0) else 0)

F_foton = lambda x, v: -alpha*v if np.abs(x) < x0 else 0

for i in range(n-1):
    a[i+1] = (F(x[i]) + F_foton(x[i], v[i]))/m
    v[i+1] = v[i] + a[i+1]*dt
    x[i+1] = x[i] + v[i+1]*dt
    t[i+1] = t[i] + dt

plt.plot(t,x)
plt.ylabel('Posisjon-x [m]')
plt.xlabel('Tid-t [s]')
plt.title('Posisjon som funksjon av tid')
# plt.savefig('j.pdf')
plt.show()

"""
(plot)
"""
