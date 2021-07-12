### Oblig 5 

import numpy as np
import matplotlib.pyplot as plt

# Konstanter
g = 9.81    # tyngdeakselerasjon i m/s^2
m = 1.      # masse til ballen i kg
v0x = 3.    # initialhastighet i m/s
h = 1.
R = 0.15    # radius til ballen i m
k = 10000   # fjærkonstanten i N/m
mu = 0.3    # friksjonskoeffisienten

I = 2/3*m*R**2 # treghetsmomentet
T = 1.   # tid vi ser på i sekunder
dt = 0.001
n = int(T/dt)

t = np.zeros((n,1),float)
r = np.zeros((n,2),float)
v = np.zeros((n,2),float)
omega = np.zeros((n,1),float)

ux= np.array([1,0])      # enhetvektor i x-retning
uy= np.array([0,1])      # enhetvektor i y-retning

# Initialbetingelsene
r[0, 1] = h  # startposisjonen y-retning
v[0, 0] = v0x # initialhastighet

# Bevegelsen
for i in range (n-1):
    if r[i, 1] < R:
        F_N = k*(R-r[i,1])**(1.5) # normalkraften
    else:
        F_N = 0
    f = -mu*F_N*np.sign(v[i,0])   #friksjon
    a = f/m*ux + (F_N/m-g)*uy
    az= R*f/I                    #vinkelakselerasjon
    # Euler-Cromer-metoden
    v[i+1, :] = v[i, :] + a*dt         #hastigheten
    r[i+1, :] = r[i, :] + v[i+1, :]*dt #posisjonen
    omega[i+1] = omega[i] +az*dt       #vinkelhastigheten
    t[i+1] = t[i] + dt                 #tiden

#plots
f1 = plt.figure(1)
f1.suptitle('Bevegelsen til ballen')
plt.subplot(2,1,1)
plt.plot(t, r[:,0])
plt.xlabel('t [sec]')
plt.ylabel('x [m]')
plt.subplot(2,1,2)
plt.plot(t, r[:,1])
plt.xlabel('t [sec]')
plt.ylabel('y [m]')
# plt.savefig('y.pdf')

f2 = plt.figure(2)
f2.suptitle('Hastigheten til ballen')
plt.subplot(3,1,1)
plt.plot(t, v[:,0])
plt.xlabel('t [sec]')
plt.ylabel('$v_x$ [m/s]')
plt.subplot(3,1,2)
plt.plot(t, v[:,1])
plt.xlabel('t [sec]')
plt.ylabel('$y_y$ [m/s]')
plt.subplot(3,1,3)
plt.plot(t, omega)
plt.xlabel('t [sec]')
plt.ylabel('$\omega_z$ [rad/s]')
plt.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
# plt.savefig('v.pdf')
plt.show()


"""
(plots)

"""
