### Oblig 1

import numpy as np
import matplotlib.pyplot as plt

### e)

time = 8    # s
dt = 1./100 # time steps
n    = int(time/dt)

t = np.zeros(n,float); x = np.zeros(n,float);
v = np.zeros(n,float); a = np.zeros(n,float)

x[0] = 0.0  # m
t[0] = 0.0  # s
v[0] = 0.0  # m/s
a[0] = 5.0  # m/s^2

#constants
F  = 400.0  # N
m  = 80.0   # kg
p  = 1.293  # kg/m^3
Cd = 1.2
A0 = 0.45   # m^2
w  = 0      # m/s

time_track = 0
# Euler-Cromer method
for i in range(n-1):
    a[i+1] = (F - 0.5*p*Cd*A0*(v[i]-w)**2)/m
    v[i+1] = v[i] + a[i]*dt
    x[i+1] = x[i] + v[i+1]*dt
    t[i+1] = t[i] + dt

    #question (f)
    if x[i+1] >= 100 and time_track == 0:
        time_track  = t[i+1]
        print (f"Running time of the sprinter: {time_track} sec.")

plt.subplot(3,1,1)
plt.plot(t, x)
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.title("Position")

plt.subplot(3,1,2)
plt.plot(t, v)
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.title("Velocity")

plt.subplot(3,1,3)
plt.plot(t, a)
plt.xlabel('t [s]')
plt.ylabel('a [m/s$^2$]')
plt.title("Acceleration")

plt.tight_layout()
plt.savefig("taskE.pdf")
plt.show()

### h)

from math import sqrt
U_T= sqrt((2*F)/(p*Cd*A0))
print(U_T)


### j)

time = 10   # s
dt = 1./1000
n = int(time/dt)

t = np.zeros(n,float); x = np.zeros(n,float);
v = np.zeros(n,float); a = np.zeros(n,float)

x[0] = 0.0  # m
t[0] = 0.0  # s
v[0] = 0.0  # m/s

#constants
F  = 400.0  # N
m  = 80.0   # kg
p  = 1.293  # kg/m^3
Cd = 1.2
A0 = 0.45   # m^2
w  = 0      # m/s
fc = 488    # N
tc = 0.67   # s
fv = 25.8   # Ns/m

time_track = 0
# Euler-Cromer method
for i in range(n-1):

    Fc = fc*np.exp(-(t[i]/tc)**2)
    Fv = - fv*v[i]
    D = 0.5*p*Cd*A0*(1 - 0.25*np.exp(-(t[i]/tc)**2))*(v[i]-w)**2

    a[i] = (F + Fc + Fv - D)/m
    v[i+1] = v[i] + a[i]*dt
    x[i+1] = x[i] + v[i+1]*dt
    t[i+1] = t[i] + dt

    if x[i+1] >=100 and time_track == 0:
        time_track  = t[i+1]
        print (f"Running time of the sprinter: {time_track} sec.")

plt.subplot(3,1,1)
plt.plot(t, x)
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.title("Position")

plt.subplot(3,1,2)
plt.plot(t, v)
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.title("Velocity")


plt.subplot(3,1,3)
plt.plot(t, a)
plt.xlabel('t [s]')
plt.ylabel('a [m/s$^2$]')
plt.title("Acceleration")

plt.tight_layout()
plt.savefig("taskJ.pdf")
plt.show()


### k)

time = 9.3; dt = 1./1000; n = int(time/dt)

a = np.zeros(n); x = np.zeros(n); v = np.zeros(n); t = np.zeros(n)
v[0] = 0; x[0] = 0; t[0] = 0

Fv = lambda v: v*fv
Fc = lambda t: fc*np.exp(-(t/tc)**2)
D = lambda t,v: A0*(1 - 0.25*np.exp(-(t/tc)**2))*0.5*p*Cd*(v-w)**2

for i in range(int(n-1)):
    a[i] = (F + Fc(t[i]) - Fv(v[i]) - D(t[i], v[i]))/m
    v[i+1] = v[i] + a[i]*dt
    x[i+1] = x[i] + v[i+1]*dt
    t[i+1] = t[i] + dt

plt.plot(t,Fc(t),t,Fv(v),t,D(t,v),[0,9.3],[400,400])
plt.legend(['Fc', 'Fv','D','F'])
plt.axis([0,9.3,-100,600])
plt.show()


"""
Running time of the sprinter: 6.7899999999999 sec.
33.849234466965406
Running time of the sprinter: 9.320000000000274 sec.
"""
