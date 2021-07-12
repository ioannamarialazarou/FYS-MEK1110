import numpy as np
import matplotlib.pyplot as plt

### e)
# konstanter
m = 0.1
k = 20.
b = 0.2
v0 = 1.

T = 2.
dt = 0.001
n = int(T/dt)

t = np.zeros((n,1),float)
xA = np.zeros((n,1),float)
xB = np.zeros((n,1),float)
vA = np.zeros((n,1),float)
vB = np.zeros((n,1),float)

# Initialbetingelsene
xA[0] = -b
xB[0] = 0
vA[0] = v0
vB[0] = 0

for i in range (n-1):
    F_BA = k*(xB[i] - xA[i] - b)
    aA = F_BA/m
    vA[i+1] = vA[i] +aA*dt
    xA[i+1] = xA[i] +vA[i+1]*dt

    F_AB = - k*(xB[i] - xA[i] - b)
    aB = F_AB/m
    vB[i+1] = vB[i] +aB*dt
    xB[i+1] = xB[i] +vB[i+1]*dt

    t[i+1] = t[i] +dt

plt.plot(t, xA, 'b', t, xB, 'm')
plt.title('Posisjonene til atom A og B som funksjon av tid ')
plt.legend(["A", "B"])
plt.xlabel('t [sec]')
plt.ylabel('x [m]')
# plt.savefig('e.pdf')
plt.show()

# plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.plot(t, xB-xA)
plt.title('$x_B - xA$ som funksjon av tid')
plt.xlabel('t [sec]')
plt.ylabel('x [m]')
# plt.savefig('e_.pdf')
plt.show()

### i)
m = 0.1
k = 20.
b = 0.2
v0 = 1.

T = 2.
dt = 0.001
n = int(T/dt)

t = np.zeros((n,1),float)
rA = np.zeros((n,2),float)
rB = np.zeros((n,2),float)
vA = np.zeros((n,2),float)
vB = np.zeros((n,2),float)

rA[0,:] = [0,b]
rB[0,:] = [0,0]
vA[0,:] = [v0,0]
vB[0,:] = [0,0]

for i in range(n-1):
    Dr = rB[i,:] - rA[i,:]
    rn = np.sqrt(np.sum(Dr*Dr))
    F_BA = k*(rn -b)*Dr/rn
    aA = F_BA/m
    vA[i+1,:] = vA[i,:] + aA*dt
    rA[i+1,:] = rA[i,:] + vA[i+1,:]*dt

    F_AB = -F_BA
    aB = F_AB/m
    vB[i+1,:] = vB[i,:] + aB*dt
    rB[i+1,:] = rB[i,:] + vB[i+1,:]*dt
    t[i+1] = t[i] + dt

plt.plot(rA[:,0], rA[:,1],'b', rB[:,0], rB[:,1],'m')
plt.legend(["A", "B"])
plt.title('Posisjon (x(t),y(t))')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
# plt.savefig('i.pdf')
plt.show()


"""
(plots)
"""
