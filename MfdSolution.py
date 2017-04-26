import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

## Compute coefficients for initial condition
pin_locs = np.array([0.25, 0.5, 0.75, 1.])
pin_vals = np.array([0.9, 1., 0.9, 0.])
MAT = np.zeros((4,4))
MAT[:,0] = pin_locs; MAT[:,1] = pin_locs**2
MAT[:,2] = pin_locs**3; MAT[:,3] = pin_locs**4
coeffs_init = np.linalg.solve(MAT,pin_vals)

## Compute coefficients for steady state
pin_locs = np.array([0.25, 0.5, 0.75, 1.])
pin_vals = 1.5*np.array([0.5, 0.55, 0.7, 0.])
MAT = np.zeros((4,4))
MAT[:,0] = pin_locs; MAT[:,1] = pin_locs**2
MAT[:,2] = pin_locs**3; MAT[:,3] = pin_locs**4
coeffs_steady = np.linalg.solve(MAT,pin_vals)


def init_func(x):
   return coeffs_init[0]*x + coeffs_init[1]*x**2 + \
          coeffs_init[2]*x**3 + coeffs_init[3]*x**4

def steady_func(x):
   return coeffs_steady[0]*x + coeffs_steady[1]*x**2 + \
          coeffs_steady[2]*x**3 + coeffs_steady[3]*x**4

xres = 100; tres = 200
x = np.linspace(0.,1.,num=xres)
t = np.linspace(0.,5.,num=tres)

X, T = np.meshgrid(x,t,indexing='ij')

sol_arr = init_func(X)*np.exp(-T) + steady_func(X)*(1.-np.exp(-T))

## Animate ##
fig = plt.figure('Manufactured Solution')
ax = plt.axes(xlim=(0,1), ylim=(0,1.2))
line, = ax.plot([],[],lw=2,label='Mfd Sol')

def init():
    line.set_data([],[])
    return line,

def animate(i):
    line.set_data(x,sol_arr[:,i])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, 
       frames=tres, interval=20)

plt.show()
