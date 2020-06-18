#############################################################################################
#############################################################################################
# Testbed for 1-D coupling model problem code
#############################################################################################
#############################################################################################

import numpy as np
import CouplingHelpers as CH
import CouplingHelpersExplicit as CHE
import matplotlib.pyplot as plt
from matplotlib import animation

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import ConvecDiffTerms as CDT

# Whether to save frames from animation
save = False
animName = 'Opt1A_noise_CN'

# Boundaries for overlapping regions
lu_lim = 0.8; rl_lim = 0.2

# Width of "buffer zones" on either side.
# Only used if  knotting = 'zero',
# meaning the solution is set to zero at 
# the internal boundaries
lbuffer = 0.25; rbuffer = 0.25;

# Number of cells in each region
ncls_l = 50; ncls_r = 50

# Number of time step steps, and run-time
numsteps = 400; TotTime = 16.0

# Vertical scale for plot
ymax_for_plot = 1.3

# Time-step at which to start time-averaged steady-state computation
start_t_avg = int(0.2*numsteps)

# 'composite' means compute one potential phi on whole domain, 
# 'separate' means compute a separate potential on each sub-domain
runmode = 'composite'
# 'consistent' means each region gets its internal boundary condition
# from the other region's value at previous time step
# 'zero' means the internal boundary conditions are just set to 0
knotting = 'consistent'
if knotting == 'zero':
    lbuff = lbuffer; rbuff = rbuffer
else:
    lbuff = 0.; rbuff = 0.

# 'semi-implicit' means linear backwards Euler,
# 'explicit' means forward Euler (will choose time-step just below
# CFL limit, overriding number of steps you ask for)
# 'Crank-Nicholson' is what it sounds like: fully implicit,
# centered in space and time, second order
time_step_mode_coupled_left = 'semi-implicit'
time_step_mode_coupled_right = 'semi-implicit'
time_step_mode_regular = 'semi-implicit'

# boundary values for phi
lbval = 0.; rbval = 0.

## Plot the forcing profile
#x = np.linspace(0.,1.,num=100)
#plt.figure('Forcing Profile')
#plt.plot(x,CDT.CleanDrive(x,0,0))

# Function defining how to average over the overlap region
def Averager(x):
    return x

num_xpts = 100
# Build a standard solver object
mySolver = CH.ConvectionDiffusion1D(num_xpts,CDT.Convec,CDT.Diffuse,CDT.Drive,CDT.PoissRHS,
                                    lbval,rbval,tstep_mode=time_step_mode_regular)

# Build coupled solver object
mySys = CH.VolumetricCoupledSystem1D(lu_lim,rl_lim,ncls_l,ncls_r,
                                    CDT.Convec,CDT.Diffuse,CDT.Drive,CDT.PoissRHS,Averager,
                                    lbval,rbval,knot_mode=knotting,
                                    ltstep_mode=time_step_mode_coupled_left,
                                    rtstep_mode=time_step_mode_coupled_right,
                                    left_buffer_width=lbuff,
                                    right_buffer_width=rbuff)


# Initialize rho
mySys.rhol = CDT.InitData(mySys.xl)
mySys.rhor = CDT.InitData(mySys.xr)
mySolver.rho = CDT.InitData(mySolver.x)
# Run simulation
solvals = mySys.RunForFixedTime(TotTime,numsteps,0.,0.,100,field_mode=runmode)
uncoupled_solvals = mySolver.RunForFixedTime(TotTime,numsteps,0.,0.)
"""eq_sol_coupled = mySys.RunToEQ(TotTime/numsteps,Averager,0.,0.,mySolver.x.shape[0],field_mode=runmode,maxsteps=10000,errtol=1.e-7)
eq_sol_normal = mySolver.RunToEQ(TotTime/numsteps,0.,0.,errtol=1.e-7,maxsteps=10000)

l2_err = np.sqrt(np.mean((eq_sol_coupled - eq_sol_normal)**2))/np.sqrt(np.mean(eq_sol_normal**2))
print 'For alpha = ' + str(alpha)
print 'Fractional L^2 error = ' + str(l2_err)

plt.figure()
x_plot = np.linspace(0.,1.,num=eq_sol_coupled.shape[0])
plt.plot(x_plot,eq_sol_coupled,lw=2)
plt.plot(mySolver.x,eq_sol_normal,'r',lw=1)
"""

## Compute (and plot) time averaged profiles at end of sim
CoupledSteadyState = np.mean(solvals[start_t_avg:],axis=0)
NormalSteadyState = np.mean(uncoupled_solvals[start_t_avg:],axis=0)
ss_fig = plt.figure('Steady State Comparison')
ss_ax = plt.axes()
x_plot = np.linspace(0.,1.,num=solvals[0,:].shape[0])
ss_ax.plot(x_plot,CoupledSteadyState,lw=2,label='Coupled numerical')
ss_ax.plot(mySolver.x,NormalSteadyState,'r',lw=2,label='Standard numerical')
ss_ax.plot(mySolver.x,CDT.AnalyticSteadyState(mySolver.x),'g',lw=2,label='Analytic')
ss_ax.legend(loc=2)

ss_axins = zoomed_inset_axes(ss_ax,8,loc=8)
ss_axins.plot(x_plot,CoupledSteadyState,lw=1)
ss_axins.plot(mySolver.x,NormalSteadyState,'r',lw=1)
ss_axins.plot(mySolver.x,CDT.AnalyticSteadyState(mySolver.x),'g',lw=1)
ss_axins.set_xlim(0.7,0.8); ss_axins.set_ylim(1.03,1.06)
plt.xticks(visible=False); plt.yticks(visible=False)

mark_inset(ss_ax,ss_axins,loc1=2,loc2=1,fc='none',ec='0.5')

if save:
    pname = 'Plots/' + animName + '_eqplot.png'
    ss_fig.savefig(pname)

## Animate result ##
fig = plt.figure('Animation')
ax = plt.axes(xlim=(0,1),ylim=(0,ymax_for_plot))
line, = ax.plot([],[],lw=2,label='Coupled Solution')
line3, = ax.plot([],[],'r',lw=2,label='Regular Solution')
line2, = ax.plot(mySolver.x,CDT.AnalyticSteadyState(mySolver.x),
                'g',lw=2,label='Analytic steady state')
ax.legend(loc=2)
time_text = ax.text(0.8,0.9*ymax_for_plot,'')

dt = TotTime/numsteps

def init():
    line.set_data([],[])
    line3.set_data([],[])
    time_text.set_text('time = 0.0')
    return line, line3, time_text

def animate(i):
    line.set_data(x_plot,solvals[i,:])
    line3.set_data(mySolver.x,uncoupled_solvals[i,:])
    time = i*dt
    time_text.set_text('time = %.1f' % time)

    if save:
        fr = animName + '%03d.png' %i
        framename = 'AnimFrames/' + fr
        fig.savefig(framename)

    return line, line3, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=numsteps,
    interval=60)

plt.show()


