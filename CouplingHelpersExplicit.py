import numpy as np
import CouplingHelpers as CH

class ConvectionDiffusion1D_Explicit:
    def __init__(self,ncls,Convec,Diffuse,Drive,F,lbdryval,rbdryval):
        self.Convec = Convec; self.Diffuse = Diffuse; self.Drive = Drive; self.F = F
        
        self.x = np.linspace(0.,1.,num=ncls)
        self.dx = self.x[1] - self.x[0]
        
        self.rho = np.zeros_like(self.x)
        self.phi = np.zeros_like(self.x)
        
        self.lbdryval = lbdryval; self.rbdryval = rbdryval
        
        self.t = 0.
       
    def TimeStep(self,dt,lfval,rfval):
        self.phi = CH.Poisson1D_Dirichlet(1.,self.F(self.x,self.rho),lfval,rfval)

        #rhonew = np.zeros_like(self.x)
        #rhonew[0] = self.lbdryval; rhonew[-1] = self.rbdryval

        C = self.Convec(self.x,self.rho,self.phi)
        D = self.Diffuse(self.x,self.rho,self.phi)
        Dr = self.Drive(self.x,self.rho,self.phi,self.t)

        #rhonew[1:-1] += self.rho[1:-1]
        self.rho[1:-1] += -dt*C[1:-1]*(self.rho[2:] - self.rho[:-2])/(2.*self.dx) \
                          +dt*D[1:-1]*(self.rho[2:] - 2.*self.rho[1:-1] + self.rho[:-2])/(self.dx*self.dx) \
                          +dt*Dr[1:-1]

        self.t += dt

    def RunForFixedTime(self,T,CFL_safety_fac,lfval,rfval):
        Diff_Max = np.amax(self.Diffuse(self.x,self.rho,self.phi))
        dt = 0.5*self.dx*self.dx/(Diff_Max*CFL_safety_fac)
        numsteps = int(T/dt) + 1
        print 'Explicit scheme needs ' + str(numsteps) + ' steps.'
        dt = T/numsteps
        rhovec = np.zeros([numsteps+1,self.x.shape[0]])
        rhovec[0,:] =  self.rho

        for i in range(numsteps):
            self.TimeStep(dt,lfval,rfval)
            rhovec[i+1,:] = self.rho
    
        return rhovec
        
