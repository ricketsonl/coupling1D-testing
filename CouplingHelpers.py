#############################################################################################
#############################################################################################
# Main class for holding data about a function defined on
# two overlapping domains, and updating them to (hopefully)
# convergence via some model PDE that they solve.
#############################################################################################
#############################################################################################

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.sparse import diags, isspmatrix_csr
import scipy.sparse.linalg as spla
from scipy.optimize import newton_krylov


def Poisson1D_Dirichlet(domain_width,RHS,val_left,val_right):
    npts = RHS.shape[0]-2
    h = domain_width/(npts+1)
    K = diags([-2./h**2,1./h**2,1./h**2],[0,1,-1],shape=(npts,npts),format='csr')
    RHS_VEC = RHS[1:-1]
    RHS_VEC[0] -= val_left/h**2; RHS_VEC[-1] -= val_right/h**2

    SOL = spla.spsolve(K,RHS_VEC)
    sol = np.zeros(npts+2)
    sol[1:-1] = SOL; sol[0] = val_left; sol[-1] = val_right

    return sol

class ConvectionDiffusion1D:
    def __init__(self,ncls,Convec,Diffuse,Drive,F,lbdryval,rbdryval,tstep_mode='semi-implicit'):
        self.Convec = Convec; self.Diffuse = Diffuse; self.Drive = Drive; self.F = F
        
        self.x = np.linspace(0.,1.,num=ncls)
        self.dx = self.x[1] - self.x[0]
        
        self.rho = np.zeros_like(self.x)
        self.phi = np.zeros_like(self.x)
        
        self.lbdryval = lbdryval; self.rbdryval = rbdryval
        
        self.tstep_mode = tstep_mode

        self.t = 0.
        
    def TimeStep(self,dt,lfval,rfval):
        self.phi = Poisson1D_Dirichlet(1.,self.F(self.x,self.rho),lfval,rfval)
         
        nunknwns = self.x.shape[0]-2
        domsize = 1.
        Conv_Coeffs = self.Convec(self.x,self.rho,self.phi,dom_size=domsize)
        Diff_Coeffs = self.Diffuse(self.x,self.rho,self.phi)
        Drive_Vec = self.Drive(self.x,self.rho,self.phi,self.t)
        
        if self.tstep_mode == 'semi-implicit':
            I = diags([1.],[0],shape=(nunknwns,nunknwns),format='csr')
            # Finite difference matrix for first derivative
            D1 = diags([0.5/self.dx, -0.5/self.dx],[1,-1],shape=(nunknwns,nunknwns),format='csr')
            # Finite difference matrix for second derivative
            D2 = diags([-2./self.dx**2,1./self.dx**2,1./self.dx**2],[0,1,-1],shape=(nunknwns,nunknwns),format='csr')
            Conv_mat = diags(Conv_Coeffs[1:-1],0,format='csr')
            Diff_mat = diags(Diff_Coeffs[1:-1],0,format='csr')
            K = I + dt*(Conv_mat.dot(D1) - Diff_mat.dot(D2))

            # Build right hand side vector
            RHS = self.rho[1:-1] + dt*Drive_Vec[1:-1]
            RHS[0] += self.lbdryval*dt*(0.5*Conv_Coeffs[0]/self.dx + Diff_Coeffs[0]/self.dx**2)
            RHS[-1] += self.rbdryval*dt*(-0.5*Conv_Coeffs[-1]/self.dx + Diff_Coeffs[-1]/self.dx**2)
        
            # Update solution rho
            self.rho[0] = self.lbdryval; self.rho[-1] = self.rbdryval
            self.rho[1:-1] = spla.spsolve(K,RHS)

        elif self.tstep_mode == 'explicit':
            self.rho[1:-1] += -dt*Conv_Coeffs[1:-1]*(self.rho[2:] - self.rho[:-2])/(2.*self.dx) \
                              +dt*Diff_Coeffs[1:-1]*(self.rho[2:]-2.*self.rho[1:-1]+self.rho[:-2])/(self.dx*self.dx) \
                              +dt*Drive_Vec[1:-1]
            
        elif self.tstep_mode == 'Crank-Nicholson':
            def Func(r):
                resid = np.zeros_like(r)
                rho_mid = 0.5*(self.rho + r)
                p = Poisson1D_Dirichlet(1.,self.F(self.x,rho_mid),lfval,rfval)
                C = self.Convec(self.x,rho_mid,p)
                D = self.Diffuse(self.x,rho_mid,p)
                Dr = self.Drive(self.x,rho_mid,p,self.t+0.5*dt)
                resid[1:-1] = (r[1:-1] - self.rho[1:-1]) + dt*C[1:-1]*(rho_mid[2:] - rho_mid[:-2])/(2.*self.dx) \
                                - dt*D[1:-1]*(rho_mid[2:] - 2.*rho_mid[1:-1] + rho_mid[:-2])/(self.dx*self.dx) \
                                - dt*Dr[1:-1]

                resid[0] = r[0] - self.lbdryval
                resid[-1] = r[-1] - self.rbdryval

                return resid

            self.rho = newton_krylov(Func,self.rho)
 
        self.t += dt

    def RunForFixedTime(self,T,numsteps,lfval,rfval):
        if self.tstep_mode == 'explicit':
            print 'Explicit mode - need to override time step for CFL safety'
            x = np.linspace(0.,1.,num=100)
            Diffuse_Max = np.amax(self.Diffuse(x,
                          self.rho,
                          self.phi))
            dt = 0.5*self.dx**2/(Diffuse_Max*1.3)
            real_numsteps = int(T/dt) + 1
            dt = T/real_numsteps
            print 'Must take ' + str(real_numsteps) + ' steps.'
        else:
            dt = T/numsteps
            real_numsteps = numsteps
         

        rhovec = np.zeros([real_numsteps+1,self.x.shape[0]])
        rhovec[0,:] =  self.rho

        for i in range(real_numsteps):
            self.TimeStep(dt,lfval,rfval)
            rhovec[i+1,:] = self.rho
     
        if real_numsteps == numsteps:
            return rhovec
        else: 
            true_rhovec = np.zeros([numsteps+1,self.x.shape[0]])
            true_rhovec[0,:] = rhovec[0,:]
            dt_desired = T/numsteps
            for i in range(numsteps):
                t_desired = (i+1)*dt_desired
                left_step = int(t_desired/dt)
                true_rhovec[i+1,:] = rhovec[left_step,:]

            return true_rhovec
        

    def RunToEQ(self,dt,lfval,rfval,maxsteps=2000,errtol=1.e-6):
        rho_new = np.zeros_like(self.rho)
        err = 1.
        it = 0
        while(err > errtol and it < maxsteps):
            self.TimeStep(dt,lfval,rfval)
            err = np.sqrt(np.mean((rho_new - self.rho)**2))
            it += 1
            rho_new = np.copy(self.rho)
        if(err > errtol):
            print 'Failed to converge in allowed steps'

        return self.rho

####################################################################################
# Class for storing data and operations on domain-decomposed
# 1-D solution of a convection-diffusion system.  Solution 
# is rho, which is coupled to a potential phi.  Solution
# on [0,1] is decomposed into solutions on [0,lu_lim] and
# [rl_lim,1], with lu_lim > rl_lim.  
# The PDE to be solved is of the form:
# rho_t + Convec(x,rho,phi)*rho_x = Diffuse(x,rho,phi)*rho_{xx} + Drive(x,rho,phi,t)
# Laplacian(phi) = F(x,rho)
####################################################################################
class VolumetricCoupledSystem1D:
    def __init__(self,lu_lim,rl_lim,ncls_left,ncls_right,Convec,Diffuse,Drive,F,
                 lbdryval,rbdryval,lbdrytype='dirichlet',rbdrytype='dirichlet',
                 knot_mode='consistent',ltstep_mode='semi-implicit',rtstep_mode='semi-implicit',
                 left_buffer_width=0.,right_buffer_width=0.):

        self.Convec = Convec; self.Diffuse = Diffuse; self.Drive = Drive; self.F = F

        self.xl = np.linspace(0.,lu_lim,num=ncls_left+1)
        self.xr = np.linspace(rl_lim,1.,num=ncls_right+1)

        self.lu_lim = lu_lim; self.rl_lim = rl_lim
        self.dxl = self.xl[1] - self.xl[0]; self.dxr = self.xr[1] - self.xr[0]

        self.rhol = np.zeros_like(self.xl)
        self.phil = np.zeros_like(self.xl)
        
        self.rhor = np.zeros_like(self.xr)
        self.phir = np.zeros_like(self.xr)

        self.lbdrytype = lbdrytype; self.rbdrytype = rbdrytype
        self.lbdryval = lbdryval; self.rbdryval = rbdryval

        self.knot_mode=knot_mode
        self.ltstep_mode=ltstep_mode
        self.rtstep_mode=rtstep_mode

        self.left_buffer_width = left_buffer_width
        self.right_buffer_width = right_buffer_width

        self.t = 0.
    
    # x should be a monotonically increasing sequence of values in [0,1]
    # AveragingMethod(x) is a monotonically increasing function satisfying
    # AveragingMethod(0) = 0, AveragingMethod(1) = 1. It defines the method
    # by which the two solutions are averaged in the overlap region.
    def CompositeSolution(self,AveragingMethod,x):
        rho_comp = np.zeros_like(x); phi_comp = np.zeros_like(x)
        
        left_x = x[x<self.rl_lim+self.left_buffer_width]; right_x = x[x>self.lu_lim-self.right_buffer_width]
        mid_x = x[x<=self.lu_lim-self.right_buffer_width]; mid_x = mid_x[mid_x>=self.rl_lim+self.left_buffer_width]
        mid_start = np.argmax(x>=self.rl_lim+self.left_buffer_width); right_start = np.argmax(x>self.lu_lim-self.right_buffer_width)

        rhol_spl = InterpolatedUnivariateSpline(self.xl,self.rhol)
        phil_spl = InterpolatedUnivariateSpline(self.xl,self.phil)
        rho_comp[:mid_start] = rhol_spl(left_x)
        phi_comp[:mid_start] = phil_spl(left_x)

        rhor_spl = InterpolatedUnivariateSpline(self.xr,self.rhor)
        phir_spl = InterpolatedUnivariateSpline(self.xl,self.phir) 
        rho_comp[right_start:] = rhor_spl(right_x)
        phi_comp[right_start:] = phir_spl(right_x)
        
        avg_x = (mid_x - mid_x[0])/(x[right_start] - mid_x[0])

        rho_comp[mid_start:right_start] = AveragingMethod(avg_x)*rhor_spl(mid_x) + (1. - AveragingMethod(avg_x))*rhol_spl(mid_x)
        phi_comp[mid_start:right_start] = AveragingMethod(avg_x)*phir_spl(mid_x) + (1. - AveragingMethod(avg_x))*phil_spl(mid_x)
        
        return rho_comp
    
    # Compute the field phi on the entire domain, with Dirichlet 
    # boundary conditions phi(0) = lfval, phi(1) = rfval.  
    # Specifically, solve: Laplacian(phi) = F(x,rho_comp)
    def CompositeField(self,AveragingMethod,npts,lfval,rfval):
        x = np.linspace(0.,1.,num=npts)
        rho_comp = self.CompositeSolution(AveragingMethod,x)
        phi_comp = Poisson1D_Dirichlet(1.,self.F(x,rho_comp),lfval,rfval)

        return phi_comp

    def SeparateFields(self,lfval,rfval):
        phil_spl = InterpolatedUnivariateSpline(self.xl,self.phil)
        phir_spl = InterpolatedUnivariateSpline(self.xr,self.phir)
        lintbdryval = phil_spl(self.xr[0])
        rintbdryval = phir_spl(self.xl[-1])
         
        Fl = self.F(self.xl,self.rhol)
        Fr = self.F(self.xr,self.rhor)
        self.phil = Poisson1D_Dirichlet(self.lu_lim,Fl,lfval,rintbdryval)
        self.phir = Poisson1D_Dirichlet(1.-self.rl_lim,Fr,lintbdryval,rfval)

    def LeftTimeStep_CompositeField(self,dt,rbdryval,x_comp,phi_comp):
        # Compute composite field on left-side grid
        phi_spl = InterpolatedUnivariateSpline(x_comp,phi_comp)
        phi_left_from_comp = phi_spl(self.xl)

        ldomsize = self.xl[-1] - self.xl[0]
       
        Conv_Coeffs = self.Convec(self.xl,self.rhol,phi_left_from_comp,dom_size=ldomsize)
        Diff_Coeffs = self.Diffuse(self.xl,self.rhol,phi_left_from_comp)
        Drive_Vec = self.Drive(self.xl,self.rhol,phi_left_from_comp,self.t)
 
        if self.ltstep_mode == 'semi-implicit':
            # Build sparse matrix for linear solve
            nunknwns = self.xl.shape[0]-2
            I = diags([1.],[0],shape=(nunknwns,nunknwns),format='csr')
            # Finite difference matrix for first derivative
            D1 = diags([0.5/self.dxl, -0.5/self.dxl],[1,-1],shape=(nunknwns,nunknwns),format='csr')
            # Finite difference matrix for second derivative
            D2 = diags([-2./self.dxl**2,1./self.dxl**2,1./self.dxl**2],[0,1,-1],shape=(nunknwns,nunknwns),format='csr')
            Conv_mat = diags(Conv_Coeffs[1:-1],0,format='csr')
            Diff_mat = diags(Diff_Coeffs[1:-1],0,format='csr')
            K = I + dt*(Conv_mat.dot(D1) - Diff_mat.dot(D2))
             
            # Build right hand side vector
            RHS = self.rhol[1:-1] + dt*Drive_Vec[1:-1]
            RHS[0] += self.lbdryval*dt*(0.5*Conv_Coeffs[0]/self.dxl + Diff_Coeffs[0]/self.dxl**2)
            RHS[-1] += rbdryval*dt*(-0.5*Conv_Coeffs[-1]/self.dxl + Diff_Coeffs[-1]/self.dxl**2)
            
            # Update solution rho
            self.rhol[0] = self.lbdryval; self.rhol[-1] = rbdryval
            self.rhol[1:-1] = spla.spsolve(K,RHS)
    
        elif self.ltstep_mode == 'explicit':
            self.rhol[-1] = rbdryval
            self.rhol[1:-1] += -dt*Conv_Coeffs[1:-1]*(self.rhol[2:]-self.rhol[:-2])/(2.*self.dxl) \
                              +dt*Diff_Coeffs[1:-1]*(self.rhol[2:]-2.*self.rhol[1:-1]+
                                    self.rhol[:-2])/(self.dxl*self.dxl) \
                              +dt*Drive_Vec[1:-1]

        elif self.ltstep_mode == 'Crank-Nicholson':
            def Func(r):
                resid = np.zeros_like(r)
                rho_mid = 0.5*(self.rhol + r)
                p = Poisson1D_Dirichlet(self.lu_lim,self.F(self.xl,rho_mid),
                                        phi_left_from_comp[0],phi_left_from_comp[-1])
                C = self.Convec(self.xl,rho_mid,p,dom_size=ldomsize)
                D = self.Diffuse(self.xl,rho_mid,p)
                Dr = self.Drive(self.xl,rho_mid,p,self.t+0.5*dt)
                resid[1:-1] = (r[1:-1] - self.rhol[1:-1]) + dt*C[1:-1]*(rho_mid[2:] - rho_mid[:-2])/(2.*self.dxl) \
                                - dt*D[1:-1]*(rho_mid[2:] - 2.*rho_mid[1:-1] + rho_mid[:-2])/(self.dxl*self.dxl) \
                                - dt*Dr[1:-1]

                resid[0] = r[0] - self.lbdryval
                resid[-1] = r[-1] - rbdryval

                return resid

            self.rhol = newton_krylov(Func,self.rhol)            

    def RightTimeStep_CompositeField(self,dt,lbdryval,x_comp,phi_comp):
        # Compute composite field on left-side grid
        phi_spr = InterpolatedUnivariateSpline(x_comp,phi_comp)
        phi_right_from_comp = phi_spr(self.xr)
        
        # Build sparse matrix for linear solve
        nunknwns = self.xr.shape[0]-2
        rdomsize = self.xr[-1] - self.xr[0]
        Conv_Coeffs = self.Convec(self.xr,self.rhor,phi_right_from_comp,dom_size=rdomsize)
        Diff_Coeffs = self.Diffuse(self.xr,self.rhor,phi_right_from_comp)
        Drive_Vec = self.Drive(self.xr,self.rhor,phi_right_from_comp,self.t)
        
        if self.rtstep_mode == 'semi-implicit':
            # Build sparse matrix for linear solve
            nunknwns = self.xr.shape[0]-2
            I = diags([1.],[0],shape=(nunknwns,nunknwns),format='csr')
            D1 = diags([0.5/self.dxr, -0.5/self.dxr],[1,-1],shape=(nunknwns,nunknwns),format='csr') # Finite difference matrix for first derivative
            D2 = diags([-2./self.dxr**2,1./self.dxr**2,1./self.dxr**2],[0,1,-1],shape=(nunknwns,nunknwns),format='csr') # Finite difference matrix for second derivative
            rdomsize = self.xr[-1] - self.xr[0]
            Conv_mat = diags(Conv_Coeffs[1:-1],0,format='csr')
            Diff_mat = diags(Diff_Coeffs[1:-1],0,format='csr')
            K = I + dt*(Conv_mat.dot(D1) - Diff_mat.dot(D2))
            
            # Build right hand side vector
            RHS = self.rhor[1:-1] + dt*Drive_Vec[1:-1]
            RHS[-1] += self.rbdryval*dt*(-0.5*Conv_Coeffs[-1]/self.dxr + Diff_Coeffs[-1]/self.dxr**2)
            RHS[0] += lbdryval*dt*(0.5*Conv_Coeffs[0]/self.dxr + Diff_Coeffs[0]/self.dxr**2)
        
            # Update solution rho
            self.rhor[0] = lbdryval; self.rhor[-1] = self.rbdryval
            self.rhor[1:-1] = spla.spsolve(K,RHS)

        elif self.rtstep_mode == 'explicit':
            self.rhor[0] = lbdryval
            self.rhor[1:-1] += -dt*Conv_Coeffs[1:-1]*(self.rhor[2:]-self.rhor[:-2])/(2.*self.dxr) \
                              +dt*Diff_Coeffs[1:-1]*(self.rhor[2:]-2.*self.rhor[1:-1]+
                                    self.rhor[:-2])/(self.dxr*self.dxr) \
                              +dt*Drive_Vec[1:-1]
        
        elif self.rtstep_mode == 'Crank-Nicholson':
            def Func(r):
                resid = np.zeros_like(r)
                rho_mid = 0.5*(self.rhor + r)
                p = Poisson1D_Dirichlet(1.-self.rl_lim,self.F(self.xr,rho_mid),
                                        phi_right_from_comp[0],phi_right_from_comp[-1])
                C = self.Convec(self.xr,rho_mid,p,dom_size=rdomsize)
                D = self.Diffuse(self.xr,rho_mid,p)
                Dr = self.Drive(self.xr,rho_mid,p,self.t+0.5*dt)
                resid[1:-1] = (r[1:-1] - self.rhor[1:-1]) + dt*C[1:-1]*(rho_mid[2:] - rho_mid[:-2])/(2.*self.dxr) \
                                - dt*D[1:-1]*(rho_mid[2:] - 2.*rho_mid[1:-1] + rho_mid[:-2])/(self.dxr*self.dxr) \
                                - dt*Dr[1:-1]

                resid[0] = r[0] - lbdryval
                resid[-1] = r[-1] - self.rbdryval

                return resid

            self.rhor = newton_krylov(Func,self.rhor)            


    def FullTimeStep_CompositeField(self,Averager,dt,lfval,rfval,nfpts):
        x_comp = np.linspace(0.,1.,num=nfpts)
        phi_comp = self.CompositeField(Averager,nfpts,lfval,rfval)
        rhol_spl = InterpolatedUnivariateSpline(self.xl,self.rhol)
        rhor_spl = InterpolatedUnivariateSpline(self.xr,self.rhor)
        if self.knot_mode == 'consistent':
            lintbdryval = rhol_spl(self.xr[0])
            rintbdryval = rhor_spl(self.xl[-1])
        elif self.knot_mode == 'zero':
            lintbdryval = 0.; rintbdryval = 0.
        else:
            print 'Invalid knotting mode... assuming consistent'
            lintbdryval = rhol_spl(self.xr[0])
            rintbdryval = rhor_spl(self.xl[-1])

        self.LeftTimeStep_CompositeField(dt,rintbdryval,x_comp,phi_comp)
        self.RightTimeStep_CompositeField(dt,lintbdryval,x_comp,phi_comp)
        self.t += dt

    def FullTimeStep_SeparateFields(self,dt,lfval,rfval):
        self.SeparateFields(lfval,rfval)
        rhol_spl = InterpolatedUnivariateSpline(self.xl,self.rhol)
        rhor_spl = InterpolatedUnivariateSpline(self.xr,self.rhor)
        if self.knot_mode == 'consistent':
            lintbdryval = rhol_spl(self.xr[0])
            rintbdryval = rhor_spl(self.xl[-1])
        elif self.knot_mode == 'zero':
            lintbdryval = 0.; rintbdryval = 0.
        else:
            print 'Invalid knotting mode... assuming consistent'
            lintbdryval = rhol_spl(self.xr[0])
            rintbdryval = rhor_spl(self.xl[-1])
            

        self.LeftTimeStep_CompositeField(dt,rintbdryval,self.xl,self.phil)
        self.RightTimeStep_CompositeField(dt,lintbdryval,self.xr,self.phir)
        self.t += dt

    def RunForFixedTime(self,T,numsteps,Averager,lfval,rfval,npts,field_mode='composite'):
        
        if self.ltstep_mode == 'explicit' or self.rtstep_mode == 'explicit':
            print 'Explicit mode - need to override time step for CFL safety'
            x = np.linspace(0.,1.,num=100)
            Diffuse_Max = np.amax(self.Diffuse(x,
                          self.CompositeSolution(Averager,x),
                          self.CompositeField(Averager,100,lfval,rfval)))
            dt = 0.5*min(self.dxl,self.dxr)**2/(Diffuse_Max*1.3)
            real_numsteps = int(T/dt) + 1
            dt = T/real_numsteps
            print 'Must take ' + str(real_numsteps) + ' steps.'
        else:
            dt = T/numsteps
            real_numsteps = numsteps

        rhovec = np.zeros([real_numsteps+1,npts])
        x_comp = np.linspace(0.,1.,num=npts)
        rhovec[0,:] =  self.CompositeSolution(Averager,x_comp)

        phi_comp = self.CompositeField(Averager,npts,lfval,rfval)
        phi_spl = InterpolatedUnivariateSpline(x_comp,phi_comp)
        self.phil = phi_spl(self.xl); self.phir = phi_spl(self.xr)
        for i in range(real_numsteps):
            if field_mode=='composite':
                self.FullTimeStep_CompositeField(Averager,dt,lfval,rfval,npts)
            else:
                self.FullTimeStep_SeparateFields(dt,lfval,rfval)

            rhovec[i+1,:] = self.CompositeSolution(Averager,x_comp)
        
        if real_numsteps == numsteps:
            return rhovec
        else: 
            true_rhovec = np.zeros([numsteps+1,npts])
            true_rhovec[0,:] = rhovec[0,:]
            dt_desired = T/numsteps
            for i in range(numsteps):
                t_desired = (i+1)*dt_desired
                left_step = int(t_desired/dt)
                true_rhovec[i+1,:] = rhovec[left_step,:]

            return true_rhovec

    def RunToEQ(self,dt,Averager,lfval,rfval,npts,field_mode='composite',maxsteps=10000,errtol=1.e-6):
        x_comp = np.linspace(0.,1.,num=npts)
        rho_new = np.zeros_like(x_comp)
        err = 1.
        it = 0
        while(err > errtol and it < maxsteps):
            if field_mode=='composite':
                self.FullTimeStep_CompositeField(Averager,dt,lfval,rfval,npts)
            else:
                self.FullTimeStep_SeparateFields(dt,lfval,rfval)
            rho_comp = self.CompositeSolution(Averager,x_comp)
            err = np.sqrt(np.mean((rho_comp - rho_new)**2))
            it += 1
            rho_new = rho_comp
        if(err > errtol):
            print 'Failed to converge in allowed number of steps'
        
        return rho_new
     
