import numpy as np

# parameters for size of convection and diffusion terms, respectively
alpha = 1.35; beta = 0.05

############################################################
## Definition of convection coefficient
############################################################
def Convec(x,rho,phi,dom_size=1.):
    u = -alpha*np.gradient(phi,dom_size/(phi.shape[0]-1),edge_order=2)
    return u

############################################################
## Definition of diffusion coefficient
############################################################
def Diffuse(x,rho,phi):
    return beta*np.ones_like(x)

# Determine coefficients for polynomial that defines forcing #
pin_locs = np.array([0.25, 0.5, 0.75, 1.])
pin_vals = 2.*np.array([0.5, -0.4, 0.8, 0.])
MAT = np.zeros((4,4))
MAT[:,0] = pin_locs; MAT[:,1] = pin_locs**2
MAT[:,2] = pin_locs**3; MAT[:,3] = pin_locs**4
coeffs = np.linalg.solve(MAT,pin_vals)

def CleanDrive(x,rho,phi):
    dr = 4.426575 - 35.50575*x + 158.88915*x**2 \
     -  596.106*x**3 + 1675.593*x**4 - 3134.376*x**5 \
     +  3632.688*x**6 - 2322.432*x**7 + 622.08*x**8
    #dr = coeffs[0]*x + coeffs[1]*x**2 + coeffs[2]*x**3 + coeffs[3]*x**4
    return dr

# Put some random sinusoidal forcing in #
num_terms = 30
min_k = 4.
max_k = 40.; max_freq = 40.; max_amp = 2.5
rand_ks = np.random.uniform(min_k,max_k,num_terms)
rand_freqs = np.random.uniform(1.,max_freq,num_terms)
rand_amps = np.random.uniform(-max_amp,max_amp,num_terms)
rand_phase = np.random.uniform(0.,2.*np.pi,num_terms)
rand_sign = 2.*np.random.binomial(1,0.5,num_terms) - 1.
rand_freqs *= rand_sign

eps = 0.0 # Coefficient that makes fluctuating forcing nonlinear in rho

############################################################
## Definition of forcing term
############################################################
def Drive(x,rho,phi,t):
    # Main forcing "profile"
    dr = CleanDrive(x,rho,phi)
    # Fluctuating forcing
    turb_dr = np.zeros_like(x)
    for i in range(num_terms):
        turb_dr += rand_amps[i]*np.sin(2.*np.pi*(rand_ks[i]*x - rand_freqs[i]*t)+rand_phase[i])
    return dr + turb_dr*(1. + eps*np.sin(rho))

############################################################
## RHS of Poisson equation that defines phi
############################################################
def PoissRHS(x,rho):
    return -(rho - 1.)

############################################################
## Initial Data  ##
############################################################
def InitData(x):
    #rho_init = 0.5*(1. + np.sign((x-0.25)*(0.75-x)))
    #rho_init = 5.*(np.exp(-20.*(x-0.5)**2) - np.exp(-20.*0.25))* x
    rho_init = 7.2*x - 20.*x**2 + 25.6*x**3 - 12.8*x**4
    return rho_init

############################################################
## Exact steady state solution without fluctuations
############################################################
def AnalyticSteadyState(x):
    a = 7.7; b = -28.5; c=44.8; d=-24.
    return a*x + b*x**2 + c*x**3 + d*x**4
