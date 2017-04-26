import numpy as np
import matplotlib.pyplot as plt
import ConvecDiffTerms as CDT
import CouplingHelpersExplicit as CHE

ncls = 100
lbdry = 0.; rbdry = 0.

TotTime = 16.
CFLsafety = 1.2

mySolver = CHE.ConvectionDiffusion1D_Explicit(ncls,CDT.Convec,
            CDT.Diffuse,CDT.Drive,CDT.PoissRHS,lbdry,rbdry)

exp_sol = mySolver.RunForFixedTime(TotTime,CFLsafety,0.,0.)

x = np.linspace(0.,1.,num=ncls)

plt.figure()
plt.plot(x,exp_sol[-1,:])
plt.plot(x,CDT.AnalyticSteadyState(x),'g')
plt.show()
