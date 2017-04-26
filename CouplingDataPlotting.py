import numpy as np
import matplotlib.pyplot as plt

# alpha = coefficient on non-linear advection term
alpha = np.array([0.,0.5,1.,1.5,2.,3.,4.,5.])

L2_err_composite = np.array([9.454e-5,7.415e-5,3.726e-4,.0204,.01666,.00921,.00646,.00510])

L2_err_separate = np.array([9.102e-5,7.194e-5,3.554e-4,.0341,.01661,.00918,.006433,.005072])

plt.figure('Error in steady state')
plt.semilogy(alpha,L2_err_composite,'x',mew=2,ms=8,label='Composite field')
plt.semilogy(alpha,L2_err_separate,'rx',mew=2,ms=8,label='Separate fields')
plt.legend(loc=4)
plt.xlabel(r'$\alpha$',fontsize=16)
plt.ylabel(r'$L^2$ error',fontsize=16)
plt.title('Error in steady state',fontsize=20)

plt.show()
