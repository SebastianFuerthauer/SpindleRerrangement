import numpy as np
from scipy.linalg import solve

NL = 256
LMAX = 1.0

### Define grid ###
l = np.linspace(0,LMAX,NL)
delta = LMAX/(NL-1)

### Define first derivative ###
dl1 = np.zeros((NL,NL))
for i in range(1,NL-1):
    dl1[i,i-1] = -1
    dl1[i,i+1] = 1
dl1[0,0] = -3
dl1[0,1] = 4
dl1[0,2] = -1
dl1[-1,-1] = 3
dl1[-1,-2] = -4
dl1[-1,-3] = 1
dl1 = dl1/2/delta


### Define second derivative ###
dl2 = np.zeros((NL,NL))
for i in range(1,NL-1):
    dl2[i,i-1] = 1
    dl2[i,i+1] = 1
    dl2[i,i] = -2
dl2[0,0] = 2
dl2[0,1] = -5
dl2[0,2] = 4
dl2[0,3] = -1
dl2[-1,-1] = 2
dl2[-1,-2] =-5
dl2[-1,-3] = 4
dl2[-1,-4] = -1
dl2 = dl2/delta**2

### Define identity operator
iden = np.eye(NL)

def pdist(a, b, alpha):
    """ Calculates the predicted length distributions
        Parameters:
            a: Turnover rate
            b: Cutting rate
            alpha: recovery after cutting
        Output:
            probability distribution
    """
    lhs = dl2 + (a+b*l[:,None])*dl1 + (2+alpha)*b*iden

    lhs[0,:] = dl1[0,:]
    lhs[0,0] = lhs[0,0]+a
    lhs[0,:] = lhs[0,:]

    lhs[-1,:] = b*alpha*l*delta
    lhs[-1,0] = 1

    rhs = np.zeros(NL)
    rhs[0] = b*(1+alpha)
    rhs[-1] = a

    out = solve(lhs, rhs)
    return out
