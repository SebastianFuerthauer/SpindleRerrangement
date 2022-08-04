from numpy.random import rand, poisson, randint

DT = 0.01
KAPPA = 1.0
ALPHA = 1.0
R = 1.0
VG = 1.0
NUC = 100


def timestep(l):
    l = destroy(l)
    l = cut(l)
    l = add(l)
    l = grow(l)
    return l


def destroy(l):
    out = []
    n = len(l)
    n_actual = poisson(n*R*DT)
    nwhich = randint(n,size=n_actual)
    for i in range(n):
        if i not in nwhich:
            out.append(l[i])
    return out


def cut(l):
    out = []
    for ll in l:
        nl = poisson(ll*KAPPA*DT)
        if nl == 0:
            out.append(ll)
        else:
            cutpos = ll*rand()
            out.append(cutpos)
            if rand()<ALPHA:
                out.append(ll-cutpos)
    return out


def add(l):
    n = poisson(NUC*DT)
    out = l
    for i in range(n):
        out.append(0.0)
    return out

def grow(l):
    return [_l + DT*VG for _l in l]
