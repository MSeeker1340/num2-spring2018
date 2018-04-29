# module expint
import numpy as np
import scipy.linalg as la

##########################
# Matrix functions using scipy.linalg.funm
# Special care is given to small arguments for numerical stability (e.g. 
# expm1 instead of exp).

def phi(x):
    # phi(x) = (exp(x) - 1) / x
    if x == 0.0:
        return 1.0
    else:
        return np.expm1(x)/x

def phim(A):
    return la.funm(A, phi)

#########################
# Integrator interface for semilinear problems
# The interface is a simplified version of scipy.integrate.OdeSolver which 
# solves the semilinear system y'(t) = Ly(t) + N(t,y)

class SemilinearOdeSolver:
    def __init__(self, L, N, t0, y0, dt):
        self.L = L
        self.N = N
        self.t = t0
        self.y = y0
        self.dt = dt

class LawsonEuler(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        self.expL = la.expm(L)

    def step():
        pass

class NorsettEuler(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        self.expL = la.expm(L)
        self.phiL = phim(L)

    def step():
        pass

##########################
# Function interface for the solvers, similar to scipy.integrate.solve_ivp
# The interface is much more simplified. We will always use fixed-dt methods 
# and return dense outputs. We also assume that the integration goes forward 
# in time. Functionality related to events handling is also removed, as well 
# as options for Jacobians and tolerances because we do not have implicit 
# methods.

def solve_semilinear(L, N, t_span, y0, dt, method=LawsonEuler):
    t0, tend = t_span
    solver = method(L, N, t0, y0, dt)
    ts = [t0]
    ys = [y0]
    while solver.t < tend:
        solver.step()
        ts.append(solver.t)
        ys.append(solver.y)
    return np.array(ts), np.array(ys)
