# module expint
import numpy as np
import scipy.linalg as la

##########################
# Matrix functions using scipy.linalg.funm
# Special care is given to small arguments for numerical stability (e.g. 
# expm1 instead of exp and using leading order Taylor expansion when x 
# is smaller than some threshold).
# Alternatively, we can also use BigFloat for higher precision.

@np.vectorize
def _phi(x):
    # phi(x) = (exp(x) - 1) / x
    if x == 0.0:
        return 1.0
    else:
        return np.expm1(x)/x # this is stabel

@np.vectorize
def _phi2(x):
    # phi2(x) = (exp(x) - 1 - x) / x^2
    #         = 1/2 + 1/6x + O(x^2)
    if np.abs(x) < 1e-7:
        return 0.5 + 1/6*x
    else:
        return (np.expm1(x) - x) / x**2

@np.vectorize
def _phi3(x):
    # phi3(x) = (exp(x) - 1 - x - 0.5x^2) / x^3
    #         = 1/6 + 1/24*x + O(x^2)
    if np.abs(x) < 1e-5:
        return 1/6 + 1/24*x
    else:
        return (np.expm1(x) - x - 0.5*x**2) / x**3

expm = la.expm
def phim(A):
    return la.funm(A, _phi)
def phi2m(A):
    return la.funm(A, _phi2)
def phi3m(A):
    return la.funm(A, _phi3)

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

    def __str__(self):
        # For pretty printing
        return "{} solver\nt = {}\ny = {}".format(type(self).__name__, self.t, self.y)

class LawsonEuler(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        self.exphL = expm(dt*L)

    def step(self):
        t, y, dt, exphL = self.t, self.y, self.dt, self.exphL
        nl = self.N(t, y)
        self.y = exphL @ (y + dt*nl)
        self.t = t + dt

class NorsettEuler(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        self.exphL = expm(dt*L)
        self.phihL = phim(dt*L)

    def step(self):
        t, y, dt, exphL, phihL = self.t, self.y, self.dt, self.exphL, self.phihL
        nl = self.N(t, y)
        self.y = exphL @ y + dt * (phihL @ nl)
        self.t = t + dt

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
