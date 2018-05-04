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
def _phi1(x):
    # phi1(x) = (exp(x) - 1) / x
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
def phi1m(A):
    return la.funm(A, _phi1)
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

# First order methods
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
        self.phihL = phi1m(dt*L)

    def step(self):
        t, y, dt, exphL, phihL = self.t, self.y, self.dt, self.exphL, self.phihL
        nl = self.N(t, y)
        self.y = exphL @ y + dt * (phihL @ nl)
        self.t = t + dt

# Second order methods
class ExpMidpoint(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        hL = dt * L
        half_hL = dt/2 * L
        self.E = expm(hL)
        self.Emid = expm(half_hL)
        self.P = phi1m(hL)
        self.Pmid = phi1m(half_hL)

    def step(self):
        t, y, dt = self.t, self.y, self.dt
        E, Emid, P, Pmid = self.E, self.Emid, self.P, self.Pmid
        N1 = self.N(t, y)
        Y2 = Emid @ y + 0.5*dt*(Pmid @ N1)
        N2 = self.N(t + 0.5*dt, Y2)
        self.y = E @ y + dt*(P @ N2)
        self.t = t + dt

class ExpTrapezoid(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        hL = dt * L
        self.exphL = expm(hL)
        self.phi1hL = phi1m(hL)
        self.phi2hL = phi2m(hL)

    def step(self):
        t, y, dt = self.t, self.y, self.dt
        exphL, phi1hL, phi2hL = self.exphL, self.phi1hL, self.phi2hL
        Ey = exphL @ y # exp(dt*L) * y
        N1 = self.N(t, y)
        P1N1 = phi1hL @ N1 # phi1(dt*L) * N1
        P2N1 = phi2hL @ N1 # phi2(dt*L) * N1
        Y2 = Ey + dt*P1N1
        N2 = self.N(t+dt, Y2)
        P2N2 = phi2hL @ N2 # phi2(dt*L) * N2
        self.y = Ey + dt*(P1N1 - P2N1 + P2N2)
        self.t = t + dt

# Fourth order methods
class ETDRK4(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        # Precompute matrix functions
        hL = dt * L
        half_hL = dt/2 * L
        self.E = expm(hL)
        self.Emid = expm(half_hL)
        self.P1 = phi1m(hL)
        self.P1mid = phi1m(half_hL)
        self.P2 = phi2m(hL)
        self.P3 = phi3m(hL)
        self.B1 = self.P1 - 3*self.P2 + 4*self.P3
        self.B2 = 2*self.P2 - 4*self.P3 # same as B3
        self.B4 = 4*self.P3 - self.P2
    
    def step(self):
        # Unpack variables
        t, y, dt = self.t, self.y, self.dt
        E, Emid = self.E, self.Emid
        P1, P1mid, P2, P3 = self.P1, self.P1mid, self.P2, self.P3
        B1, B2, B4 = self.B1, self.B2, self.B4
        Ey = E @ y
        Emidy = Emid @ y
        # Stage 1
        N1 = self.N(t, y)
        # Stage 2
        Y2 = Emidy + dt/2*(P1mid @ N1)
        N2 = self.N(t + dt/2, Y2)
        # Stage 3
        Y3 = Emidy + dt/2*(P1mid @ N2)
        N3 = self.N(t + dt/2, Y3)
        # Stage 4
        Y4 = Emid @ Y2 + dt/2*(P1mid @ (2*N3 - N1))
        N4 = self.N(t + dt, Y4)
        self.y = Ey + dt*(B1 @ N1 + B2 @ (N2 + N3) + B4 @ N4)
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
