# module RK
import numpy as np
from expint import SemilinearOdeSolver

############################
# Implements traditional Runge-Kutta solvers for SemilinearOdeSolver.
# The semilinear structure is not used. Rather, the whole right hand side 
# is treated as a black box function f(t,y).
class SemilinearRKSolver(SemilinearOdeSolver):
    def __init__(self, L, N, t0, y0, dt):
        super().__init__(L, N, t0, y0, dt)
        self.f = lambda t, y: self.L @ y + self.N(t, y)

class RK4(SemilinearRKSolver):
    def step(self):
        t, y, dt, f = self.t, self.y, self.dt, self.f
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2*k1)
        k3 = f(t + dt/2, y + dt/2*k2)
        k4 = f(t + dt, y + dt*k3)
        self.y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.t = t + dt
