import numpy as np

#########################
# Test example: 1d Allen Cahn equation 
# d/dt u(t,x) = D*(d/dx)^2 u(t,x) + u(t,x) - u(t,x)^3, 0 <= x <= R
# 
# Transform to discrete approximation by deviding the space to m points:
# semilinear system y'(t) = Ly(t) + N(t,y)

def AllenCahn(m, R, D):
	Ld = np.identity(m)
	Ld = Ld * (-2)
	for i in range(m-1):
		Ld[i+1,i] = 1
	for i in range(m-1):
		Ld[i,i+1] = 1
	dx = R/(m+1)
	L = D / (dx**2) * Ld + np.identity(m)
	N = lambda t, u: -u**3
	X = np.arange(1, m+1) * dx
	return L, N, X
