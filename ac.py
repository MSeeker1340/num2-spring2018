import numpy as np

#########################
# Test example: 1d Allen Cahn equation 
# d/dt u(t,x) = D*(d/dx)^2 u(t,x) + u(t,x) - u(t,x)^3, 0 <= x <= 1
# 
# Transform to discrete approximation by deviding the space to m points:
# semilinear system y'(t) = Ly(t) + N(t,y)

def AllenCahn(m, D):
	Ld = np.identity(m, dtype=None)
	Ld = Ld * (-2)
	for i in range(m-1):
		Ld[i+1,i] = 1
	for i in range(m-1):
		Ld[i,i+1] = 1
	dt = 1.0/(m+1)
	L = D * 1/(dt**2) * Ld + np.identity(m, dtype=None)
	N = lambda u: u**3
	return L,N