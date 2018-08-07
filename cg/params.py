from model import *
from meshes import *
import numpy as np

# set_log_active(False)
parameters["allow_extrapolation"] = True
parameters["linear_algebra_backend"] = "Eigen"

S = 2e-3
mesh, domains, bounds = paper(S, 20) #parallel(S, 20, H=0.4, AR=4) #plate(S, 20) rectangle cross parallel

t = 0.0     # start time s
dt = 1.    # time step s
tf = 3000.   # finish time s
Crate = -8.0     # Cathode boundary potential flux 1C (A/cm2)
dirichlet_pa = False
dirichlet_pc = False

one = Constant(1)
dx = Measure("dx", subdomain_data=domains)
Va = assemble(one*dx(1, domain=mesh))
Vc = assemble(one*dx(2, domain=mesh))
dS = Measure("dS", subdomain_data=bounds)
Ia = assemble(one*dS(4, domain=mesh))
Ic = assemble(one*dS(5, domain=mesh))
ds = Measure("ds", subdomain_data=bounds)
Ga = assemble(one*ds(6, domain=mesh))
Gc = assemble(one*ds(7, domain=mesh))
print("Sizes: ", [Va, Vc], [Ia, Ic], [Ga, Gc])

ca0 = 0.1*cam   # Initial concentration in anode mol/cm3
cc0 = 0.9*ccm   # Initial concentration in cathode mol/cm3
ce0 = 0.001     # Initial concentration in electrolyte mol/cm3
g = Crate*C1(Vc, Gc)
pa0 = Pa(ca0)
pe0 = Pe(Ia, Gc, g, ce0, ca0, pa0)
pc0 = Pc(Ic, Gc, g, cc0, ce0, pe0)
print("Initial: ", [ca0, ce0, cc0], [pa0, pe0, pc0], g)

ni_max = 20    # max iterations of newton method
rt_min = 1e-6   # relative tolerance of newton method
at_min = 1e-7   # absolute tolerance of newton method

suffix = ('_da' if dirichlet_pa else '_na') \
       + ('_dc' if dirichlet_pc else '_nc') \
       + '/Crate=%G' % Crate 

method = "direct"       #"bicgstab" #
precondition ="none"    #"ilu" # "amg" #
mt = 1e-14              # method tolerance

