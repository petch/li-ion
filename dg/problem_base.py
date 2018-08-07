from params import *

def DomainConstant(domains, values, degree=0):
    f = Expression("v[d]", d=domains, v=Constant(values), degree=degree)
    return project(f, FunctionSpace(domains.mesh(), "DG", degree))

# Interior penalty method
def ipm(k, u, v, i):
    return inner(k*grad(u), grad(v))*dx(i) \
         - dot(k*jump(u, n), avg(grad(v)))*dS(i) \
         - dot(avg(k*grad(u)), jump(v, n))*dS(i) \
         + alpha/h_avg*k*jump(u)*jump(v)*dS(i)

def s(v):
    return conditional(lt(d("+"), 2.5), v("+"), v("-"))
def l(v):
    return conditional(gt(d("+"), 2.5), v("+"), v("-"))

def pec(c, p):
    return p - (1.-te)*RTF*ln(c)
def cep(c, p):
    return c - te/F*ke/De*pec(c, p)


c0 = DomainConstant(domains, [0, ca0, cc0, ce0])
p0 = DomainConstant(domains, [0, pa0, pc0, pe0])
k = DomainConstant(domains, [0, ka, kc, ke])
d = Expression("d", d=domains, degree=0)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2.
alpha = 4.
gamma = 8.

solver_parameters={"newton_solver":{
    "maximum_iterations": ni_max, 
    "relative_tolerance": rt_min, 
    "absolute_tolerance": at_min, 
    "report": False,
    "error_on_nonconvergence": False
}}

def newton(eq, u):
    F = eq.lhs
    J = derivative(F, u)
    problem = NonlinearVariationalProblem(eq.lhs, u, None, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update(solver_parameters)
    (ni, converged) = solver.solve()
    return ni, converged
