from problem_base import *

def potential():
    V = FunctionSpace(mesh, "DG", 1)
    bc = DirichletBC(V, pa0, bounds, 6, method="geometric")
    p = project(p0, V)
    print(mesh.num_vertices(), len(p.vector()[:]))
    q = TestFunction(V)
    F = ipm(ka, p, q, 1) \
      + ipm(kc, p, q, 2) \
      + ipm(ke, p, q, 3) \
      - ia(ca0, ce0, s(p), l(p))*jump(q)*dS(4) \
      - ic(cc0, ce0, s(p), l(p))*jump(q)*dS(5) \
      + g*q*ds(7)
    solve(F == 0, p, bc)
    return p

if __name__ == "__main__":
    p = potential()
    plot(p, title="Potential")
    plt.figure()
    plot(k*grad(p), title="Current")
    plt.show()
