from problem_base import *

def problem():
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    V = VectorFunctionSpace(mesh, E, 2)

    w = Function(W)
    (p, c) = split(w)
    w_ = project(as_vector([p0, c0]), W)
    (p_, c_) = split(w_)
    w.assign(w_)
    (q, v) = TestFunction(W)

    Fp = ipm(ka, p, q, 1) \
       + ipm(kc, p, q, 2) \
       + ipm(ke, pec(c, p), q, 3) \
       - ia(s(c), l(c), s(p), l(p))*jump(q)*dS(4) \
       - ic(s(c), l(c), s(p), l(p))*jump(q)*dS(5)
    if dirichlet_pa:
        Fp += - dot(grad(q), p*n)*ds(6) - dot(q*n, grad(p))*ds(6) \
              + (gamma/h)*p*q*ds(6) + Constant(0)*dot(grad(q), n)*ds(6)
    else:
        Fp += - g*q*ds(6)
    if dirichlet_pc:
        Fp += - dot(grad(q), p*n)*ds(7) - dot(q*n, grad(p))*ds(7) \
              + (gamma/h)*p*q*ds(7) + Constant(0)*dot(grad(q), n)*ds(7)
    else:
        Fp += + g*q*ds(7)

    Fc = (c-c_)/dt*v*dx \
       + ipm(Da, c, v, 1) \
       + ipm(Dc, c, v, 2) \
       + ipm(De, cep(c, p), v, 3) \
       - ia(s(c), l(c), s(p), l(p))/F*jump(v)*dS(4) \
       - ic(s(c), l(c), s(p), l(p))/F*jump(v)*dS(5)
    p, c = w.split()

    global t
    kk = DomainConstant(domains, [0, 0, 0, ke])
    D = DomainConstant(domains, [0, Da, Dc, De])
    DD = DomainConstant(domains, [0, 0, 0, 1.0])
    j = Function(V)
    N = Function(V)

    while t < tf:
        t += dt
        timer = Timer("Newton solver")
        (ni, converged) = newton(Fp + F/ce0*Fc == 0, w)
        timer.stop()
        if not converged:
            return p, c, w, j, N
        w_.assign(w)
        pa_cur = p(0, S/2)
        w.assign(w - project(Constant([pa_cur, 0]), W))
        assign(j, project(-k*grad(p) + kk*(1-te)*RTF*grad(ln(c)), V))
        assign(N, project(-D*grad(c) + DD*te/F*j, V))
        iav = assemble(ia(s(c), l(c), s(p), l(p))*dS(4))/g/Gc
        icv = assemble(ic(s(c), l(c), s(p), l(p))*dS(5))/g/Gc
        info('t %G n %d ia %G ic %G' % (t, ni, iav, icv))
        yield w, p, c, j, N

if __name__ == "__main__":
    global t
    path = "dg/coupled" + suffix + "/"
    file_c = XDMFFile(path + "c.xdmf")
    file_p = XDMFFile(path + "p.xdmf")
    file_j = XDMFFile(path + "j.xdmf")
    file_N = XDMFFile(path + "N.xdmf")
    for w, p, c, j, N in problem():
        timer = Timer("XML write")
        File(path + "w/w%g.xml" % t) << w
        timer.stop()
        timer = Timer("XDMF write")
        file_c.write(c, t)
        file_p.write(p, t)
        file_j.write(j, t)
        file_N.write(N, t)
        timer.stop()
    list_timings(TimingClear_clear, [TimingType_wall])

