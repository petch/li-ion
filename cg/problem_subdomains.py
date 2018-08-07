from problem_base import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class SubdomainProblem(Problem):
    def __init__(s, name):
        super(SubdomainProblem, s).__init__(name)
        timer = Timer('Init subdomain problem')
        s.a.nit = s.e.nit = s.c.nit = 0
        s.a.kit = s.e.kit = s.c.kit = 0
        timer.stop()

    def newton_step_local(s, l):
        timer = Timer("Assemble problems")
        A = assemble(l.dF[0])
        b = assemble(-l.F)
        timer.stop()

        timer = Timer('Apply boundary conditions')
        if l.dbc is not None:
            l.dbc.apply(A)
            l.dbc.apply(b)
        timer.stop()

        ki, y = s.solve(s.to_csr(A), b)

        timer = Timer('Vector to function')
        l.dw.vector()[:] = y
        l.w.vector()[:] += l.dw.vector()[:]
        timer.stop()

        dwn, wn = l.norms()
        rn = dwn/wn
        return ki, rn, dwn

    def newton(s, t, l):
        timer = Timer('Newton')
        rt = 1 + rt_min
        at = 1 + at_min
        ni = kil = 0
        while rt > rt_min and at > at_min and ni < ni_max:
            ni += 1
            ki, rt, at = s.newton_step_local(l)
            kil += ki
            # info('n %d k %d' % (ni, ki) + ' r {:.1e}'.format(rt) + ' a {:.1e}'.format(at))
        info('t %G n %d k %d' % (t, ni, kil) + ' r {:.1e}'.format(rt) + ' a {:.1e}'.format(at) + ' ia %G' % (s.ia/g) + ' ic %G' % (s.ic/g))
        l.nit += ni
        l.kit += kil
        timer.stop()
        return not math.isnan(at)

    def time_step(s, t):
        timer = Timer('Time step')
        is_performed = s.newton(t, s.a) and s.newton(t, s.c) and s.newton(t, s.e)
        timer.stop()
        return is_performed

    def simulate(s):
        super(SubdomainProblem, s).simulate()
        p = dt/s.t
        info('Newton iterations %G, %G, %G, per time step %G, %G, %G'   % (s.a.nit, s.c.nit, s.e.nit, s.a.nit*p, s.c.nit*p, s.e.nit*p))
        pa, pc, pe = 1./s.a.nit, 1./s.c.nit, 1./s.e.nit
        info('Krylov iterations %G, %G, %G, per newton step %G, %G, %G' % (s.a.kit, s.c.kit, s.e.kit, s.a.kit*pa, s.c.kit*pc, s.e.kit*pe))

if __name__ == "__main__":
    SubdomainProblem("subdomains").simulate()
