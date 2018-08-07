from problem_base import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class SubdomainProblem(Problem):
    def __init__(s, name):
        super(SubdomainProblem, s).__init__(name)
        timer = Timer('Init subdomain problem')
        s.nit = 0
        s.kit = 0
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

    def newton(s, t):
        timer = Timer('Newton')
        rt = 1 + rt_min
        at = 1 + at_min
        ni = kil = 0
        while rt > rt_min and at > at_min and ni < ni_max:
            ni += 1
            kia, rta, ata = s.newton_step_local(s.a)
            kic, rtc, atc = s.newton_step_local(s.c)
            kie, rte, ate = s.newton_step_local(s.e)
            kil += kia + kic + kie
            rt = rta + rtc + rte
            at = ata + atc + ate
            # info('n %d k %d' % (ni, ki) + ' r {:.1e}'.format(rt) + ' a {:.1e}'.format(at))
        info('t %G n %d k %d' % (t, ni, kil) + ' r {:.1e}'.format(rt) + ' a {:.1e}'.format(at) + ' ia %G' % (s.ia/g) + ' ic %G' % (s.ic/g))
        s.nit += ni
        s.kit += kil
        timer.stop()
        return not math.isnan(at)

    def time_step(s, t):
        timer = Timer('Time step')
        is_performed = s.newton(t)
        timer.stop()
        return is_performed

    def simulate(s):
        super(SubdomainProblem, s).simulate()
        info('Newton iterations %G, per time step %G'   % (s.nit, s.nit*dt/s.t))
        info('Krylov iterations %G, per newton step %G' % (s.kit, s.kit/s.nit))

if __name__ == "__main__":
    SubdomainProblem("subdomains2").simulate()
