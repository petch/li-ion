from problem_base import *

class ConstantProblem(Problem):
    def __init__(s, name):
        super(ConstantProblem, s).__init__(name)
        s.ca, s.cc, s.pa, s.pe, s.pc = ca0, cc0, pa0, pe0, pc0

    def time_step(s, t):
        timer = Timer('Time step')
        s.ca = Ca(s.ca, dt, g, Gc, Vc)
        s.cc = Cc(s.cc, dt, g, Gc, Vc)
        if s.ca <= 0 or s.ca >= cam or s.cc <= 0 or s.cc >= ccm:
            return False
        s.pa = Pa(s.ca)
        s.pe = Pe(Ia, Gc, g, ce0, s.ca, s.pa)
        s.pc = Pc(Ic, Gc, g, s.cc, ce0, s.pe)
        assign(s.a.c, project(Constant(s.ca), s.a.V))
        assign(s.c.c, project(Constant(s.cc), s.c.V))
        assign(s.a.p, project(Constant(s.pa), s.a.V))
        assign(s.e.p, project(Constant(s.pe), s.e.V))
        assign(s.c.p, project(Constant(s.pc), s.c.V))
        info('t %G ca %G cc %G pa %G pe %G pc %G' % (t, s.ca, s.cc, s.pa, s.pe, s.pc))
        timer.stop()
        return True

if __name__ == "__main__":
    ConstantProblem("constant").simulate()
