from params import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os

class Local(object):
    def __init__(s, name, index, N, dN, j, dj, c0, p0):
        timer = Timer('Init local')
        s.mesh, s.bounds = split_mesh(mesh, domains, bounds, index)
        s.dx = Measure('dx', domain=s.mesh)
        s.ds = Measure('ds', domain=s.mesh, subdomain_data=s.bounds, metadata={ 'quadrature_degree': 1})
        s.dt = Constant(dt)

        s.E = FiniteElement('CG', triangle, 1)
        s.V = FunctionSpace(s.mesh, s.E)
        s.W = FunctionSpace(s.mesh, s.E*s.E)
        s.dof = vertex_to_dof_map(s.W)
        (s.dc, s.dp) = TrialFunction(s.W)
        (s.v, s.q) = TestFunction(s.W)

        s.w = Function(s.W)
        (s.c, s.p) = s.w.split()
        s.j = Function(VectorFunctionSpace(s.mesh, 'DG', 0))
        s.j.exp = j(s.c, s.p)
        s.c.rename('c' + str(index), 'concentration')
        s.p.rename('p' + str(index), 'potential')
        s.j.rename('j' + str(index), 'current')
        s.path = 'cg/' + name + '/'
        s.c_file = XDMFFile(s.path + s.c.name() + '.xdmf')
        s.p_file = XDMFFile(s.path + s.p.name() + '.xdmf')
        s.j_file = XDMFFile(s.path + s.j.name() + '.xdmf')
        s.w_path = s.path + 'w/w' + str(index) + '%g.xml'
        s.c_ = Function(s.V)
        assign(s.c, project(Constant(c0), s.V))
        assign(s.p, project(Constant(p0), s.V))
        assign(s.c_, s.c)

        s.F = F*(s.c - s.c_)/s.dt*s.v*s.dx \
            - F*dot(N(s.c, s.p), grad(s.v))*s.dx \
            - dot(j(s.c, s.p), grad(s.q))*s.dx

        s.dF = [None]
        s.dF[0] = F*s.dc/s.dt*s.v*s.dx \
                - F*dot(dN(s.c, s.p, s.dc, s.dp), grad(s.v))*s.dx \
                - dot(dj(s.c, s.p, s.dc, s.dp), grad(s.q))*s.dx

        s.dw = Function(s.W)
        s.dbc = None
        timer.stop()

    def connect(s, index, i, didc, didp):
        timer = Timer('Connect problems')
        s.F += i*s.v*s.ds(index) \
             + i*s.q*s.ds(index)
        di = didc*s.dc*s.v*s.ds(index) + didp*s.dp*s.v*s.ds(index) \
           + didc*s.dc*s.q*s.ds(index) + didp*s.dp*s.q*s.ds(index)
        s.dF[0] += di
        s.dF.append(-di)
        timer.stop()

    def time_next(s, t):
        assign(s.c_, s.c)
        assign(s.j, project(s.j.exp, s.j.function_space()))

    def save(s, t):
        s.c_file.write(s.c, t)
        s.p_file.write(s.p, t)
        s.j_file.write(s.j, t)
        File(s.w_path % t) << s.w

    def norms(s):
        timer = Timer('Compute norms')
        dwn = norm(s.dw)
        wn = norm(s.w)
        timer.stop()
        return dwn, wn

class Problem(object):
    def __init__(s, name):
        timer = Timer('Init problem')
        name += suffix
        s.name = name
        s.path = 'cg/' + name + '/'
        if not os.path.exists(s.path):
            os.makedirs(s.path)

        s.a = Local(name, 1, Na, dNa, ja, dja, ca0, pa0)
        s.c = Local(name, 2, Nc, dNc, jc, djc, cc0, pc0)
        s.e = Local(name, 3, Ne, dNe, je, dje, ce0, pe0)

        s.a.connect(4, s.fa(ia), s.fa(diaca), s.fa(diapa))
        s.e.connect(4,-s.fa(ia),-s.fa(diace),-s.fa(diape))
        s.c.connect(5, s.fc(ic), s.fc(diccc), s.fc(dicpc))
        s.e.connect(5,-s.fc(ic),-s.fc(dicce),-s.fc(dicpe))

        s.ia = 1
        s.ic = -1
        s.g_file = open(s.path + 'g.txt', 'w')
        if dirichlet_pa:
            s.a.dbc = DirichletBC(s.a.W.sub(1), Constant(0.), s.a.bounds, 6)
        else:
            s.a.F -= Constant(g)*s.a.q*s.a.ds(6)
        if dirichlet_pc:
            s.c.dbc = DirichletBC(s.c.W.sub(1), Constant(0.), s.c.bounds, 7)
        else:
            s.c.F += Constant(g)*s.c.q*s.c.ds(7)

        s.U_file = open(s.path + 'U.txt', 'w')
        for f in SubsetIterator(bounds, 6):
            v = Vertex(mesh, f.entities(0)[0])
            s.bound_a = [v.point().x(), v.point().y()]
            break
        for f in SubsetIterator(bounds, 7):
            v = Vertex(mesh, f.entities(0)[0])
            s.bound_c = [v.point().x(), v.point().y()]
            break
        timer.stop()

    def time_step(s, t):
        pass

    def fa(s, func):
        return func(s.a.c, s.e.c, s.a.p, s.e.p)

    def fc(s, func):
        return func(s.c.c, s.e.c, s.c.p, s.e.p)

    def save(s, t):
        timer = Timer('Save')
        s.a.save(t) 
        s.e.save(t)
        s.c.save(t)
        pa = s.a.p(s.bound_a)
        pc = s.c.p(s.bound_c)
        s.g_file.write('%G %G %G\n' % (t, s.ia, s.ic))
        s.U_file.write('%G %G %G %G\n' % (t, pa, pc, pc - pa))
        timer.stop()

    def simulate(s):
        timer = Timer('Solve problem')
        t = 0.
        while True:
            s.save(t)
            t += dt
            if t > tf or not s.time_step(t):
                t -= dt
                break
            s.a.time_next(t)
            s.c.time_next(t)
            s.e.time_next(t)
            s.ia = assemble(s.fa(ia)*s.e.ds(4))/Gc/g
            s.ic = assemble(s.fc(ic)*s.e.ds(5))/Gc/g
        s.t = t
        timer.stop()
        info(timings(TimingClear_clear, [TimingType_user]).str(True))

    def to_csr(s, A):
        row, col, val = as_backend_type(A).data(False)
        return sp.csr_matrix((val,col,row))

    def solve(s, A, b):
        timer = Timer('Solving system')
        ki = 0
        if method == 'direct':
            y = spla.spsolve(A, b)
        else:
            def callback (xk):
                ki += 1
            M = None
            if precondition == 'ilu':
                ILU = spla.spilu(A, pt)
                M_x = lambda x: ILU.solve(x)
                M = spla.LinearOperator(ILU.shape, M_x)
            (y, code)  = getattr(spla, method)(A, b, tol=mt, M=M, callback=callback)
        timer.stop()
        return ki, y


