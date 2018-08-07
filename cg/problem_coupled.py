from problem_base import *

class CoupledProblem(Problem):
    def __init__(s, name):
        super(CoupledProblem, s).__init__(name)
        timer = Timer('Init coupled problem')
        s.map12 = compute_vertex_map(s.a.mesh, s.e.mesh)
        s.map21 = compute_vertex_map(s.e.mesh, s.a.mesh)
        s.map23 = compute_vertex_map(s.e.mesh, s.c.mesh)
        s.map32 = compute_vertex_map(s.c.mesh, s.e.mesh)
        s.nit = s.ni = 0
        s.kit = s.ki = 0
        s.n_file = open(s.path + 'newton.txt', 'w')
        timer.stop()

    def bind(s, Ni, Nj, mapij, Aij, pi, pj):
        row, col, data = [], [], []
        Aptr = Aij.indptr
        Aind = Aij.indices
        Adata = Aij.data
        for i, j in mapij. items():
            r1 = pj.dof[2*j]
            r2 = pj.dof[2*j+1]
            size1 = Aptr[r1+1]-Aptr[r1]
            size2 = Aptr[r2+1]-Aptr[r2]
            row += [pi.dof[2*i]] * size1
            row += [pi.dof[2*i+1]] * size2
            col.append(Aind[Aptr[r1]:Aptr[r1+1]])
            col.append(Aind[Aptr[r2]:Aptr[r2+1]])
            data.append(Adata[Aptr[r1]:Aptr[r1+1]])
            data.append(Adata[Aptr[r2]:Aptr[r2+1]])
        col = np.concatenate(col)
        data = np.concatenate(data)
        return sp.csr_matrix((data, (row, col)), shape=(Ni, Nj))

    def assemble_problems(s):
        timer = Timer('Assemble matrices')
        A21 = assemble(s.a.dF[1])
        A11 = assemble(s.a.dF[0])
        A32 = assemble(s.e.dF[2])
        A12 = assemble(s.e.dF[1])
        A22 = assemble(s.e.dF[0])
        A23 = assemble(s.c.dF[1])
        A33 = assemble(s.c.dF[0])
        timer.stop()

        timer = Timer('Assemble rhs')
        b1 = assemble(-s.a.F)
        b2 = assemble(-s.e.F)
        b3 = assemble(-s.c.F)
        timer.stop()

        timer = Timer('Apply boundary conditions')
        if s.a.dbc is not None:
            s.a.dbc.apply(A11)
            s.a.dbc.apply(b1)
        if s.c.dbc is not None:
            s.c.dbc.apply(A33)
            s.c.dbc.apply(b3)
        timer.stop()

        timer = Timer('Convert to sparse matrices')
        J21 = s.to_csr(A21)
        J11 = s.to_csr(A11)
        J12 = s.to_csr(A12)
        J32 = s.to_csr(A32)
        J22 = s.to_csr(A22)
        J23 = s.to_csr(A23)
        J33 = s.to_csr(A33)
        timer.stop()

        timer = Timer('Binding matrices')
        N1 = J11.shape[0]
        N2 = J22.shape[0]
        N3 = J33.shape[0]
        J12 = s.bind(N1, N2, s.map12, J12, s.a, s.e)
        J21 = s.bind(N2, N1, s.map21, J21, s.e, s.a)
        J23 = s.bind(N2, N3, s.map23, J23, s.e, s.c)
        J32 = s.bind(N3, N2, s.map32, J32, s.c, s.e)
        timer.stop()

        timer = Timer('Building global matrix')
        format = 'csr'
        if precondition == 'ilu':
            format = 'csc'
        A = sp.bmat([[J11, J12, None], [J21, J22, J23], [None, J32, J33]], format)
        timer.stop()

        timer = Timer('Building global rhs')
        b = np.concatenate((b1, b2, b3))
        timer.stop()

        return A, b, N1, N2, N3

    def clamp(s, w, cmin, cmax):
        v = w.vector().get_local()
        c = v[::2]
        np.clip(c, cmin, cmax, c)
        v[::2] = c
        w.vector()[:] = v

    def newton_step(s):
        timer = Timer('Assemble problems')
        A, b, N1, N2, N3 = s.assemble_problems()
        timer.stop()

        ki, y = s.solve(A, b)

        timer = Timer('Vector to function')
        s.a.dw.vector()[:] = y[:N1]
        s.e.dw.vector()[:] = y[N1:N1+N2]
        s.c.dw.vector()[:] = y[N1+N2:]
        s.a.w.vector()[:] += s.a.dw.vector()[:]
        s.e.w.vector()[:] += s.e.dw.vector()[:]
        s.c.w.vector()[:] += s.c.dw.vector()[:]
        # EPS = 1e-7
        # s.clamp(s.a.w, EPS, cam - EPS)
        # s.clamp(s.e.w, EPS, 10*ce0)
        # s.clamp(s.c.w, EPS, ccm - EPS)
        timer.stop()

        dna, na = s.a.norms()
        dne, ne = s.e.norms()
        dnc, nc = s.c.norms()
        (dwn, wn) = ((dna**2 + dne**2 + dnc**2)**0.5, (na**2 + ne**2 + nc**2)**0.5)
        rn = dwn/wn
        return ki, rn, dwn

    def save(s, t):
        super(CoupledProblem, s).save(t)
        s.n_file.write('%G %d %d\n' % (t, s.ni, s.ki))

    def time_step(s, t):
        timer = Timer('Time step init')
        rt = 1 + rt_min
        at = 1 + at_min
        ni = kil = 0
        timer.stop()
        timer = Timer('Time step newton')
        while rt > rt_min and at > at_min and ni < ni_max:
            ni += 1
            ki, rt, at = s.newton_step()
            kil += ki
        info('t %G n %d k %d r %.1e a %.1e ia %G ic %G' % (t, ni, kil, rt, at, s.ia, s.ic))
        s.nit += ni
        s.ni = ni
        s.kit += kil
        s.ki = kil
        timer.stop()
        return not math.isnan(at)

    def simulate(s):
        super(CoupledProblem, s).simulate()
        info('Newton iterations %G, per time step %G'   % (s.nit, s.nit*dt/s.t))
        info('Krylov iterations %G, per newton step %G' % (s.kit, s.kit*1./s.nit))

if __name__ == "__main__":
    CoupledProblem("coupled").simulate()
