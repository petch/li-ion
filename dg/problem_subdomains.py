from problem_base import *

# Interior penalty method
def ipm(k, u, v, dx, dS, n, h_avg):
    return dot(k*grad(u), grad(v))*dx \
         - dot(k*jump(u, n), avg(grad(v)))*dS \
         - dot(avg(k*grad(u)), jump(v, n))*dS \
         + alpha/h_avg*k*jump(u)*jump(v)*dS

CombineCode = '''
namespace dolfin {
    class Combine : public Expression {
    public:
        Combine() : Expression(2) { }
        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
            std::size_t i = (*domains)[c.index];
            if (i == 1)
                wa->eval(values, x);
            else if (i == 2)
                wc->eval(values, x);
            else
                we->eval(values, x);
        }
        std::shared_ptr<const Function > wa;
        std::shared_ptr<const Function > wc;
        std::shared_ptr<const Function > we;
        std::shared_ptr<const MeshFunction<std::size_t> > domains;
    };
}'''
def Combine(domains, wa, wc, we):
    m = Expression(cppcode=CombineCode, degree=1)
    m.wa = wa
    m.wc = wc
    m.we = we
    m.domains = domains
    return m

def problem():
    E = FiniteElement("DG", triangle, 1)

    mesha, boundsa = split_mesh(mesh, domains, bounds, 1)
    dxa = Measure('dx', domain=mesha)
    dsa = Measure('ds', domain=mesha, subdomain_data=boundsa, metadata={ 'quadrature_degree': 1})
    dSa = Measure('dS', domain=mesha, subdomain_data=boundsa, metadata={ 'quadrature_degree': 1})
    na = FacetNormal(mesha)
    ha = CellDiameter(mesha)
    ha_avg = (ha('+') + ha('-'))/2.

    meshc, boundsc = split_mesh(mesh, domains, bounds, 2)
    dxc = Measure('dx', domain=meshc)
    dsc = Measure('ds', domain=meshc, subdomain_data=boundsc, metadata={ 'quadrature_degree': 1})
    dSc = Measure('dS', domain=meshc, subdomain_data=boundsc, metadata={ 'quadrature_degree': 1})
    nc = FacetNormal(meshc)
    hc = CellDiameter(meshc)
    hc_avg = (hc('+') + hc('-'))/2.

    meshe, boundse = split_mesh(mesh, domains, bounds, 3)
    dxe = Measure('dx', domain=meshe)
    dse = Measure('ds', domain=meshe, subdomain_data=boundse, metadata={ 'quadrature_degree': 1})
    dSe = Measure('dS', domain=meshe, subdomain_data=boundse, metadata={ 'quadrature_degree': 1})
    ne = FacetNormal(meshe)
    he = CellDiameter(meshe)
    he_avg = (he('+') + he('-'))/2.

    Wa = FunctionSpace(mesha, E*E)
    Wc = FunctionSpace(meshc, E*E)
    We = FunctionSpace(meshe, E*E)

    wa = Function(Wa)
    (pa, ca) = split(wa)
    wa_ = project(as_vector([pa0, ca0]), Wa)
    (pa_, ca_) = split(wa_)
    wa.assign(wa_)
    (qa, va) = TestFunction(Wa)

    wc = Function(Wc)
    (pc, cc) = split(wc)
    wc_ = project(as_vector([pc0, cc0]), Wc)
    (pc_, cc_) = split(wc_)
    wc.assign(wc_)
    (qc, vc) = TestFunction(Wc)

    we = Function(We)
    (pe, ce) = split(we)
    we_ = project(as_vector([pe0, ce0]), We)
    (pe_, ce_) = split(we_)
    we.assign(we_)
    (qe, ve) = TestFunction(We)

    w = Combine(domains, wa, wc, we)

    Fpa = ipm(ka, pa, qa, dxa, dSa, na, ha_avg) \
        + ia(ca, ce, pa, pe)*qa*dsa(4)
    if dirichlet_pa:
        Fpa += - dot(grad(qa), pa*na)*dsa(6) - dot(qa*na, grad(pa))*dsa(6) \
               + (gamma/ha)*pa*qa*dsa(6) + Constant(0)*dot(grad(qa), na)*dsa(6)
    else:
        Fpa += - g*qa*dsa(6)
    Fca = (ca-ca_)/dt*va*dxa \
        + ipm(Da, ca, va, dxa, dSa, na, ha_avg) \
        + ia(ca, ce, pa, pe)/F*va*dsa(4)

    Fpc = ipm(kc, pc, qc, dxc, dSc, nc, hc_avg) \
        + ic(cc, ce, pc, pe)*qc*dsc(5)
    if dirichlet_pc:
        Fpc += - dot(grad(qc), pc*nc)*dsc(7) - dot(qc*nc, grad(pc))*dsc(7) \
               + (gamma/hc)*pc*qc*dsc(7) + Constant(0)*dot(grad(qc), nc)*dsc(7)
    else:
        Fpc += + g*qc*dsc(7)
    Fcc = (cc-cc_)/dt*vc*dxc \
        + ipm(Dc, cc, vc, dxc, dSc, nc, hc_avg) \
        + ic(cc, ce, pc, pe)/F*vc*dsc(5)

    Fpe = ipm(ke, pec(ce, pe), qe, dxe, dSe, ne, he_avg) \
        - ia(ca, ce, pa, pe)*qe*dse(4) \
        - ic(cc, ce, pc, pe)*qe*dse(5)
    Fce = (ce-ce_)/dt*ve*dxe \
        + ipm(De, cep(ce, pe), ve, dxe, dSe, ne, he_avg) \
        - ia(ca, ce, pa, pe)/F*ve*dse(4) \
        - ic(cc, ce, pc, pe)/F*ve*dse(5)

    pa, ca = wa.split()
    pc, cc = wc.split()
    pe, ce = we.split()
    global t
    while t < tf:
        t += dt
        timer = Timer("Newton solver")
        (na, convergeda) = newton(Fpa + F/ce0*Fca == 0, wa)
        (nc, convergedc) = newton(Fpc + F/ce0*Fcc == 0, wc)
        (ne, convergede) = newton(Fpe + F/ce0*Fce == 0, we)
        timer.stop()
        if not convergeda or not convergedc or not convergede:
            return w
        timer = Timer("Next assign")
        wa_.assign(wa)
        wc_.assign(wc)
        we_.assign(we)
        timer.stop()
        timer = Timer("Potential update")
        pa_cur = pa(0, S/2)
        wa.assign(wa - project(Constant([pa_cur, 0]), Wa))
        wc.assign(wc - project(Constant([pa_cur, 0]), Wc))
        we.assign(we - project(Constant([pa_cur, 0]), We))
        timer.stop()
        iav = assemble(ia(ca, ce, pa, pe)*dse(4))/g/Gc
        icv = assemble(ic(cc, ce, pc, pe)*dse(5))/g/Gc
        info('t %G na %d nc %d ne %d ia %G ic %G' % (t, na, nc, ne, iav, icv))
        yield w

if __name__ == "__main__":
    global t
    path = "dg/subdomains" + suffix + "/"
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    w = Function(W)
    p, c = w.split()
    file_c = XDMFFile(path + "c.xdmf")
    file_p = XDMFFile(path + "p.xdmf")
    for wcombine in problem():
        assign(w, project(wcombine, W))
        timer = Timer("XML write")
        File(path + "w%g.xml" % t) << w
        timer.stop()
        timer = Timer("XDMF write")
        file_c.write(c, t)
        file_p.write(p, t)
        timer.stop()
    list_timings(TimingClear_clear, [TimingType_wall])
