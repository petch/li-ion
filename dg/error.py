from params import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FinerCode = '''
#include <dolfin/geometry/BoundingBoxTree.h>
namespace dolfin {
    class Finer : public Expression {
    public:
        Finer() : Expression(2) { }
        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
            const Cell cell_fine(*mesh_fine, c.index);
            const Point point = cell_fine.midpoint();
            unsigned int id = mesh->bounding_box_tree()->compute_first_entity_collision(point);
            const Cell cell(*mesh, id);
            ufc::cell ufc_cell;
            cell.get_cell_data(ufc_cell);
            f->eval(values, x, cell, ufc_cell);
        }
        std::shared_ptr<const Function > f;
        std::shared_ptr<const Mesh > mesh;
        std::shared_ptr<const Mesh > mesh_fine;
    };
}'''
def Finer(f, domain):
    m = Expression(cppcode=FinerCode, degree=1, domain=domain)
    m.f = f
    m.mesh = f.function_space().mesh()
    m.mesh_fine = domain
    return m

CoarserCode = '''
#include <dolfin/geometry/BoundingBoxTree.h>
namespace dolfin {
    class Coarser : public Expression {
    public:
        Coarser() : Expression(2) { }
        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
            const Cell cell_coarse(*mesh_coarse, c.index);
            const Point point = cell_coarse.midpoint();
            double w = 0.01;
            double x0 = (x[0] + w*point.x())/(1+w); 
            double x1 = (x[1] + w*point.y())/(1+w); 
            Point p(x0, x1);
            unsigned int id = mesh->bounding_box_tree()->compute_first_entity_collision(p);
            const Cell cell(*mesh, id);
            ufc::cell ufc_cell;
            cell.get_cell_data(ufc_cell);
            f->eval(values, x, cell, ufc_cell);
        }
        std::shared_ptr<const Function > f;
        std::shared_ptr<const Mesh > mesh;
        std::shared_ptr<const Mesh > mesh_coarse;
    };
}'''
def Coarser(f, domain):
    m = Expression(cppcode=CoarserCode, degree=1, domain=domain)
    m.f = f
    m.mesh = f.function_space().mesh()
    m.mesh_coarse = domain
    return m


def space_step(dt, N=4):
    E = FiniteElement("DG", triangle, 1)
    meshes = {}
    Ws = {}
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(N):
        meshes[i], domains, bounds = square(S, 10*2**i, 0.4, 0.6)
        Ws[i] = FunctionSpace(meshes[i], E*E)
        paths[i] = "results/dg_h/N%d_dt%g/" % (meshes[i].num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_h.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_h.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(N):
        es[i] = Function(Ws[i])
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(Ws[N-1], paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(N-1):
            w = Function(Ws[i], paths[i] % t)
            assign(es[i], project(Coarser(wf, meshes[i])-w, Ws[i]))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(N-1):
        plt.figure(1)
        plt.semilogy(ts, epL2[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(2)
        plt.semilogy(ts, ecL2[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(3)
        plt.semilogy(ts, epH1[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(4)
        plt.semilogy(ts, ecH1[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
    plt.figure(1)
    plt.gcf().canvas.set_window_title("Potential relative error norm L2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/pren_h_l2.pdf")
    plt.figure(2)
    plt.gcf().canvas.set_window_title("Concentration relative error norm L2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/cren_h_l2.pdf")
    plt.figure(3)
    plt.gcf().canvas.set_window_title("Potential relative error norm H1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/pren_h_h1.pdf")
    plt.figure(4)
    plt.gcf().canvas.set_window_title("Concentration relative error norm H1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/cren_h_h1.pdf")

def space_step_n(dt, N=4):
    E = FiniteElement("DG", triangle, 1)
    meshes = {}
    Ws = {}
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(N):
        meshes[i], domains, bounds = square(S, 10*2**i, 0.4, 0.6)
        Ws[i] = FunctionSpace(meshes[i], E*E)
        paths[i] = "results/dg_n/N%d_dt%g/" % (meshes[i].num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_h.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_h.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(N):
        es[i] = Function(Ws[N-1])
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(Ws[N-1], paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(N-1):
            w = Function(Ws[i], paths[i] % t)
            assign(es[i], project(wf-Finer(w, meshes[N-1]), Ws[N-1]))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(N-1):
        plt.figure(1)
        plt.semilogy(ts, epL2[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(2)
        plt.semilogy(ts, ecL2[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(3)
        plt.semilogy(ts, epH1[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
        plt.figure(4)
        plt.semilogy(ts, ecH1[i], label="$h=%g \\,\\mu m$" % (S/(10*2**i)*1e4))
    plt.figure(1)
    plt.gcf().canvas.set_window_title("Potential relative error norm L2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_pren_h_l2.pdf")
    plt.figure(2)
    plt.gcf().canvas.set_window_title("Concentration relative error norm L2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_cren_h_l2.pdf")
    plt.figure(3)
    plt.gcf().canvas.set_window_title("Potential relative error norm H1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_pren_h_h1.pdf")
    plt.figure(4)
    plt.gcf().canvas.set_window_title("Concentration relative error norm H1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_cren_h_h1.pdf")

def space_step_pc(dt, N=4):
    E = FiniteElement("DG", triangle, 1)
    meshes = {}
    Ws = {}
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(N):
        meshes[i], domains, bounds = square(S, 10*2**i, 0.4, 0.6)
        Ws[i] = FunctionSpace(meshes[i], E*E)
        paths[i] = "results/dg_d/N%d_dt%g/" % (meshes[i].num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_h.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_h.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(N):
        es[i] = Function(Ws[N-1])
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(Ws[N-1], paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(N-1):
            w = Function(Ws[i], paths[i] % t)
            assign(es[i], project(wf-Finer(w, meshes[N-1]), Ws[N-1]))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(N-1):
        if (i == 0):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'red')
        if (i == 1):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'red')
        if (i == 2):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), color = 'red')
    plt.figure(1)
    plt.gcf().canvas.set_window_title("Relative error norm L2")
    plt.legend(ncol=2)
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/ren_h_l2.eps")
    #plt.figure(2)
    #plt.gcf().canvas.set_window_title("Relative error norm H1")
    #plt.legend()
    #plt.ylim(10**(-8), 10**0)
    #plt.grid(True)
    #plt.savefig("results/dg/ren_h_h1.pdf")

def space_step_pc_n(dt, N=4):
    E = FiniteElement("DG", triangle, 1)
    meshes = {}
    Ws = {}
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(N):
        meshes[i], domains, bounds = square(S, 10*2**i, 0.4, 0.6)
        Ws[i] = FunctionSpace(meshes[i], E*E)
        paths[i] = "results/dg_n/N%d_dt%g/" % (meshes[i].num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_h.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_h.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(N):
        es[i] = Function(Ws[N-1])
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(Ws[N-1], paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(N-1):
            w = Function(Ws[i], paths[i] % t)
            assign(es[i], project(wf-Finer(w, meshes[N-1]), Ws[N-1]))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(N-1):
        if (i == 0):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), linestyle = 'dotted', color = 'red')
        if (i == 1):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), dashes=[5, 5], color = 'red')
        if (i == 2):
            plt.figure(1)
            plt.semilogy(ts, epL2[i], label="$\\varepsilon_{\\varphi}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), color = 'blue')
            plt.semilogy(ts, ecL2[i], label="$\\varepsilon_{c}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), color = 'red')
            #plt.figure(2)
            #plt.semilogy(ts, epH1[i], label="$\\varepsilon_{\\varphi}, h=%g \\,\\mu m$" % (S/(10*2**i)*1e4), color = 'blue')
            #plt.semilogy(ts, ecH1[i], label="$\\varepsilon_{c}, h=%g\\,\\mu m$" % (S/(10*2**i)*1e4), color = 'red')
    plt.figure(1)
    plt.gcf().canvas.set_window_title("Relative error norm L2")
    plt.legend(ncol=2)
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_ren_h_l2.eps")
    #plt.figure(2)
    #plt.gcf().canvas.set_window_title("Relative error norm H1")
    #plt.legend()
    #plt.ylim(10**(-8), 10**0)
    #plt.grid(True)
    #plt.savefig("results/dg/n_ren_h_h1.pdf")

def time_step(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg_d/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        plt.figure(5)
        plt.semilogy(ts[i], epL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(6)
        plt.semilogy(ts[i], ecL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(7)
        plt.semilogy(ts[i], epH1[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(8)
        plt.semilogy(ts[i], ecH1[i], label="$\\tau=%g \\,s$" % dts[i])
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Potential relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/pren_dt_l2.pdf")
    plt.figure(6)
    plt.gcf().canvas.set_window_title("Concentration relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/cren_dt_l2.pdf")
    plt.figure(7)
    plt.gcf().canvas.set_window_title("Potential relative error norm h1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/pren_dt_h1.pdf")
    plt.figure(8)
    plt.gcf().canvas.set_window_title("Concentration relative error norm h1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/cren_dt_h1.pdf")

def time_step_n(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg_n/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        plt.figure(5)
        plt.semilogy(ts[i], epL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(6)
        plt.semilogy(ts[i], ecL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(7)
        plt.semilogy(ts[i], epH1[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(8)
        plt.semilogy(ts[i], ecH1[i], label="$\\tau=%g \\,s$" % dts[i])
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Potential relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_pren_dt_l2.pdf")
    plt.figure(6)
    plt.gcf().canvas.set_window_title("Concentration relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_cren_dt_l2.pdf")
    plt.figure(7)
    plt.gcf().canvas.set_window_title("Poncentration relative error norm h1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_pren_dt_h1.pdf")
    plt.figure(8)
    plt.gcf().canvas.set_window_title("Concentration relative error norm h1")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_cren_dt_h1.pdf")

def time_step_pc(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        if (i == 0):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
            #plt.figure(7)
            #plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            #plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
        if (i == 1):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
            #plt.figure(7)
            #plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            #plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
        if (i == 2):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
            #plt.figure(7)
            #plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            #plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/ren_dt_l2.eps")
    #plt.figure(7)
    #plt.gcf().canvas.set_window_title("Relative error norm h1")
    #plt.legend()
    #plt.ylim(10**(-8), 10**0)
    #plt.grid(True)
    #plt.savefig("results/dg/ren_dt_h1.pdf")

def time_step_pc_n(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg_n/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        if (i == 0):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
        if (i == 1):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
        if (i == 2):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/n_ren_dt_l2.eps")
    #plt.figure(7)
    #plt.gcf().canvas.set_window_title("Relative error norm h1")
    #plt.legend()
    #plt.ylim(10**(-8), 10**0)
    #plt.grid(True)
    #plt.savefig("results/dg/n_ren_dt_h1.pdf")

def time_step_subdomain(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg_subdomain/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"
    paths[len(dts)-1] = paths[len(dts)-1].replace("_subdomain", "")

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        if (i == 0):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], linestyle = 'dotted', color = 'red')
        if (i == 1):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], dashes=[5, 5], color = 'red')
        if (i == 2):
            plt.figure(5)
            plt.semilogy(ts[i], epL2[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            plt.semilogy(ts[i], ecL2[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
            plt.figure(7)
            plt.semilogy(ts[i], epH1[i], label="$\\varepsilon_{\\varphi},\\tau=%g \\,s$" % dts[i], color = 'blue')
            plt.semilogy(ts[i], ecH1[i], label="$\\varepsilon_{c},\\tau=%g \\,s$" % dts[i], color = 'red')
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Potential relative error norm l2")
    plt.legend()
    plt.ylim(10**(-8), 10**0)
    plt.grid(True)
    plt.savefig("results/dg/pren_dt_subdomain_l2.eps")
    plt.figure(7)
    #plt.gcf().canvas.set_window_title("Potential relative error norm h1")
    #plt.legend()
    #plt.ylim(10**(-8), 10**0)
    #plt.grid(True)
    #plt.savefig("results/dg/pren_dt_subdomain_h1.pdf")


def time_step_subdomain2(N, dts):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(len(dts)):
        paths[i] = "results/dg_subdomain2/N%d_dt%g/" % (mesh.num_vertices(), dts[i])
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"
    paths[len(dts)-1] = paths[len(dts)-1].replace("_subdomain2", "")

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    ts = {}
    for i in range(len(dts)):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []
        ts[i] = []

    t = 0.
    while t < tf:
        t += dts[-1]
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[N-1] % t)
        pf, cf = wf.split()
        for i in range(len(dts)-1):
            if t % dts[i] > 0:
                continue
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            ts[i].append(t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], 'H1')/norm(pf, 'H1'))
            ecH1[i].append(norm(ecs[i], 'H1')/norm(cf, 'H1'))
    
    for i in range(len(dts)-1):
        plt.figure(5)
        plt.semilogy(ts[i], epL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(6)
        plt.semilogy(ts[i], ecL2[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(7)
        plt.semilogy(ts[i], epH1[i], label="$\\tau=%g \\,s$" % dts[i])
        plt.figure(8)
        plt.semilogy(ts[i], ecH1[i], label="$\\tau=%g \\,s$" % dts[i])
    plt.figure(5)
    plt.gcf().canvas.set_window_title("Potential relative error norm l2")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/dg/pren_dt_subdomain2_l2.pdf")
    plt.figure(6)
    plt.gcf().canvas.set_window_title("Concentration relative error norm l2")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/dg/cren_dt_subdomain2_l2.pdf")
    plt.figure(7)
    plt.gcf().canvas.set_window_title("Potential relative error norm_h1")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/dg/pren_dt_subdomain2_h1.pdf")
    plt.figure(8)
    plt.gcf().canvas.set_window_title("Concentration relative error norm_h1")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/dg/cren_dt_subdomain2_h1.pdf")

    
def diff_bc(N, dt):
    mesh, domains, bounds = square(S, 10*2**(N-1), 0.4, 0.6)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(1):
        paths[i] = "results/dg_d/N%d_dt%g/" % (mesh.num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"
    paths[1] = paths[0].replace("dg_d", "dg")

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(1):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[1] % t)
        pf, cf = wf.split()
        for i in range(1):
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], "H1")/norm(pf, "H1"))
            ecH1[i].append(norm(ecs[i], "H1")/norm(cf, "H1"))
    
    for i in range(1):
        plt.figure(3)
        plt.semilogy(ts, epL2[i], label="$\\varepsilon^\\varphi_{L_2}$")
        plt.semilogy(ts, ecL2[i], label="$\\varepsilon^c_{L_2}$")
        plt.semilogy(ts, epH1[i], label="$\\varepsilon^\\varphi_{H_1}$")
        plt.semilogy(ts, ecH1[i], label="$\\varepsilon^c_{H_1}$")
    plt.legend()
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("$\\varepsilon$")
    plt.savefig("results/ren_bc.pdf")

def diff_bc2(N, dt):
    mesh, domains, bounds = circleWith(S, 10*2**(N-1), 0.4, 0.6, 0.2, 0.8, 0.1, 0.15, 0.85, 0.1)
    E = FiniteElement("DG", triangle, 1)
    W = FunctionSpace(mesh, E*E)
    paths = {}
    files_ep = {}
    files_ec = {}
    for i in range(1):
        paths[i] = "results/dg_d/N%d_dt%g/" % (mesh.num_vertices(), dt)
        files_ep[i] = XDMFFile(paths[i] + "ep_dt.xdmf")
        files_ec[i] = XDMFFile(paths[i] + "ec_dt.xdmf")
        paths[i] += "w%g.xml"
    paths[1] = paths[0].replace("dg_d", "dg")

    es = {}
    eps = {}
    ecs = {}
    epL2 = {}
    ecL2 = {}
    epH1 = {}
    ecH1 = {}
    for i in range(1):
        es[i] = Function(W)
        eps[i], ecs[i] = es[i].split()
        epL2[i] = []
        ecL2[i] = []
        epH1[i] = []
        ecH1[i] = []

    t = 0.
    ts = []
    while t < tf:
        t += dt
        ts.append(t)
        info("time %g / %g" % (t, tf))
        wf = Function(W, paths[1] % t)
        pf, cf = wf.split()
        for i in range(1):
            w = Function(W, paths[i] % t)
            assign(es[i], project(wf-w, W))
            files_ep[i].write(eps[i], t)
            files_ec[i].write(ecs[i], t)
            epL2[i].append(norm(eps[i])/norm(pf))
            ecL2[i].append(norm(ecs[i])/norm(cf))
            epH1[i].append(norm(eps[i], "H1")/norm(pf, "H1"))
            ecH1[i].append(norm(ecs[i], "H1")/norm(cf, "H1"))
    
    for i in range(1):
        plt.figure(3)
        plt.semilogy(ts, epL2[i], label="$\\varepsilon^\\varphi_{L_2}$")
        plt.semilogy(ts, ecL2[i], label="$\\varepsilon^c_{L_2}$")
        plt.semilogy(ts, epH1[i], label="$\\varepsilon^\\varphi_{H_1}$")
        plt.semilogy(ts, ecH1[i], label="$\\varepsilon^c_{H_1}$")
    plt.legend()
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("$\\varepsilon$")
    plt.savefig("results/ren_bc.pdf")

#space_step(10, 4)
#space_step_n(10, 4)
#space_step_pc(10, 4)
#space_step_pc_n(10, 4)
#time_step(4, [80, 40, 20, 10])
#time_step_n(4, [80, 40, 20, 10])
#, dashes=[5, 5]

#time_step_pc(4, [80, 40, 20, 10])
time_step_pc_n(4, [80, 40, 20, 10])
#time_step_subdomain(4, [80, 40, 20, 10])

#time_step_subdomain2(4, [80, 40, 20, 10])\
#diff_bc2(2, 10)
plt.show()
