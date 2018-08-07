from dolfin import *
import math
import matplotlib.pyplot as plt

# domains: 1 - anode, 2 - cathode, 3 - electrolyte
# interior: 1 - anode facets, 2 - cathode facets, 3 - electrolyte facets, 4 - anode interface, 5 - cathode interface
# exterior: 6 - anode collector, 7 - cathode collector, 8 - boundary

def plate(S=50.0, N=10, A=0.4, C=0.6):
    def anode(x):
        return between(x[0], (0, A*S))
    def cathode(x):
        return between(x[0], (C*S, S))
    def boundary(x, on_bound):
        return on_bound and (near(x[1], 0) or near(x[1], S))

    timer = Timer('Plate generation')
    mesh = RectangleMesh(Point(0, 0), Point(S, S), N, N)
    mesh.init()

    domains = MeshFunction("size_t", mesh, 2, 3)
    domains.anode = AutoSubDomain(anode)
    domains.anode.mark(domains, 1)
    domains.cathode = AutoSubDomain(cathode)
    domains.cathode.mark(domains, 2)
    
    bounds = bounds_by_domains(domains)
    mesh.boundary = AutoSubDomain(boundary)
    mesh.boundary.mark(bounds, 8)

    timer.stop()
    return mesh, domains, bounds

def rectangle(S=50.0, N=10, A=0.4, C=0.6, H=0.6, AR=4.):
    W = AR*S
    def anode(x):
        return between(x[0], (0, A*W)) and between(x[1], ((0.5-H/2)*S, (0.5+H/2)*S))
    def cathode(x):
        return between(x[0], (C*W, W)) and between(x[1], ((0.5-H/2)*S, (0.5+H/2)*S))

    timer = Timer('Rectangle generation')
    mesh = RectangleMesh(Point(0, 0), Point(W, S), int(AR*N), N)
    mesh.init()

    domains = MeshFunction("size_t", mesh, 2, 3)
    domains.anode = AutoSubDomain(anode)
    domains.anode.mark(domains, 1)
    domains.cathode = AutoSubDomain(cathode)
    domains.cathode.mark(domains, 2)
    
    bounds = bounds_by_domains(domains)

    timer.stop()
    return mesh, domains, bounds

def cross(S=50.0, N=10, A=0.4, C=0.6, H=0.4, AR=10.):
    W = AR*S
    def anode(x):
        if not between(x[0], (0, A*W)):
            return False
        r = H/2
        if not between(x[1], ((0.5-r)*S, (0.5+r)*S)):
            return False
        r = H/4 
        if between(x[1], ((0.5-r)*S, (0.5+r)*S)):
            return True
        for i in range(int(round(2*AR*A/H))):
            if between(x[0], (2*i*H*S-DOLFIN_EPS_LARGE, 2*(i+0.5)*H*S+DOLFIN_EPS_LARGE)):
                return True
        return False
    def cathode(x):
        if not between(x[0], (C*W, W)) :
            return False
        r = H/2
        if not between(x[1], ((0.5-r)*S, (0.5+r)*S)):
            return False
        r = H/4 
        if between(x[1], ((0.5-r)*S, (0.5+r)*S)):
            return True
        for i in range(int(round(2*AR*(1-C)/H))):
            if between(x[0], (W-2*(i+0.5)*H*S-DOLFIN_EPS_LARGE, W-2*i*H*S+DOLFIN_EPS_LARGE)):
                return True
        return False

    timer = Timer('Rectangle generation')
    mesh = RectangleMesh(Point(0, 0), Point(W, S), int(AR*N), N)
    mesh.init()

    domains = MeshFunction("size_t", mesh, 2, 3)
    domains.anode = AutoSubDomain(anode)
    domains.anode.mark(domains, 1)
    domains.cathode = AutoSubDomain(cathode)
    domains.cathode.mark(domains, 2)
    
    bounds = bounds_by_domains(domains)

    timer.stop()
    return mesh, domains, bounds


def parallel(S=50.0, N=10, A=0.9, C=0.1, H=0.4, AR=4.):
    W = AR*S
    def anode(x):
        return between(x[0], (0, A*W)) and between(x[1], ((0.75-H/2)*S, (0.75+H/2)*S))
    def cathode(x):
        return between(x[0], (C*W, W)) and between(x[1], ((0.25-H/2)*S, (0.25+H/2)*S))

    timer = Timer('Rectangle generation')
    mesh = RectangleMesh(Point(0, 0), Point(W, S), int(AR*N), N)
    mesh.init()

    domains = MeshFunction("size_t", mesh, 2, 3)
    domains.anode = AutoSubDomain(anode)
    domains.anode.mark(domains, 1)
    domains.cathode = AutoSubDomain(cathode)
    domains.cathode.mark(domains, 2)
    
    bounds = bounds_by_domains(domains)

    timer.stop()
    return mesh, domains, bounds


def paper(S=50.0, N=10, A=0.4, C=0.6):
    def anode(x):
        return between(x[0], (0, A*0.25*S)) or between(x[0], (A*0.25*S, A*0.75*S)) and between(x[1], (0.4*S, 0.6*S)) or between(x[0], (A*0.25*S, A*S)) and between(x[1], (0.2*S, 0.3*S)) or between(x[0], (A*0.25*S, A*S)) and between(x[1], (0.7*S, 0.8*S))
    def cathode(x):
        return between(x[0], ((1.0 - A*0.25)*S, S)) or between(x[0], ((1.0 - A*0.75)*S, (1.0 - A*0.25)*S)) and between(x[1], (0.1*S, 0.2*S)) or between(x[0], ((1.0 - A*0.75)*S, (1.0 - A*0.25)*S)) and between(x[1], (0.8*S, 0.9*S)) or between(x[0], ((1.0 - A)*S, (1.0 - A*0.25)*S)) and between(x[1], (0.4*S, 0.6*S))
    def boundary(x, on_bound):
        return on_bound and (near(x[1], 0) or near(x[1], S))

    timer = Timer('Square generation')
    mesh = RectangleMesh(Point(0, 0), Point(S, S), N, N)#, "crossed")
    mesh.init()

    domains = MeshFunction("size_t", mesh, 2, 3)
    domains.anode = AutoSubDomain(anode)
    domains.anode.mark(domains, 1)
    domains.cathode = AutoSubDomain(cathode)
    domains.cathode.mark(domains, 2)
    
    bounds = bounds_by_domains(domains)
    mesh.boundary = AutoSubDomain(boundary)
    mesh.boundary.mark(bounds, 8)
    timer.stop()
    return mesh, domains, bounds


def bounds_by_domains(domains):
    bounds = MeshFunction("size_t", domains.mesh(), domains.mesh().topology().dim() - 1, 0)
    for f in facets(domains.mesh()):
        cs = [int(i) for i in f.entities(2)]
        if len(cs) == 1:
            bounds[f] = domains[cs[0]] + 5
        elif domains[cs[0]] != domains[cs[1]]:
            bounds[f] = domains[cs[0]] + domains[cs[1]]
        else:
            bounds[f] = domains[cs[0]]
    return bounds


def adapt_bounds(submesh, mesh, bounds):
    timer = Timer('Adapt bounds')
    subbounds = MeshFunction("size_t", submesh, mesh.topology().dim() - 1, 0)
    parent_vertices = submesh.data().array("parent_vertex_indices", 0)
    for f in facets(submesh):
        vs = [parent_vertices[int(i)] for i in f.entities(0)]
        fs = [set(Vertex(mesh, v).entities(1)) for v in vs]
        pf = list(set.intersection(*fs))
        subbounds[f] = bounds[int(pf[0])]
    timer.stop()
    return subbounds


def split_mesh(mesh, domains, bounds, index):
    timer = Timer('Split mesh')
    submesh = SubMesh(mesh, domains, index)
    subbounds = adapt_bounds(submesh, mesh, bounds)
    timer.stop()
    return submesh, subbounds


if __name__ == "__main__":
    # mesh, domains, bounds = plate()
    # mesh, domains, bounds = rectangle()
    # mesh, domains, bounds = cross()
    # mesh, domains, bounds = parallel()
    mesh, domains, bounds = paper()
    plt.figure()
    plot(domains)
    submesh, subbounds = split_mesh(mesh, domains, bounds, 3)
    plot(submesh)
    plt.show()