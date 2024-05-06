# # # # # # # # # # # #
#  MESH of a 2D house
# # # # # # # # # # # # 
import fenics as fe
from fenics import *
from dolfin import *
from mshr import Polygon, generate_mesh
import matplotlib.pyplot as plt

# Create empty Mesh
mesh = fe.Mesh()

# Create list of polygonal domain vertices
domain_vertices = [
    fe.Point(0.0, 0.0), # edge of house
    fe.Point(4.0, 0.0), # edge fireplace
    fe.Point(4.0, 1.0), # edge fireplace
    fe.Point(6.0, 1.0), # edge fireplace
    fe.Point(6.0, 0.0), # edge fireplace
    fe.Point(10.0, 0.0), # edge of house
    fe.Point(10.0, 1.0), # edge window
    fe.Point(10.0, 2.0), # edge window
    fe.Point(10.0, 5.0), # edge of house
    fe.Point(8.0, 5.0), # edge chimney
    fe.Point(8.0, 7.0), # edge chimney
    fe.Point(7.0, 7.0), # edge chimney
    fe.Point(7.0, 5.0), # edge chimney
    fe.Point(0.0, 5.0), # edge of house
]

domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 20)
fe.plot(mesh)
plt.title("Mesh of a simple Hut")
plt.show()

# Initialize mesh function for interior domains
domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())  # CellFunction
domains.set_all(0)

# Define new measures associated with the interior domains
dx = fe.Measure("dx", domain=mesh, subdomain_data=domains)


# Define the boundaries
wall_thick = 1e-7 # 0.5
roof_thick = 1e-7 # 0.8
chimney_thick = 1e-7 # 0.2
floor_thick = 1e-7 # 0.5
window_thick = 1e-7 # 0.06
fireplace_thick = 1e-7 # 0.10

class Wall(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            fe.near(x[0], 0)
            or (fe.near(x[0], 10) and x[1] <= 1)
            or (fe.near(x[0], 10) and x[1] >= 2)
        )

class Roof(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (fe.near(x[1], 5) and x[0] <= 7) or (fe.near(x[1], 5) and x[0] >= 8)
        )

class Chimney(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (fe.near(x[0], 7) and x[1] >= 5)
            or (fe.near(x[0], 8) and x[1] >= 5)
            or (fe.near(x[1], 7))
        )

class Floor(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0) and (x[0] <= 4 or x[0] >= 6)

class Window(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (fe.near(x[0], 10) and fe.between(x[1], (1, 3.5)))

class Fire(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (fe.near(x[1], 1) and fe.between(x[0], (4, 6)))

class Brick(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (fe.near(x[0], 4) and x[1] <= 1) or (fe.near(x[0], 6) and x[1] <= 1)

class Obstacle(fe.SubDomain):
    def inside(self, x, on_boundary):
        return fe.between(x[1], (0.5, 0.7)) and fe.between(x[0], (0.2, 1.0))


# create a cell function over the boundaries edges
sub_boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # mesh.topology().dim()-1
# set marker to 6
sub_boundaries.set_all(0)

wall = Wall()
wall.mark(sub_boundaries, 1)

roof = Roof()
roof.mark(sub_boundaries, 2)

chimney = Chimney()
chimney.mark(sub_boundaries, 3)

floor = Floor()
floor.mark(sub_boundaries, 4)

window = Window()
window.mark(sub_boundaries, 5)

fire = Fire()
fire.mark(sub_boundaries, 6)

brick = Brick()
brick.mark(sub_boundaries, 7)

# redefining integrals over boundaries
ds = fe.Measure('ds', domain=mesh, subdomain_data=sub_boundaries)

# Define new measures associated with the interior domains
dx = fe.Measure("dx", domain=mesh, subdomain_data=domains)

    
ETC



# # # # # # # # # # # 
# MULTIPLE NEUMAN BCs
# # # # # # # # # # #

# Create classes for defining parts of the boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol_bc)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0, tol_bc)
    
left = Left()
top = Top()

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)

ds = ds(subdomain_data = boundaries)
#ds(1) = Gamma_in ; ds(2) = Gamma_wall
    
a = inner(grad(u_n),grad(v)) * dx + c * u_n*v * ds(2) 

solve(a == F,u_n,bc0)
    
