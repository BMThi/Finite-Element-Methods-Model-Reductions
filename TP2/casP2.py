import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import matplotlib.tri as tri
import math
import fenics as fe
from fenics import *
from dolfin import *
from mshr import  Polygon, generate_mesh,Rectangle 

# Create mesh and function space

# Define the dimensions of the square
x_min, x_max = 1.0, 3.0
y_min, y_max = 1.0, 3.0
N =20 # Number of divisions along x and y directions
#nx,ny=20, 20

# Generate the mesh
domain = Rectangle(Point(x_min, y_min), Point(x_max, y_max))
mesh = generate_mesh(domain, N)
#mesh = RectangleMesh(Point(x_min, y_min), Point(x_max, y_max), nx, ny)

k = 2 ; print('Order of the Lagrange FE k = ', k)
V = FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k

fe.plot(mesh)
plt.title("Mesh of the domain")
plt.show()
plt.savefig('Mesh_of_the_domain.png')

#The right hand side
fp_exp = Expression('1.', degree=V.ufl_element().degree())
fp = interpolate(fp_exp,V)
# Trial & Test functions
u = TrialFunction(V); v = TestFunction(V)

#Weak formulation
F0 = dot(grad(u),grad(v))*dx+dot(u,v)*dx- fp * v * dx

# The bilinear and linear forms
a0 = lhs(F0); L0 = rhs(F0)

# Solve the linear system
u0 = Function(V)
solve(a0 == L0, u0)

# Plot the first solution u0 
plt.figure()
p=plot(u0, title='Approximate solution by $P_2$-FE')
p.set_cmap("rainbow"); plt.colorbar(p);
plt.show(block=True)
plt.savefig('u0.png')  # Save the plot to a file
