import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Mesh parameters
Nx, Ny = 20, 20
x_m, x_M = 1., 3.
y_m, y_M = 1., 3.
np.random.seed(28)
# Create a uniform meshgrid
x = np.linspace(x_m, x_M, Nx+2)
y = np.linspace(y_m, y_M, Ny+2)
x = np.random.uniform(x_m,x_M,Nx+2)
y = np.random.uniform(y_m,y_M,Ny+2)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
triangulation = tri.Triangulation(X, Y)

# Mesh properties
NTri = np.shape(triangulation.triangles)[0]
NSom = np.shape(triangulation.x)[0]

# Initialize global matrices
M = np.zeros((NSom, NSom))
R = np.zeros((NSom, NSom))
F = np.zeros(NSom)  # Source term

for i in range(NTri):
    nodes = triangulation.triangles[i]
    pts = np.array([[X[n], Y[n]] for n in nodes])
    a = 0.5 * np.linalg.det(np.array([[1, pts[0,0], pts[0,1]], [1, pts[1,0], pts[1,1]], [1, pts[2,0], pts[2,1]]]))

    # Gradients for stiffness matrix calculation
    B = np.array([[pts[1, 1] - pts[2, 1], pts[2, 1] - pts[0, 1], pts[0, 1] - pts[1, 1]],
                  [pts[2, 0] - pts[1, 0], pts[0, 0] - pts[2, 0], pts[1, 0] - pts[0, 0]]])
    C = np.dot(B.T, B) / (4*a)
    
    # Local mass matrix
    Me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * (a/12)
    
    # Assemble global matrices
    for k in range(3):
        F[nodes[k]] += a/3  # Assuming f=1 over the entire domain
        for l in range(3):
            M[nodes[k], nodes[l]] += Me[k, l]
            R[nodes[k], nodes[l]] += C[k, l]

# Since we have Neumann boundary conditions, no modifications for boundary nodes are needed
u = npl.solve(R + M, F)


# Plotting the solution
plt.figure(figsize=(10, 8))
plt.gca().set_aspect('equal')
plt.tripcolor(triangulation.x, triangulation.y, triangulation.triangles, u, shading='flat')
plt.colorbar(label='Solution $u_h$')
plt.title('Approximate solution by $P_1$-FE')
plt.show()