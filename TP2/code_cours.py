import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import matplotlib.tri as tri

Nx=20
Ny=20

x_m=1.
x_M=3.
y_m=1.
y_M=3.

np.random.seed(28)

# Uniform meshgrid 
x=np.linspace(x_m,x_M,Nx+2)
y=np.linspace(y_m,y_M,Ny+2)
#x=np.random.uniform(x_m,x_M,Nx+2)
#y=np.random.uniform(y_m,y_M,Ny+2)

X,Y=np.meshgrid(x,y)

X=X.flatten()
Y=Y.flatten()

triang = tri.Triangulation(X, Y)

NTri=np.shape(triang.triangles)[0]
NSom=np.shape(triang.x)[0]

#Table with nodes coordinates
TabSom=np.zeros([NSom,2])
TabSom[:,0]=triang.x
TabSom[:,1]=triang.y

# Table with triangle nodes
TabTri=triang.triangles

# Représentation du maillage
plt.figure(1)
plt.gca().set_aspect('equal')
plt.triplot(X,Y,triang.triangles, 'b-', lw=0.5)
plt.title('maillage')
plt.show()


# Initialize Global Element Matrices : Mass Matrix and Stiffness matrix
M = np.zeros((NSom, NSom)) #Mass matrix
R = np.zeros((NSom, NSom)) #Stiffness matrix


#Initialize Right hand side
F = np.zeros(NSom)

# Define Local Element Matrices (placeholders for now)
Me_local = np.zeros((3, 3))
Re_local = np.zeros((3, 3))

# Function to compute area of a triangle given by vertices
def tri_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

# Define Local Mass and Stiffness Matrices for P1 element (placeholders for simplicity)
def local_matrices_p1(x1, y1, x2, y2, x3, y3):
    # Area of the triangle
    area = tri_area(x1, y1, x2, y2, x3, y3)

    # Local Mass Matrix for P1 element
    Me_local = (area / 12.0) * np.array([[2, 1, 1],
                                         [1, 2, 1],
                                         [1, 1, 2]])

    # Calculate gradients of shape functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Local Stiffness Matrix for P1 element
    Re_local = (1 / (4 * area)) * np.outer(b, b) + np.outer(c, c)

    return Me_local, Re_local

##### Algorithm to implement the mass matrix and the stifness matrix
for l in range(NTri):
    nodes = TabTri[l]
    # Get the vertices of the triangle
    x1, y1 = TabSom[nodes[0], 0], TabSom[nodes[0], 1]
    x2, y2 = TabSom[nodes[1], 0], TabSom[nodes[1], 1]
    x3, y3 = TabSom[nodes[2], 0], TabSom[nodes[2], 1]
    
    area = tri_area(x1, y1, x2, y2, x3, y3)
    
    # Calculate the local matrices for this triangle
    Me_local, Re_local = local_matrices_p1(x1, y1, x2, y2, x3, y3)

    # Assemble global matrices
    for i in range(3):  # Loop over local nodes
        I = nodes[i]  # Global index
        for j in range(3):
            J = nodes[j]  # Global index
            M[I, J] += Me_local[i, j]  # Assemble global mass matrix
            R[I, J] += Re_local[i, j]  # Assemble global stiffness matrix
        #F[I] += area / 3  # Assuming constant f=1, equally distributed to each node


##### Algorithm to implement the right hand side
# Calculate the right-hand side F for f=1
F = np.sum(M, axis=1)
# Calculate u_h
u = npl.solve(R+M,F)


plt.figure(2)
plt.gca().set_aspect('equal')
plt.tripcolor(triang.x,triang.y,triang.triangles, u, shading='flat')
plt.colorbar()
plt.title('solution approchee par EF P1')
plt.show()


# Define the true solution and its Laplacian
def u_true(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)

def f_true(x, y):
    return (2 * np.pi**2 + 1) * np.cos(np.pi * x) * np.cos(np.pi * y)

# Compute the true right-hand side f on each node
F_true = f_true(X, Y)

# Calculate the L2 norm error
u_exact = u_true(X, Y)

# Calculate the interpolation
I_h = u_true(TabSom[:, 0], TabSom[:, 1])

# Calculate u_h
u_h = npl.solve(R+M,F_true)

error_L2 = np.sqrt((I_h - u_h).T @ M @ (I_h - u_h))
error_L2_log = np.log(error_L2)

def grad_u_true(x, y):
    du_dx = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    du_dy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return np.array([du_dx, du_dy])

def grad_uh(triangles, TabSom, u_h):
    grad_uh = np.zeros((len(triangles), 2))  # Each row will hold the grad_uh for one triangle
    for i, triangle in enumerate(triangles):
        # Get the nodes making up this triangle
        nodes = triangle
        # Get the coordinates of the vertices
        P1, P2, P3 = TabSom[nodes[0]], TabSom[nodes[1]], TabSom[nodes[2]]
        # Create the matrix with rows (1, x, y) for each vertex
        A = np.array([
            [1, P1[0], P1[1]],
            [1, P2[0], P2[1]],
            [1, P3[0], P3[1]]
        ])
        # Create the vector of u_h values at the vertices
        b = np.array([u_h[nodes[0]], u_h[nodes[1]], u_h[nodes[2]]])
        # Solve for the coefficients of the local linear function
        coeffs = np.linalg.solve(A, b)
        # The coefficients related to x and y are the components of the gradient
        grad_uh[i] = coeffs[1:]  # Ignore the constant term
    return grad_uh

# Calculate the H1 semi-norm error using the stiffness matrix A
grad_uh = grad_uh(TabTri, TabSom, u)
grad_u_exact = grad_u_true(X, Y)
#error_H1 = npl.norm(grad_uh - grad_u_exact, 2) / npl.norm(grad_u_exact, 2)
#error_H1_log = np.log(error_H1)

def calculate_mesh_size(TabSom, TabTri):
    h_max = 0
    for triangle in TabTri:
        vertices = TabSom[triangle]
        
        # Calculate the distance between each pair of vertices
        for i in range(3):
            for j in range(i+1, 3):
                h = np.linalg.norm(vertices[i] - vertices[j])
                h_max = max(h_max, h)
                
    return h_max

# Now calculate the mesh size h for your mesh
h = calculate_mesh_size(TabSom, TabTri)
plt.figure()
plt.loglog(h, error_L2, 'o-', label='L2 norm error')
#plt.loglog(h, error_H1, 's-', label='H1 semi-norm error')
plt.xlabel('log(1/h)')
plt.ylabel('log(Error)')
plt.legend()
plt.title('Error analysis')
plt.show()
