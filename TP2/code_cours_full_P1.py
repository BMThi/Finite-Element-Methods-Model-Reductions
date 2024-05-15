import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import matplotlib.tri as tri
from scipy.integrate import dblquad

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
x=np.random.uniform(x_m,x_M,Nx+2)
y=np.random.uniform(y_m,y_M,Ny+2)

def mesh(x, y):    
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
    return X, Y, TabSom, TabTri, NSom, NTri, triang

X, Y, TabSom, TabTri, NSom, NTri, triang = mesh(x, y)

# Représentation du maillage
plt.figure(figsize=(10,8))
plt.gca().set_aspect('equal')
plt.triplot(X,Y,triang.triangles, 'b-', lw=0.5)
plt.title('Mesh')
plt.show()

# Function to compute area of a triangle given by vertices
def tri_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs((x1-x3)*(y2-y3) - (y1-y3)*(x2-x3))

# Define Local Mass and Stiffness Matrices for P1 element (placeholders for simplicity)
def local_matrices_p1(x1, y1, x2, y2, x3, y3):
    """
    Compute the local mass and stiffness matrices for a linear triangle element (P1).

    Parameters:
    x1, y1, x2, y2, x3, y3 : The coordinates of the vertices of the triangle.

    Returns:
    Me (array): Local mass matrix (3x3).
    Ae (array): Local stiffness matrix (3x3).
    """
    # Area of the triangle
    area = tri_area(x1, y1, x2, y2, x3, y3)

    # Local Mass Matrix for P1 element
    Me_local = (area / 12.0) * np.array([[2, 1, 1],
                                         [1, 2, 1],
                                         [1, 1, 2]])

    # Calculate gradients of shape functions
    b = np.array([[y2 - y3, y3 - y1, y1 - y2], 
                  [x3 - x2, x1 - x3, x2 - x1]])

    # Local Stiffness Matrix for P1 element
    Ae_local = (1 / (4 * area)) * (b.T @ b)

    return Me_local, Ae_local

def local_matrices_p2(x1, y1, x2, y2, x3, y3):
    """
    Compute the local mass and stiffness matrices for a quadratic triangle element (P2).

    Parameters:
    x1, y1, x2, y2, x3, y3 : The coordinates of the vertices of the triangle.

    Returns:
    Me (array): Local mass matrix (6x6).
    Ke (array): Local stiffness matrix (6x6).
    """

    # Midpoints
    x12, y12 = (x1 + x2) / 2, (y1 + y2) / 2
    x23, y23 = (x2 + x3) / 2, (y2 + y3) / 2
    x31, y31 = (x3 + x1) / 2, (y3 + y1) / 2

    # Nodes: vertices + midpoints
    nodes = np.array([
        [x1, y1], [x2, y2], [x3, y3],
        [x12, y12], [x23, y23], [x31, y31]
    ])

    # Area of the triangle
    area = tri_area(x1, y1, x2, y2, x3, y3)

    # Initialize matrices
    Me = np.zeros((6, 6))
    Ke = np.zeros((6, 6))

    # Numerical integration (Gaussian quadrature, etc.) needed here for accurate integration
    # This is a placeholder:
    for i in range(6):
        for j in range(6):
            Me[i, j] = area / (12 if i == j else 24)  # Simplified, typically needs actual integration
            # Compute stiffness matrix entries (based on derivatives of basis functions)
            # Placeholder for demonstration:
            Ke[i, j] = area * (i + j + 1) / 36  # Simplified, replace with actual derivatives

    return Me, Ke

def calculMA(NSom, NTri, TabSom, TabTri):
    # Initialize Global Element Matrices : Mass Matrix and Stiffness matrix
    M = np.zeros((NSom, NSom)) #Mass matrix
    A = np.zeros((NSom, NSom)) #Stiffness matrix
    
    ##### Algorithm to implement the mass matrix and the stifness matrix
    for l in range(NTri):
        nodes = TabTri[l]
        # Get the vertices of the triangle
        x1, y1 = TabSom[nodes[0]]
        x2, y2 = TabSom[nodes[1]]
        x3, y3 = TabSom[nodes[2]]
        
        # Calculate the local matrices for this triangle
        Me_local, Ae_local = local_matrices_p1(x1, y1, x2, y2, x3, y3)
    
        # Assemble global matrices
        for i in range(3):  # Loop over local nodes
            I = nodes[i]  # Global index
            for j in range(3):
                J = nodes[j]  # Global index
                M[I, J] += Me_local[i, j]  # Assemble global mass matrix
                A[I, J] += Ae_local[i, j]  # Assemble global stiffness matrix
    return M, A
 
M, A = calculMA(NSom, NTri, TabSom, TabTri)
# Calculate the right-hand side F for f=1
F = np.sum(M, axis=1)
# Calculate u_h
U = npl.solve(A+M,F)


plt.figure(figsize=(10, 8))
plt.gca().set_aspect('equal')
plt.tripcolor(triang.x, triang.y, triang.triangles, U, shading='flat')
plt.colorbar(label='Solution $u_h$')
plt.title('Approximate solution by $P_1$-FE')
plt.show()


# Define the true solution and its Laplacian
def u_true(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)

def f_true(x, y):
    return (2 * np.pi**2 + 1) * np.cos(np.pi * x) * np.cos(np.pi * y)

def grad_u_true(x, y):
    du_dx = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    du_dy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return np.array([du_dx, du_dy])

# Define the function and its derivatives
def grad_u_squared(x, y):
    return (np.pi * np.sin(np.pi * x) * np.cos(np.pi * y))**2 + (np.pi * np.cos(np.pi * x) * np.sin(np.pi * y))**2

def laplace_u_squared(x, y):
    return (2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y))**2

# Compute L2 norm
area = dblquad(lambda x, y: u_true(x, y)**2, 1, 3, lambda x: 1, lambda x: 3)[0]
L2_norm = np.sqrt(area)

# Compute H1 norm
grad_area = dblquad(grad_u_squared, 1, 3, lambda x: 1, lambda x: 3)[0]
H1_seminorm = np.sqrt(grad_area)
H1_norm = np.sqrt(area + grad_area)

# Compute H2 norm
laplace_area = dblquad(laplace_u_squared, 1, 3, lambda x: 1, lambda x: 3)[0]
H2_seminorm = np.sqrt(laplace_area)
H2_norm = np.sqrt(area + grad_area + laplace_area)


n = 10
h = np.zeros(n)
error_L2 = np.zeros(n)
error_H1 = np.zeros(n)

def calculate_mesh_size(TabSom, TabTri):
    h_max = 0
    for triangle in TabTri:
        vertices = TabSom[triangle]
        
        # Calculate the distance between each pair of vertices
        for i in range(3):
            for j in range(i+1, 3):
                h = npl.norm(vertices[i] - vertices[j])
                h_max = max(h_max, h)
                
    return h_max

for i in range(n):
    #np.random.seed(i+28)
    #x=np.random.uniform(x_m,x_M,Nx+2)
    #y=np.random.uniform(y_m,y_M,Ny+2)
    Nx, Ny = 10*(i+1),10*(i+1)
    x=np.linspace(x_m,x_M,Nx+2)
    y=np.linspace(y_m,y_M,Ny+2)
    X, Y, TabSom, TabTri, NSom, NTri, triang = mesh(x, y)
    M, A = calculMA(NSom, NTri, TabSom, TabTri)
    
    # Interpolate f at each node
    f_values = f_true(TabSom[:, 0], TabSom[:, 1])
    # Compute the RHS F using the mass matrix M
    F = M @ f_values
    # Calculate u_h
    u_h = npl.solve(A+M,F)
    
    # Calculate the interpolation
    I_h = u_true(TabSom[:, 0], TabSom[:, 1])
    
    # Calculate the L2 norm error
    error_L2[i] = np.sqrt((I_h - u_h).T @ M @ (I_h - u_h))
    
    # Calculate the H1 semi-norm error using the stiffness matrix A
    error_H1[i] = np.sqrt((I_h - u_h).T @ A @ (I_h - u_h))
    
    # Now calculate the mesh size h for our mesh
    h[i] = calculate_mesh_size(TabSom, TabTri)

#error_L2_log = np.log(error_L2)
plt.figure(figsize=(10,8))
plt.loglog(1/h, error_L2/L2_norm, 'o-', label='L2 norm error')
plt.loglog(1/h, error_H1/H1_seminorm, 's-', label='H1 semi-norm error')
plt.xlabel('log(1/h)')
plt.ylabel('log(Error)')
plt.legend()
plt.title('Error analysis')
plt.show()

