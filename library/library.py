import matplotlib.pyplot as plt
import numpy as np
import math


# fixed point method
def fixed_point_iteration(g, x0, tol, max_iterations):
    """
    Solves an equation using the fixed-point iteration method.
    
    Parameters:
        g (function): The function defining the fixed-point iteration.
        x0 (float): Initial guess for the solution.
        tol (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations allowed.
    
    Returns:
        float: Approximation of the solution.
    """
    x_prev = x0
    for i in range(max_iterations):
        x_next = g(x_prev)
        if abs(x_next - x_prev) < tol:
            return x_next
        x_prev = x_next
    raise ValueError("Fixed-point iteration did not converge within the maximum number of iterations.")














#------------------------------------------------------------------------------------------------------------
# MATRIX SOLVE METHODS:
def load_matrix(file):
    with open(file, 'r' ) as f:
        mat_M = [[int(num) for num in row.split(' ')] for row in f]
    return mat_M

def is_symmetric(matrix):
    if not matrix:
        return False
    
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    if num_rows != num_cols:
        return False

    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True
            
def transpose_matrix(matrix):
    if not matrix:
        return []

    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Create a new matrix to store the transpose
    transpose = [[0 for _ in range(num_rows)] for _ in range(num_cols)]

    # Fill the transpose matrix
    for i in range(num_rows):
        for j in range(num_cols):
            transpose[j][i] = matrix[i][j]

    return transpose

def diag_domin(matrix):
  for i in range(len(matrix)):
      index = i 
      max_value = matrix[i][i]
      for j in range(len(matrix)):
        if matrix[j][i]>max_value:
          index = j
      matrix[index],matrix[i]=matrix[i],matrix[index]
  return matrix
 
# Gauss Jordan elemination method solve function for a n row matrix
def gauss_jordan(A, b):
    num_equations = len(A)
    num_variables = len(A[0])
    
    # Augment the matrix with the constant vector
    augmented_matrix = [row + [bi] for row, bi in zip(A, b)]

    for i in range(num_equations):
        # Partial pivoting
        max_row_index = i
        for j in range(i + 1, num_equations):
            if abs(augmented_matrix[j][i]) > abs(augmented_matrix[max_row_index][i]):
                max_row_index = j
        augmented_matrix[i], augmented_matrix[max_row_index] = augmented_matrix[max_row_index], augmented_matrix[i]

        # Make the diagonal elements 1
        divisor = augmented_matrix[i][i]
        if divisor == 0:
            return None  # No unique solution
        augmented_matrix[i] = [element / divisor for element in augmented_matrix[i]]

        # Make the other elements in the column zero
        for j in range(num_equations):
            if j != i:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = [element_j - factor * element_i for element_i, element_j in zip(augmented_matrix[i], augmented_matrix[j])]

    # Extract the solution
    solution = [row[-1] for row in augmented_matrix]
    return solution



# LU decomposition solution method
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1

    for i in range(n):
        for j in range(i, n):
            sum_val = sum(L[i, k] * U[k, j] for k in range(i))
            U[i, j] = A[i, j] - sum_val

        for j in range(i + 1, n):
            sum_val = sum(L[j, k] * U[k, i] for k in range(i))
            L[j, i] = (A[j, i] - sum_val) / U[i, i]

    return L, U

def LU_decom_solve(A, b):
    L, U = lu_decomposition(A)
    n = len(A)
    
    # Solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Solve Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x




def gauss_siedel(matrix, constant, p):
    
    solution = [0 for i in range(len(matrix[0]))]
    row = len(matrix)
 
    for k in range(1000):
        n = 0
        for i in range(row):
    
            temp_sum1 = sum(matrix[i][j]*solution[j] for j in range(i))
            temp_sum2 = sum(matrix[i][j]*solution[j] for j in range(i+1,row))
            
            solved = (1/matrix[i][i])*(constant[i]-temp_sum1 - temp_sum2)
            if abs(solved-solution[i]) < p:
                n +=1
            solution[i] = solved
        if i == len(matrix):
            break
    return solution

def forback_substitution(mat_L, matrix, vec_B):

    
    x = [0] * len(matrix[0])
    y = [0] * len(mat_L[0])
    
     
    for i in range(len(mat_L)):
        sum = 0
        for j in range(i):
            sum += mat_L[i][j]*y[j]
        y[i] = vec_B[i] - sum

    
    for i in reversed(range(len(matrix))):
        sum = 0
        for j in reversed(range(i, len(matrix[0]))):
            sum += matrix[i][j]*x[j]
        x[i] = (y[i] - sum)/matrix[i][i]

    return x


def check_for_inverse(matrix):
    det = 1
    for i in range(len(matrix)):
        det = det*matrix[i][i]

    if det == 0:
        raise Warning("Inverse of the matrix does not exist")
        return False
    
    else:
        return True


def matrix_inverse(matrix):

    n = len(matrix)

    inverse_mat = [[0 for x in range(n)] for y in range(n)] 

    L,U = lu_decomposition(matrix)

    if check_for_inverse(U) == True:

        for i in range(n):


            B = [0 for x in range(n)]
            B[i] = 1

            column = forback_substitution(L, U, B)

            for j in range(n):
                inverse_mat[j][i] = round(column[j], 3)

        return inverse_mat

    else:
        return None


# cholesky decomposition
def cholesky(A):
    
    r = len(A)
    c = len(A[0])
    L = [[0.0 for x in range(0,c)] for y in range(0,r)]
    
    for i in range(r):
        for k in range(i+1):
            temp_sum = sum(L[i][j]*L[k][j] for j in range(k))
            
            if i == k:
                L[i][k] = round((A[i][i] - temp_sum)**0.5, 2)
                
            else:
                L[i][k] = round((1/L[k][k])*(A[i][k]-temp_sum), 2)
    
        
    return L

















# INTEGRATONS _________________________________________________________________________________________________________________________

# mid-point method
def mid_point_int(func, lower_lim, upper_lim, N):
    h = (upper_lim-lower_lim)/N
    int_value = 0
    for n in range(N):
        x_n = (2*lower_lim + (2*n+1)*h)/2
        int_value = int_value + h*func(x_n)
    sol = round(int_value,8)
    return sol

# trapezoidal method
def trapez_int(func, lower_limit, upper_limit, N):
    h = (upper_limit-lower_limit)/N
    int_value = 0
    x = lower_limit
    temp = 0
    for n in range(N):
        x_n = x+h
        int_value = int_value + h*(func(x_n)+func(x))/2
        x = x_n
    sol = round(int_value,8)
    return sol

# simpsons method
def simp_int(func, lower_lim, upper_lim, N):
    h = (upper_lim-lower_lim)/N
    x_n = lower_lim
    int_value = (h/3)*func(lower_lim)
    for n in range(N-1):
        x_n = x_n+h
        if n%2 == 0:
            int_value = int_value + 4*(h/3)*func(x_n)
        else:
            int_value = int_value + 2*(h/3)*func(x_n)
    int_value = int_value + (h/3)*func(upper_lim)
    sol = round(int_value,8)
    return sol

# Guassian quadrature method to evaluate the integration
def gaussian_quad(f, lowlim, uplimit):
    # Weights and nodes for 3-point Gaussian quadrature
    weights = np.array([5/9, 8/9, 5/9])
    nodes = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])

    # Change of interval from [-1, 1] to [lowlim, uplim]
    x = 0.5 * (uplimit - lowlim) * nodes + 0.5 * (lowlim + uplimit)
    w = 0.5 * (uplimit - lowlim) * weights

    # Evaluate the integrand and sum up
    r = np.sum(w * f(x))
    return r

# ceiling
def ceil(x):
    if (x/1.0).is_integer() != True:
        x = int(x)+1
    return x

# Calculating N from a given upper bound of error
def err_simpN(func,b,a,x,err):
    N = (((b-a)**5)*(func(x))/(180*err))**(1/4)
    N = ceil(N)
    return N

def err_midN(func,b,a,x,err):
    N = ((b-a)**3*(func(x))/(24*err))**0.5
    N = ceil(N)
    return N

def err_trapN(func,b,a,x,err):
    N = ((b-a)**3*(func(x))/(12*err))**0.5
    N = ceil(N)
    return N

# monte carlo
def monte_carlo(func,low_bnd,up_bnd,N):
    epsilon = rndt_lcg_seeded(N)
    x_i = [low_bnd + (up_bnd-low_bnd)*x for x in epsilon]
    f = []
    Fn = 0
    Fn_2 = 0
    
    for i in range(N):
        f.append(Fn)
        Fn += func(x_i[i])
        Fn_2 += func(x_i[i]**2)
        
    F_n = ((up_bnd-low_bnd)/N)*Fn
    sigma_2 = Fn_2/N - (Fn/N)**2
    sigma = (sigma_2)**(0.5)
    
    return F_n, sigma

#______________________________________________________________________________________________________________________________________
# Odinary differential Equations and Partial differential equation_____________________________________________________________________

def crank_nicolson_solver(L, T, Nx, Nt, alpha, initial_condition, boundary_conditions):
    # Discretization
    dx = L / (Nx - 1)
    dt = T / Nt

    x_values = np.linspace(0, L, Nx)
    t_values = np.linspace(0, T, Nt)

    # Initialize solution matrix
    u = np.zeros((Nx, Nt))

    # Set initial condition
    u[:, 0] = initial_condition(x_values)

    # Set boundary conditions
    u[0, :] = boundary_conditions['x_start'](t_values)
    u[-1, :] = boundary_conditions['x_end'](t_values)

    # Coefficients for the tridiagonal system
    a = -alpha / 2
    b = 1 + alpha
    c = -alpha / 2

    # Time-stepping loop
    for n in range(1, Nt):
        # Right-hand side of the system
        rhs = np.zeros(Nx)
        rhs[1:-1] = u[1:-1, n-1] + (alpha / 2) * (u[:-2, n-1] - 2 * u[1:-1, n-1] + u[2:, n-1])

        # Forward substitution (without external modules)
        for i in range(1, Nx):
            m = a / b
            b = b - m * c
            rhs[i] = rhs[i] - m * rhs[i-1]

        u[-1, n] = rhs[-1] / b

        for i in range(Nx-2, -1, -1):
            u[i, n] = (rhs[i] - c * u[i+1, n]) / b

    return x_values, t_values, u




def rungekutta_4(func,x0,y0,xn,h):
    Y = [y0]
    X = [x0]
    i = 0
    
    while X[i] < xn:
        
        k1 = h*func(X[i],Y[i])
        k2 = h*func(X[i]+h/2, Y[i]+k1/2)
        k3 = h*func(X[i]+h/2, Y[i]+k2/2)
        k4 = h*func(X[i]+h, Y[i]+k3)
        
        Y.append(Y[i]+(1/6)*(k1+2*k2+2*k3+k4))
        X.append(X[i]+h)
        i += 1
    return X, Y, i

# Coupled ODE 2D
def coupled_rungekutta_2D(x0, v0, t, func_dxdt, func_dvdt, dt, Tn): 
                           
    X = []
    V = []
    T = []
    
    while t < Tn:
        T.append(t)
        k1x = dt*func_dxdt(x0, v0, t)
        k1v = dt*func_dvdt(x0,v0,t)
        
        k2x = dt*func_dxdt(x0+(k1x/2), v0+(k1v/2), t+(dt/2))
        k2v = dt*func_dvdt(x0+(k1x/2), v0+(k1v/2), t+(dt/2))
        
        k3x = dt*func_dxdt(x0+(k2x/2), v0+(k2v/2), t+(dt/2))
        k3v = dt*func_dvdt(x0+(k2x/2), v0+(k2v/2), t+(dt/2))        
        
        k4x = dt*func_dxdt(x0+(k3x/2), v0+(k3v/2), t+(dt/2))
        k4v = dt*func_dvdt(x0+(k3x/2), v0+(k3v/2), t+(dt/2))        
        
        x0 += (k1x + 2*k2x + 2*k3x + k4x)/6
        v0 += (k1v + 2*k2v + 2*k3v + k4v)/6
        t += dt
        X.append(x0)
        V.append(v0)
        
    return X,V,T

# runge kutta shooting method
def RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z0, xf, h):                         
    x = [x0]
    y = [y0]
    z = [z0]
    N = int((xf-x0)/h)
    for i in range(N):        
        k1 = h * func_dydx(x[i], y[i], z[i])
        l1 = h * Func_d2ydx2(x[i], y[i], z[i])
        
        k2 = h * func_dydx(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        l2 = h * Func_d2ydx2(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        
        k3 = h * func_dydx(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        l3 = h * Func_d2ydx2(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        
        k4 = h * func_dydx(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * Func_d2ydx2(x[i] + h, y[i] + k3, z[i] + l3)
        
        x.append(x[i] + h)
        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        z.append(z[i] + (l1 + 2*l2 + 2*l3 + l4)/6)
    return x, y, z

def Lagrange_interpol(zeta_h, zeta_l, yh, yl, y):                                            # Lagrange interpolation
    zeta = zeta_l + (zeta_h - zeta_l) * (y - yl)/(yh - yl)
    return zeta

def RKshooting_method_solve(Func_d2ydx2, func_dydx, x0, y0, xf, yf, z1, z2, h, tol=1e-6):                 #Shooting method
    x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z1, xf, h)
    yn = y[-1]
    if abs(yn - yf) > tol:
        if yn < yf:
            zeta_l = z1
            yl = yn
            x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z2, xf, h)
            yn = y[-1]
            if yn > yf:
                zeta_h = z2
                yh = yn
                zeta = Lagrange_interpol(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, zeta, xf, h)
                return x, y
            else:
                print("Invalid bracketing.")
        elif yn > yf:
            zeta_h = z1
            yh = yn
            x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z2, xf, h)
            yn = y[-1]
            if yn < yf:
                zeta_l = z2
                yl = yn
                zeta = Lagrange_interpol(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, zeta, xf, h)
                return x, y
            else:
                print("Invalid bracketig.")
    else:
        return x, y


# heat equation
def Heat_Equation_solve(temperature_func, Lx, Nx, Lt, Nt):
    
    hx = Lx/Nx
    ht = Lt/Nt
    alpha = ht/(hx**2)
    V0 = [0 for i in range(Nx+1)]
    V1 = [0 for i in range(Nx+1)]
    i_list = []
    
    if alpha < 0.5:
        for i in range(Nx+1):
            V0[i] = (temperature_func(Nx, i))
            i_list.append(i)
            
    for j in range(0, 1000):
        for i in range(1, Nx):
            if i == 0:
                V1[i] = (1 - 2*alpha)*V0[i] + alpha*V0[i+1]
            elif i == Nx:
                V1[i] = alpha*V0[i-1] + (1 - 2*alpha)*V0[i]
            else:
                V1[i] = alpha*V0[i-1] + (1 - 2*alpha)*V0[i] + alpha*V0[i+1]
        for i in range(1, Nx):
            V0[i] = V1[i]
        if j == 0  or j == 5 or j == 10 or j == 50 or j == 100 or j == 500 or j == 1000:
            plt.plot(i_list, V0)
    plt.show()
    
    