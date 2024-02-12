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
    
    