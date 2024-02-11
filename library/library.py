import matplotlib.pyplot as plt
import numpy as np


def rndt_lcg(a,c,m,x,N):
    X=[]
    n=[]
    for i in range (N):
        x = ((a*x + c)%m)/m
        X.append(x)
        n.append(i)
    return X

def rndt_lcg_seeded(N):
    X=[]
    n=[]
    x = 10
    for i in range (N):
        x = ((1103515245*x + 12345)%32768)/32768
        X.append(x)
        n.append(i)
    return X

def sum_oddn(n):
    arr = []
    for i in range(0,n):
        arr.append(2*i + 1)
    print(sum(arr))

# factorial of n:
def fact_n(n):
    x = 1
    for i in range(1,n+1):
        x = x*i
    print(x)

def AP_sum(n,a,d):
    arr = [a]
    j=a
    for i in range(1,n):
        arr.append(j+d)
        j=j+d
    print(sum(arr))

def GP_sum(n,a,r):
    arr = [a]
    temp = a
    for i in range(1,n):
        arr.append(temp*r)
        temp = temp*r
    print(sum(arr))

def sum_ser(n):
    N = []
    arr = [1/2]
    temp = (1)/2
    for i in range(1,n):
        arr.append(((-1)/2)*temp)
        temp = temp*(-1)/2*temp
        s = sum(arr)
    for i in range(1,n+1):
        N.append(i)
    print('%.4f'%s)
    plt.plot(N,arr)
    plt.xlabel('n')
    plt.ylabel('Sum')

def sum_ser(n):
    N = []
    arr = [1/2]
    temp = (1)/2
    for i in range(1,n):
        arr.append(((-1)/2)*temp)
        temp = temp*(-1)/2*temp
        s = sum(arr)
    for i in range(1,n+1):
        N.append(i)
    print('%.4f'%s)
    plt.plot(N,arr)
    plt.xlabel('n')
    plt.ylabel('Sum')

def matrix_prod(A,B):
    
    
    
    #shapes of the input matrices
    r1 = len(A)
    c1 = len(A[0])
    r2 = len(B)
    c2 = len(B[0])
    result = [[0 for i in range(c2)] for j in range(r1)]
    
    #criteria:
    if c1 != r2:
        print("Input matrices doesn't satify dimension criteria of matrix multiplication")
    else:
        #defining an result matrix with zero values
        for i in range(0,r1):
            for j  in range(0,c2):
                for k in range(0,c1):
                    result[i][j] += A[i][k]*B[k][j]
    
    return result


def matr_prod(A,B):
    
    import numpy as np
    
    #shapes of the input matrices
    r1, c1 = A.shape
    r2, c2 = B.shape
    result = np.zeros(shape=(r1,c2))
    
    #criteria:
    if c1 != r2:
        print("Input matrices doesn't satify dimension criteria of matrix multiplication")
    else:
        #defining an result matrix with zero values
        for i in range(0,r1):
            for j  in range(0,c2):
                for k in range(0,c1):
                    result[i,j] += A[i,k]*B[k,j]
    
    return(result)

class mycomplex:
    
    def __init__(self, RealPart, ImagPart):
        self.real = RealPart
        self.imaginary = ImagPart
        
    def comp_sum(self, n1, n2):
        temp = mycomplex(0,0)
        temp.real = n1.real + n2.real
        temp.imaginary = n1.imaginary + n2.imaginary
        print(temp.real, '+ ',temp.imaginary,'i') 
    
    def comp_mult(self, n1, n2):
        temp = mycomplex(0,0)
        temp.real = n1.real*n2.real - n1.imaginary*n2.imaginary
        temp.imaginary = n1.real*n2.imaginary + n1.imaginary*n2.real
        print(temp.real, '+ ',temp.imaginary,'i')
        
    def comp_abs(self):
        return((self.real**2 + self.imaginary**2)**(1/2))





#------------------------------------------------------------------------------------------------------------
# MATRIX SOLVE METHODS:
def load_matrix(file):
    with open(file, 'r' ) as f:
        mat_M = [[int(num) for num in row.split(' ')] for row in f]
    return mat_M

def check_symmetric(A):
    A_t = transpose(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A_t[i][j] == A[i][j]:
                return True
            
def transpose(A):
    r = len(A)
    c = len(A[0])
    A_t = [[0.0 for x in range(0,c)] for y in range(0,r)]
    for i in range(len(A)):
        for j in range(len(A[0])):
            A_t[i][j] = A[j][i]
    return A_t

def diag_domin(matrix,const):
  for i in range(len(matrix)):
      index = i 
      max_value = matrix[i][i]
      for j in range(len(matrix)):
        if matrix[j][i]>max_value:
          index = j
      matrix[index],matrix[i]=matrix[i],matrix[index]
      const[index],const[i]=const[i],const[index]
  return matrix, const
 
# Gauss Jordan elemination method solve function for a n row matrix
def gj_solve(A):

    import numpy as np
    A = np.array(A)
    # defining a solution array
    solution = []
    
    #swapping 
    r,c = A.shape
    m = []
    for i in range(0,r):
        m.append(A[i,0])
        M = max(m)
    j = m.index(M)
    temp = A[0].copy()
    A[0] = A[j]
    A[j] = temp
    
    
    #Loop to convert the diagonal elements to 1 and other elements to zero   
    for i in range(0,r):
        
        # converting the diagonal element [i,i] to 1
        temp = A[i]/A[i,i].copy()
        A[i] = temp
        
        
        #reducing all other elements of ith column to zero
        for k in range(0,i):
            A[k] = A[k] - A[i]*A[k,i]
        for k in range(i+1,r):
            A[k] = A[k] - A[i]*A[k,i]
        
        
        # checking if any diagonal element is zero and thus swapping
        for l in range(1,r-1):
            if A[l,l] == 0:
                temp = A[l+1].copy()
                A[l+1] = A[l]
                A[l] = temp
    
    # appending the last column elements to solution array
    for i in range(0,r):
        solution.append(round(A[i,c-1],3))
        
    return solution



# LU decomposition solution method
def LU_decomp_solve(matrix,b):
    
    n  = len(matrix)  
    #convert the matrix to upper and lower triangular matrix
    for j in range(n):
        for i in range(n):
            if i <= j :
                    sum = 0
                    for k in range(i):
                        sum += matrix[i][k]*matrix[k][j]
                    matrix[i][j] = matrix[i][j] - sum
            else  :
                    sum = 0
                    for k in range(j):
                        sum += matrix[i][k]*matrix[k][j]
                    matrix[i][j] = (matrix[i][j] - sum)/matrix[j][j]       


    for i in range(n):
        sum = 0
        for j in range(i):
            sum += matrix[i][j]*b[j]
        b[i] = b [i] - sum       

    for i in range(n-1,-1,-1):
        sum = 0 
        for j in range(i+1,n):
            sum += matrix[i][j]*b[j]
        b[i] = (b[i] - sum)/(matrix[i][i])
    return b  



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
def LU_decomposition_(matrix):

    n = len(matrix)

    
    L = [[0 for x in range(n)] for y in range(n)] 
    U = [[0 for x in range(n)] for y in range(n)]

    
    for i in range(len(matrix)):

        
        L[i][i] = 1

        
        for k in range(i, len(matrix[0])):
            sum = 0
            for j in range(i): 
                sum += (L[i][j] * U[j][k])
  
            U[i][k] = matrix[i][k] - sum

        
        for k in range(i+1, len(matrix[0])):             
            sum = 0 
            for j in range(i): 
                sum += (L[k][j] * U[j][i]) 
            
            #The order of element is reversed from i,k to k,i
            L[k][i] = (matrix[k][i] - sum) / U[i][i] 
  
    return L,U

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

    L,U = LU_decomposition_(matrix)

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

#_____________________________________________________________________________________________________________________________________
# Roots of Non-Linear Equations_______________________________________________________________________________________________________



def poly_derivative(f,x):
	return (f(x+0.0001)-f(x - 0.0001))/(2*0.0001)

def poly_double_derivative(f,x):
    return (poly_derivative(f, x+0.0001) - poly_derivative(f, x-0.0001))/2*0.0001

def func_polynomial(coeffs):
    def func(x):
        n = len(coeffs)
        y = 0
        for i in range(n):
            y += coeffs[i]*x**(n-i-1)
        return y
    return func


# Bisection Method
# to check if a and b are opposite signs and the change them
def check_bracket(func, a, b, alpha):
    if func(a)*func(b) > 0:
        i = 0
        while func(a)*func(b) > 0:
            if func(a)<func(b):
                a = a - alpha*abs(b-a)
            else:
                b = b + alpha*abs(b-a)
            i = i+1
            
    return a, b

def bisect_solve1(func, a_0, b_0, precision):
    
    I = []
    val = []
    i = 0
    if a_0 > b_0:
        print('left element of the interval cannot be greater than right element')
        return False
    if func(a_0)*func(b_0) > 0:
        print('The intervals not appropriate')
        return False
    
    while abs(b_0-a_0)>precision:
        
        i = i+1
        I.append(i)
        c = (a_0+b_0)/2
        val.append(c)
        if func(a_0)*func(c) < 0:
            b_0 = c
        else:
            a_0 = c
    return c, i, val, I

# Regula falsi method
def regula_falsi(func, a_0, b_0, precision):
    
    I = []
    val = []
    
    i = 0
    if a_0 > b_0:
        print('left element of the interval cannot be greater than right element')
        return False
    if func(a_0)*func(b_0) > 0:
        a_0, b_0 = check_bracket(func, a_0, b_0, 1)

    c = b_0 - ((b_0-a_0)*func(b_0))/(func(b_0)-func(a_0))
    c_0 = 0
    while abs(func(c))>precision:
        
        i = i+1
        I.append(i)
        c = b_0-((b_0-a_0)*func(b_0))/(func(b_0)-func(a_0))
        val.append(c)
        if  func(a_0)*func(c)<0:
            b_0 = c
        elif func(b_0)*func(c)<0:
            a_0 = c
            
    return c, i, val, I

# Newton-Raphson method
def newton_raphson(func, dfunc, x_guess, precision):
    x = x_guess - (func(x_guess)/dfunc(x_guess))
    i = 0
    counter = 0
    while abs(x-x_guess) > precision:
        i += 1
        x_guess = x
        x = x_guess - func(x_guess)/dfunc(x_guess)
        counter = counter+1
        if counter == 40000:
            print("Warning:Iter Limit reached not converged")
            break
    return x, i

# LAguerre Method
def deflate_syndiv(coeffs, root):
    n = len(coeffs)
    
    newcoeffs = []
    newcoeffs.append(coeffs[0])
    for i in range(1, n):
        newcoeffs.append(newcoeffs[i-1]*root+coeffs[i])
    return newcoeffs[:-1]

def laguerre(coeffs, x_guess, prec, max_iteration):
    
    n = len(coeffs)
    
    pol = func_polynomial(coeffs)
    x = x_guess
    
    if pol(x) == 0:
        return x
    for i in range(max_iteration):
        
        x_guess = x
        
        G = poly_derivative(pol,x)/pol(x)
        H = G**2 - poly_double_derivative(pol,x)/pol(x)
        denominator1 = G+((n-1)*(n*H - G**2))**0.5
        denominator2 = G-((n-1)*(n*H - G**2))**0.5
        
        if abs(denominator2)>abs(denominator1):
            a = n/denominator2
        else:
            a = n/denominator1
        x = x - a
        
        if abs(x - x_guess) < prec:
            break
    return x

def laguerre_poly_solve(coeffs, x, prec):
    n = len(coeffs)
    sol_roots = []
    for i in range(n-1):
        
        root_value = laguerre(coeffs, x, prec, 200)
        sol_roots.append(root_value)
        
        coeffs = deflate_syndiv(coeffs, root_value)
    return sol_roots

#..... laguerre not working


# ____________________________________________________________________________________________________________________________________
# Interpolation and data fitting _________________________________________________________________________________________________________________________

# Interpolation
def lagr_interpol(x_data, y_data, x):
    y = 0
    n = len(x_data)
    for i in range(n):
        prod = 1
        for k in range(len(y_data)):
            if i!= k:
                prod = prod*((x-x_data[k])/(x_data[i]-x_data[k]))
        y = y+prod*y_data[i]
        
    return y

# Linear fitting by least square method
def f_linear(x,a,b):
    return a*x + b

# Ceiling function
def ceil(x):
    if (x/1.0).is_integer() != True:
        x = int(x)+1
    return x

def linear_fit(x_data, y_data):
    N = len(x_data)
    
    # (a_1+s_1)x + (a_2+s_2)
    parameter = []
    p_error = []
    
    # define S_x, S_y, S_xx, S_xy, and S
    S = N
    S_x = 0
    S_y = 0
    S_xx = 0
    S_xy = 0
    S_yy = 0
    
    for i in range(N):
        S_x += x_data[i]
        S_y += y_data[i]
        S_xx += x_data[i]**2
        S_xy += x_data[i]*y_data[i]
        S_yy += y_data[i]**2
    
    # defining delta
    delta = S*S_xx - S_x**2
    
    # parameters
    parameter.append((S_xx*S_y - S_x*S_xy)/delta)
    parameter.append((S_xy*S - S_x*S_y)/delta)
    
    # errors
    p_error.append((S_xx/delta)**(0.5))
    p_error.append((S/delta)**(0.5))
    
    # pearson coeffcient
    R = (S_xy)**2/(S_xx*S_xy)
    
    y = [f_linear(x_data[i],parameter[1],parameter[0]) for i in range(N)]
    plt.plot(x_data,y)
    plt.scatter(x_data,y_data)
    #print(f'y = ({round(parameter[1],4)}+{round(p_error[1],4)})x + ({round(parameter[0],4)}+{round(p_error[0],4)})')
    return parameter, p_error, R

# polynomial fitting
def sum_power(x_data, k_):
    sum_ = 0
    for i in range(len(x_data)):
        sum_ = sum_ + x_data[i]**k_
        
    return sum_

def lin_pol_fit(x_data, y_data, par): # Code not working
    n = len(x_data)
    matrix = [[0 for i in range(par)] for j in range(par)]
    Y = [0 for k in range (par)]
    
    for k in range(par):
        sum_ = 0
        for i in range(n):
            sum_ = sum_ + (x_data[i]**k)*(y_data[i])
        Y[k] = sum_
    
    matrix[0][0] = n
    for r in range(1,par):
        for c in range(1,par):
            matrix[r][c] = sum_power(x_data,(r+c))
            
    sol = LU_decomp_solve(matrix,Y)
    return sol

def polYnomial_fit_solve(x,y,a):
    
    n=len(x)
    A=[[0 for i in range(a)]
          for j in range(a)]
    X=[0 for i in range(a)]
    
    for i in range(a):
        for j in range(a):
            if i==0 and j==0:
                A[0][0]=n
            else:
                sum=0
                for k in range(n):
                    sum+=x[k]**(i+j)
                A[i][j]=sum
    
    for i in range(a):
        sum=0
        for j in range(n):
            sum+=(x[j]**i)*y[j]
        X[i]=sum
    return A,X
# ____________________________________________________________________________________________________________________________________
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
    N = ((b-a)**3*(func(x))/(24*error))**0.5
    N = ceil(N)
    return N

def err_trapN(func,b,a,x,err):
    N = ((b-a)**3*(func(x))/(12*error))**0.5
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
    return X, Y

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
    
    