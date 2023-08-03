""" Module for finding solutions for linear systems. """

# third party import
import numpy as np

# local import
from timer import timer

"""
Linear Algebra Operations
"""

def vectorsum(A, B):
    """
    Sum of two n by 1 vectors.

    Parameters
    ----------
    A : list
        n-dimensional vector
    B : list
        n-dimensional vector

    Returns
    -------
    C : list
        n-dimensional vector
    """
    n = len(A)
    C = [0] * n
    for i in range(n):
        C[i] = A[i] +  B[i]
    return C

def vectordiff(A, B):
    """
    Difference of two n by 1 vectors.

    Parameters
    ----------
    A : list
        n-dimensional vector
    B : list
        n-dimensional vector

    Returns
    -------
    C : list
        n-dimensional vector
    """
    n = len(A)
    C = [0] * n
    for i in range(n):
        C[i] = A[i] -  B[i]
    return C

def matrixsum(A, B):
    """
    Sum of two n by m matrix.

    Parameters
    ----------
    A : list
        n by m matrix
    B : list
        n by m matrix

    Returns
    -------
    C : list
        n by m matrix
    """
    n = len(A)
    m = len(A[0])
    C = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] +  B[i][j]
    return C

def matrixdiff(A, B):
    """
    Difference of two n by m matrix stored in A.

    Parameters
    ----------
    A : list
        n by m matrix
    B : list
        n by m matrix

    Returns
    -------
    C : list
        n by m matrix
    """
    n = len(A)
    m = len(A[0])
    C = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] -  B[i][j]
    return C

def scalarvectorprod(A, x):
    """
    Scalar multiplication for a n by 1 vector with a real number x.

    Parameters
    ----------
    A : list
        n-dimensional vector
    x : float
        scalar

    Returns
    -------
    C : list
        n-dimensional vector
    """
    n = len(A)
    C = [0] * n
    for i in range(n):
        C[i] = x * A[i]
    return C

def scalarmatrixprod(A, x):
    """
    Scalar multiplication for a n by m matrix with a real number x.

    Parameters
    ----------
    A : list
        n by m matrix
    x : float
        scalar

    Returns
    -------
    A : list
        n by m matrix
    """
    n = len(A)
    m = len(A[0])
    C = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = x * A[i][j]
    return C


def matrixvectorSDOT(A, x):
    """
    Single precision dot product of matrix A and vector x.

    Parameters
    ----------
    A : list
        m by n matrix
    x : list
        n-dimensional vector

    Returns
    -------
    y : list
        m-dimensional vector
    """
    m = len(A)
    n = len(A[0])
    y = [0] * m
    for i in range(m):
        for j in range(n):
            y[i] += A[i][j] * x[j]
    return y


def matrixvectorSAXPY(A, x):
    """
    Single precision A times x plus y product of matrix A and vector x.

    Parameters
    ----------
    A : list
        m by n matrix
    x : list
        n-dimensional vector

    Returns
    -------
    y : list
        m-dimensional vector
    """
    m = len(A)
    n = len(A[0])
    y = [0] * m
    for j in range(n):
        for i in range(m):
            y[i] += A[i][j] * x[j]
    return y


def matrixprodkji(A, B):
    """
    Product of two matrices A and B with the appropriate sizes.

    Parameters
    ----------
    A : list
        m by n matrix
    B : list
        n by p matrix

    Returns
    -------
    C : list
        m by p matrix
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    C = [[0] * m for i in range(n)] 
    for k in range(p):
        for j in range(n):
            for i in range(m):
                C[i][k] += A[i][j] * B[j][k]
    return C

def innerprod(x, y):
    """
    Inner product of n-vector x and y 

    Parameters
    ----------
    x : list
        n-dimensional vector
    y : list
        n-dimensional vector

    Returns
    -------
    s : float
        Inner product of x and y
    """
    n = len(x)
    s = 0
    for k in range(n):
        s += x[k] * y[k]
    return s


def outerprod(x, y):
    """
    Outer product of n-vector x and y 

    Parameters
    ----------
    x : list
        n-dimensional vector
    y : list
        n-dimensional vector

    Returns
    -------
    A : list
        Outer product of x and y
    """
    n = len(x)
    A = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = x[i] * y[j]
    return A


def transpose(A):
    """
    Transpose of a n by m matrix A.
    
    Parameters
    ----------
    A : list
        n by m matrix

    Returns
    -------
    T : list
        m by n matrix
    """
    n = len(A)
    m = len(A[0])
    T = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def Euclidnorm(X):
    """
    
    """
    n = len(X)
    sum = 0
    for i in range(n):
        sum += X[i]**2
    return np.sqrt(sum)

def Maxnorm(X):
    """
    
    """
    n = len(X)
    for i in range(n):
        X[i] = abs(X[i])
    return max(X)


"""
Solving Linear Systems
"""

# Direct methods
class Result_Direct:
    """
    Class for solution of linear systems by direct methods
    
    Attributes
    ----------
    x : list
        approximate solution
    residual : float
        maximum residual norm of Ax - b
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the method
    """
    
    
    def __init__(self, x, residual_max, elapsed_time, method_name):
        """
        Class initialization
        """
        self.x = x
        self.residual_max = residual_max
        self.elapsed_time = elapsed_time
        self.method_name = method_name

    def __str__(self):
        """
        Class string representation.
        """
        list_str_repr = [
            "METHOD:                {}\n".format(self.method_name),
            "APPROXIMATE SOLUTION:  {:.16f}\n".format(self.x),
            "RESIDUAL MAX:              {:.16e}\n".format(self.residual_max),
            "ELAPSED TIME:          {:.16e} seconds\n".format(self.elapsed_time)
            ]
        return "".join(list_str_repr)
    

def forwardsubrow(L, b):
    """
    Forward substitution by rows of a lower triangular matrix that solves the
        equation Lx = b.

    Parameters
    ----------
    L : list
        lower triangular coefficient matrix
    b : list
        constant vector

    Returns
    -------
    x : list
        approximate solution
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    x = []
    x.append(b[0] / L[0][0])
    for i in range(1, n):
        s = 0
        for j in range(0, i):
            s += L[i][j] * x[j]
        x.append((b[i] - s) / L[i][i])
    stopwatch.stop()
    Lx = matrixvectorSAXPY(L, x)
    residual_max = Maxnorm(vectordiff(Lx, b))
    return Result_Direct(x, residual_max, stopwatch.get_elapsed_time, "ForwardSubRow")


def forwardsubcol(L, b):
    """
    Forward substitution by columns of a lower triangular matrix that solves
        the equation Lx = b, stored in b.

    Parameters
    ----------
    L : list
        lower triangular coefficient matrix
    b : list
        constant vector

    Returns
    -------
    b : list
        approximate solution
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    b_old = b
    for j in range(0, n - 1):
        b[j] = b[j] / L[j][j]
        for i in range(j + 1, n):
            b[i] = b[i] - L[i][j] * b[j]
    b[n - 1] = b[n - 1] / L[n - 1][n - 1]
    stopwatch.stop()
    Lb = matrixvectorSAXPY(L, b)
    residual_max = Maxnorm(vectordiff(Lb, b_old))
    return Result_Direct(b, residual_max, stopwatch.get_elapsed_time, "ForwardSubCol")


def backwardsubrow(U, b):
    """
    Backward substitution by rows of an upper triangular matrix that solves the
        equation Ux = b.

    Parameters
    ----------
    U : list
        upper triangular coefficient matrix
    b : list
        constant vector

    Returns
    -------
    x : list
        approximate solution
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    x = [0] * n
    x[n - 1] = b[n - 1] / U[n - 1][n - 1]
    for i in range(n - 2, -1, - 1):
        s = 0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (b[i] - s) / U[i][i]
    stopwatch.stop()
    Ux = matrixvectorSAXPY(U, x)
    residual_max = Maxnorm(vectordiff(Ux, b))
    return Result_Direct(x, residual_max, stopwatch.get_elapsed_time, "BackwardSubRow")


def backwardsubcol(U, b):
    """
    Backward substitution by columns of an upper triangular matrix that solves
        the equation Ux = b, stored in b.

    Parameters
    ----------
    U : list
        upper triangular coefficient matrix
    b : list
        constant vector

    Returns
    -------
    b : list
        approximate solution
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    b_old = b
    for j in range(n - 1, 0, -1):
        b[j] = b[j] / U[j][j]
        for i in range(0, j):
            b[i] = b[i] - U[i][j] * b[j]
    b[0] = b[0] / U[0][0]
    stopwatch.stop()
    Ub = matrixvectorSAXPY(U, b)
    residual_max = Maxnorm(vectordiff(Ub, b_old))
    return Result_Direct(b, residual_max, stopwatch.get_elapsed_time, "BackwardSubCol")
    

def LUkji(A):
    """
    Single-precision a*x plus y version of LU factorization of A

    Parameters
    ----------
    A : list
        n by n matrix

    Returns
    -------
    A : list
        n by n matrix where the upper triangular part is U, lower triangular
        part is L and the digonal entries as 1
    """
    n = len(A)
    for k in range(0, n):
        for j in range(k + 1, n):
            A[j][k] = A[j][k] / A[k][k]
        for j in range(k + 1, n):
            for i in range(k + 1, n):
                A[i][j] = A[i][j] -  A[i][k] * A[k][j]
    return A


def LUjki(A):
    """
    Generalized single-precision a*x plus y version of LU factorization of A

    Parameters
    ----------
    A : list
        n by n matrix

    Returns
    -------
    A : list
        n by n matrix where the upper triangular part is U, lower triangular
        part is L, and the digonal entries as 1
    """
    n = len(A)
    for j in range(0, n):
        for k in range(0, j):
            for i in range(k + 1, n):
                A[i][j] = A[i][j] -  A[i][k] * A[k][j]
        for k in range(j + 1, n):
            A[k][j] = A[k][j] / A[j][j]
    return A


def LUijk(A):
    """
   Doolittle method of LU factorization of A

    Parameters
    ----------
    A : list
        n by n matrix

    Returns
    -------
    A : list
        n by n matrix where the upper triangular part is U, lower triangular
        part is L, and the digonal entries as 1
    """
    n = len(A)
    for j in range(1, n):
        A[j][0] = A[j][0] / A[0][0]
    for i in range(1, n):
        for j in range(i, n):
            s = 0
            for k in range(0, i):
                s += A[i][k] * A[k][j]
            A[i][j] = A[i][j] - s
        for j in range(i + 1, n):
            s = 0
            for k in range(0, i):
                s += A[j][k] * A[k][i]
            A[j][i] = (A[j][i] - s) / A[i][i]
    return A


def getLU(A):
    """
    Returns the LU factorization of A as separate matrices from LU
        factorization method.

    Parameters
    ----------
    A : list
        LU factorization of A obtained from any LU factorization methods

    Returns
    -------
    L, U : list
        n by n matrices 
    """
    n = len(A)
    L = [[0] * n for i in range (0, n)]
    U = [[0] * n for i in range (0, n)]
    for i in range(0, n):
        L[i][i] = 1
        for j in range(0, i):
            L[i][j] = A[i][j]
        for j in range(i, n):
            U[i][j] = A[i][j]
    return L, U


def LU_Solve(A, b, method='kji'):
    """
    Solves the equation Ax = b through LU factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : list
        LU factorization of A obtained from any LU factorization methods
    b : list
        n-dimensional vector
    method : 

    Returns
    -------
    x : list
        n-dimensional vector
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = A
    if method == 'kji':
        A = LUkji(A)
        method_name = 'LUSolve - KJI'
    if method == 'jki':
        A = LUjki(A)
        method_name = 'LUSolve - JKI'
    if method == 'ijk':
        A = LUijk(A)
        method_name = 'LUSolve - IJK'
    L, U = getLU(A)
    y = forwardsubcol(L, b).x
    x = backwardsubcol(U, y).x
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A_old, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Direct(x, residual_max, stopwatch.get_elapsed_time, method_name)
    

def ModifiedGramSchmidt(A):
    """
    QR factorization of A by the Gram-Schidmt Orthogonalization
    
    Parameters
    ----------
    A : list
        m by n matrix
        
    Returns
    -------
    Q : list
        orthogonal m by n matrix
    R : list
        upper triangular n by n matrix
    """
    m = len(A)
    n = len(A[0])
    Q = [[0] * n for i in range(m)]
    R = A
    q = Euclidnorm(A[0])
    for i in range(m):
        Q[i][0] = A[i][0] / q
    for k in range(1, n):
        for j in range(k):
            s = 0
            for i in range(m):
                s = s + Q[i][j] * A[i][k]
            for i in range(m):
                A[i][k] = A[i][k] - s * Q[i][j]
        q = Euclidnorm(A[k])
        for i in range(m):
            Q[i][k] = A[i][k] / q
    R = matrixprodkji(transpose(Q), R)
    return Q, R


def ModifiedGramSchmidt2(A):
    """
    QR factorization of A by the Gram-Schidmt Orthogonalization 
        with modification to guarantee linear independence
    
    Parameters
    ----------
    A : list
        m by n matrix
        
    Returns
    -------
    Q : list
        orthogonal m by n matrix
    R : list
        upper triangular n by n matrix
    """
    m = len(A)
    n = len(A[0])
    Q = [[0] * n for i in range(m)]
    R = [[0] * n for i in range(n)]
    for k in range(n):
        R[k][k] = Euclidnorm(A[k])
        for j in range(m):
            Q[j][k] = A[j][k] / R[k][k]
        for j in range(k + 1, n):
            for i in range(m):
                R[k][j] = R[k][j] + Q[i][k]
            for i in range(m):
                A[i][j] = A[i][j] - R[k][j] * Q[i][k]
    return Q, R


def QRSolve(A, b, method = 0):
    """
    Solves the equation Ax = b through QR factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : list
        QR factorization of A obtained from any QR factorization methods
    b : list
        n-dimensional vector
    method : bool
        0 for default method and 1 for the second GramSchmidt method

    Returns
    -------
    x : list
        n-dimensional vector
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = A
    if method == 0:
        Q, R = ModifiedGramSchmidt(A)
        method_name = 'GramSchmidt'
    else:
        Q, R = ModifiedGramSchmidt2(A)
        method_name = 'GramSchmidt2'
    Q_T = transpose(Q)
    x = backwardsubcol(R, matrixvectorSAXPY(Q_T, b)).x
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A_old, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Direct(x, residual_max, stopwatch.get_elapsed_time, method_name)


def Cholesky(A):
    """
    LU factorization of a symmetric positive definite n by n matrix (SPD) A
        where U = L^T
    
    Parameters
    ----------
    A : list
        SPD n by n matrix
    
    Returns
    -------
    L : list
        lower triangular matrix L
    """
    n = len(A)
    L = [[0] * n for i in range(n)]
    L[0][0] = np.sqrt(A[0][0])
    for i in range(1, n):
        for j in range(i - 1):
            s = 0
            for k in range(j - 1):
                s = s + L[i][k] * L[j][k]
            L[i][j] = (A[i][j] - s) / L[j][j]
        s = 0
        for j in range(n):
            s =  s + L[i][j] * L[i][j]
        L[i][i] = np.sqrt(A[i][i] - s)
    return L


def LL_TSolve(A, b):
    """
    Solves the equation Ax = b through Cholesky factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : list
        Cholesky factorization of A
    b : list
        n-dimensional vector

    Returns
    -------
    x : list
        n-dimensional vector
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = A
    L = Cholesky(A)
    L_T = transpose(L)
    y = forwardsubcol(L, b).x
    x = backwardsubcol(L_T, y).x
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A_old, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Direct(x, residual_max, stopwatch.get_elapsed_time, 'Chol_Solve')
    

# Iterative Methods
class Result_Iterative:
    """
    Class for solution of linear systems by direct methods
    
    Attributes
    ----------
    x : list
        approximate solution
    numit : int
        number of iterations
    maxit : int
        maximum number of iterations
    residual_max : float
        maximum residual norm of Ax - b
    tol : float
        tolerance of the method
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the method
    termination_flag : str
        either 'Fail' or 'Success'
    """
    
    
    def __init__(self, x, numit, maxit, residual_max, tol, elapsed_time, method_name, termination_flag):
        """
        Class initialization
        """
        self.x = x
        self.numit = numit
        self.maxit = maxit
        self.residual_max = residual_max
        self.tol = tol
        self.elapsed_time = elapsed_time
        self.method_name = method_name
        self.termination_flag = termination_flag

    def __str__(self):
        """
        Class string representation.
        """
        list_str_repr = [
            "METHOD:                                {}\n".format(self.method_name),
            "APPROXIMATE SOLUTION/ LAST ITERATE:    {:.16f}\n".format(self.x),
            "TERMINATION:                           {}\n".format(self.termination_flag),
            "RESIDUAL MAX:                          {:.16e}\n".format(self.residual_max),
            "TOLERANCE:                             {:.16e}\n".format(self.tol),
            "NUM ITERATIONS:                        {}\n".format(self.numit),
            "MAX ITERATIONS:                        {}\n".format(self.maxit),
            "ELAPSED TIME:                          {:.16e} seconds\n".format(self.elapsed_time)
            ]
        return "".join(list_str_repr)


class param():
    """
    Class for parameters in iterative methods.

    Attributes
    ----------
    tol : float
        tolerance of the method, default is 1e-12
    maxit : int
        maximum number of iterations, default is 10000
    omega : float
        relaxation parameter
    """

    def __init__(self, tol=1e-12, maxit=1e4, omega=0.5):
        """
        Class initialization.
        """
        self.tol = tol
        self.maxit = maxit
        self.omega = omega
        

def JOR(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(0, n):
            x_old = x
            s = 0
            for j in range(0, n):
                if j != i:
                    s = s + A[i][j] * x_old[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i] + (1 - parameter.omega) * x_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "JOR", term_flag)

def SOR(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(0, n):
            x_old = x
            s = 0
            for j in range(0, i):
                s = s + A[i][j] * x[j]
            for j in range(i + 1, n):
                s = s + A[i][j] * x_old[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i] + (1 - parameter.omega) * x_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "SOR", term_flag)
    

def BSOR(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(0, n):
            x_old = x
            s = 0
            for j in range(0, i):
                s = s + A[i][j] * x_old[j]
            for j in range(i + 1, n):
                s = s + A[i][j] * x[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i] + (1 - parameter.omega) * x_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "BSOR", term_flag)
    

def SSOR(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        y = x
        for i in range(0, n):
            x_old = x
            y_old = y
            r = 0
            s = 0            
            for j in range(0, i):
                r = r + A[i][j] * y[j]
                s = s + A[i][j] * y[j]
            for j in range(i + 1, n):
                r = r + A[i][j] * x_old[j]
                s = s + A[i][j] * x[j]
            y[i] = parameter.omega * (b[i] - r) / A[i][i] + (1 - parameter.omega) * x_old[i]
            x[i] = parameter.omega * (b[i] - s) / A[i][i] + (1 - parameter.omega) * y_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "SSOR", term_flag)


# Gradient Methods
def steepestdescent(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    Ax = matrixvectorSAXPY(A, x)
    r = vectordiff(b, Ax)
    rho = innerprod(r, r)
    k = 0
    while k < parameter.maxit:
        s = innerprod(r, matrixvectorSAXPY(A, r))
        alpha = rho / s
        x = vectorsum(x, scalarvectorprod(r, alpha))
        r = vectordiff(b, matrixvectorSAXPY(A, x))
        rho = innerprod(r, r)
        if np.sqrt(rho) <= parameter.tol:
            break
        k += 1
    if Euclidnorm(r) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "SteepestDescent", term_flag)
    

def conjugategradient(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = vectordiff(b, matrixvectorSAXPY(A, x))
    d = r
    rho = innerprod(r, r)
    k = 0
    while k < parameter.maxit:
        w = matrixvectorSAXPY(A, d)
        alpha = rho / innerprod(d, w)
        x = vectorsum(x, scalarvectorprod(d, alpha))
        r = vectordiff(r, scalarvectorprod(w, alpha))
        rho_old = rho
        rho = innerprod(r, r)
        if np.sqrt(rho) <= parameter.tol:
            break
        beta = rho_old / rho
        d = vectorsum(r, scalarvectorprod(d, beta))
        k += 1
    if Euclidnorm(r) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "ConjugateGradient", term_flag)
    

def cgnormalresidual(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = vectordiff(b, matrixvectorSAXPY(A, x))
    d = matrixvectorSAXPY(transpose(A), r)
    q = d
    sigma = innerprod(q, q)
    k = 0
    while k < parameter.maxit:
        w = matrixvectorSAXPY(A, d)
        alpha = sigma / innerprod(w, w)
        x = vectorsum(x, scalarvectorprod(d, alpha))
        r = vectordiff(r, scalarvectorprod(w, alpha))
        if Euclidnorm(r) <= parameter.tol:
            break
        sigma_old = sigma
        q = matrixvectorSAXPY(transpose(A), r)
        sigma = innerprod(q, q)
        beta = sigma_old / sigma
        d = vectorsum(q, scalarvectorprod(d, beta))
        k += 1
    if Euclidnorm(r) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "CGNormalResidual", term_flag)
    

def cgresidual(A, b, x, parameter):
    """
    
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = vectordiff(b, matrixvectorSAXPY(A, x))
    d = r
    k = 0
    while k < parameter.maxit:
        w = matrixvectorSAXPY(A, d)
        w_norm = innerprod(w, w)
        alpha = innerprod(r, w) / w_norm
        x = vectorsum(x, scalarvectorprod(d, alpha))
        r = vectordiff(r, scalarvectorprod(w, alpha))
        if Euclidnorm(r) <= parameter.tol:
            break
        beta = -innerprod(w, matrixvectorSAXPY(A, r)) / w_norm
        d = vectorsum(r, scalarvectorprod(d, beta))
        k += 1
    if Euclidnorm(r) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    Ax = matrixvectorSAXPY(A, x)
    residual_max = Maxnorm(vectordiff(Ax, b))
    return Result_Iterative(x, k, parameter.maxit, residual_max, parameter.tol, stopwatch.get_elapsed_time,
        "CGResidual", term_flag)