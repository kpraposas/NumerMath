""" Module for finding solutions for linear systems. """

# third party import
import numpy as np
from typing import Union

# local import
from timer import timer # type: ignore

def polyinterp():
    pass

class vector(list):
    """
    Class implentation of vector with real coefficients
    """
    def __init__(self, iterable=list()):
        if not iterable:
            super().__init__()
        else:
            super().__init__(iterable)
    
    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __str__(self):
        return super().__str__()
    
    def __add__(self, other):
        if len(self) == len(other):
            return self.__class__(a + b for a, b in zip(self, other))
        raise IndexError("Vectors need to be the same length")
    
    __radd__ = __add__
    
    def __neg__(self):
        return self.__class__(-element for element in self)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other - self
    
    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return self.__class__(other*element for element in self)
        return NotImplemented
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return self.__class__(element/other for element in self)
        return NotImplemented
    
    def norm(self, ord:Union[int, str]=2):
        if type(ord) == int:
            s = 0
            for element in self:
                s += abs(element)**ord
            return s**(1./ord)
        if ord == "max":
            return max([abs(element) for element in self])
        if ord == "min":
            return min([abs(element) for element in self])

class matrix(list):
    """
    Class implentation of matrix with real coefficients
    """
    def __init__(self, iterable):
        super().__init__(iterable)
        
    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __str__(self):
        os = []
        for element in self:
            os.append(element.__str__())
        return "\n".join(os)
    
    def __add__(self, other):
        if len(self) != len(other) or len(self[0]) != len(other[0]):
            raise IndexError("Matrices' dimensions do not match")
        return self.__class__(vector(a)+vector(b) for a,b in zip(self, other))
    
    __radd__ = __add__
    
    def __neg__(self):
        return self.__class__(-vector(element) for element in self)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other - self
    
    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return self.__class__(other*vector(element) for element in self)
        elif type(other) == list or type(other) == vector:
            m, n = len(self), len(self[0])
            y = vector([0]*m)
            for j in range(n):
                for i in range(m):
                    y[i] += self[i][j]*other[j]
            return y
        elif type(other) == matrix:
            m, n, p = len(self), len(self[0]), len(other[0])
            if n != len(other):
                raise IndexError("Matrices are not in the right dimensions")
            A = [[0]*p for _ in range(m)]
            for k in range(p):
                for j in range(n):
                    for i in range(m):
                        A[i][k] += self[i][j]*other[j][k]
            return self.__class__(A)
        return NotImplemented
    
    def __rmul__(self, other):
        if type(other) == int or type(other) == float:
            return self*other
        if type(other) == list or type(other) == vector:
            return self.transpose()*other
        return NotImplemented
    
    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return self.__class__(vector(element)/other for element in self)
        return NotImplemented
    
    def transpose(self):
        n, m = len(self), len(self[0])
        other = matrix([[0]*n for _ in range(m)])
        for i in range(n):
            for j in range(m):
                other[j][i] = self[i][j]
        return other

def dot(x: vector, y:vector) -> float:
    """
    Returns the inner product two vectors

    Parameters
    ----------
    x : vector
        first vector
    y : vector
        second vector

    Returns
    -------
    float
        inner product of x and y
    """
    n = len(x)
    s = 0
    for k in range(n):
        s += x[k]*y[k]
    return s

def cross(x: vector, y: vector) -> matrix:
    """
    Returns the outer product two vectors

    Parameters
    ----------
    x : vector
        first vector
    y : vector
        second vector

    Returns
    -------
    matrix
        outer product of x and y
    """
    n = len(x)
    A = [[0]*n]*n
    for i in range(n):
        for j in range(n):
            A[i][j] = x[i]*y[j]
    return matrix(A)

# Constant for machine epsilon
eps = np.finfo("float").eps
"""
Solving Linear Systems
"""

# Direct methods
class direct:
    """
    Class for solution of linear systems by direct methods
    
    Attributes
    ----------
    x : vector
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
            "RESIDUAL MAX:          {:.16e}\n".format(self.residual_max),
            "ELAPSED TIME:          {:.16e} seconds\n".format(self.elapsed_time)
            ]
        return "".join(list_str_repr)

def forwardsubrow(L: matrix, b: vector) -> direct:
    """
    Forward substitution by rows of a lower triangular matrix that solves the
        equation Lx = b.

    Parameters
    ----------
    L : matrix
        lower triangular coefficient matrix
    b : vector
        constant vector

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    x = vector()
    x.append(b[0] / L[0][0])
    for i in range(1, n):
        s = 0
        for j in range(0, i):
            s += L[i][j] * x[j]
        x.append((b[i] - s) / L[i][i])
    stopwatch.stop()
    residual_max = (L*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, "ForwardSubRow")

def forwardsubcol(L: matrix, b: vector) -> direct:
    """
    Forward substitution by columns of a lower triangular matrix that solves
        the equation Lx = b, stored in b.

    Parameters
    ----------
    L : matrix
        lower triangular coefficient matrix
    b : vector
        constant vector

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    b_old = vector(b[:])
    for j in range(0, n - 1):
        b[j] = b[j] / L[j][j]
        for i in range(j + 1, n):
            b[i] = b[i] - L[i][j] * b[j]
    b[n - 1] = b[n - 1] / L[n - 1][n - 1]
    stopwatch.stop()
    residual_max = (L*b - b_old).norm("max")
    return direct(b, residual_max, stopwatch.get_elapsed_time, "ForwardSubCol")

def backwardsubrow(U: matrix, b: vector) -> direct:
    """
    Backward substitution by rows of an upper triangular matrix that solves the
        equation Ux = b.

    Parameters
    ----------
    U : matrix
        upper triangular coefficient matrix
    b : matrix
        constant vector

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    x = vector([0] * n)
    x[n - 1] = b[n - 1] / U[n - 1][n - 1]
    for i in range(n - 2, -1, - 1):
        s = 0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (b[i] - s) / U[i][i]
    stopwatch.stop()
    residual_max = (U*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, "BackwardSubRow")

def backwardsubcol(U: matrix, b: vector) -> direct:
    """
    Backward substitution by columns of an upper triangular matrix that solves
        the equation Ux = b, stored in b.

    Parameters
    ----------
    U : matrix
        upper triangular coefficient matrix
    b : vector
        constant vector

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(b)
    b_old = vector(b[:])
    for j in range(n - 1, 0, -1):
        b[j] = b[j] / U[j][j]
        for i in range(0, j):
            b[i] = b[i] - U[i][j] * b[j]
    b[0] = b[0] / U[0][0]
    stopwatch.stop()
    residual_max = (U*b - b_old).norm("max")
    return direct(b, residual_max, stopwatch.get_elapsed_time, "BackwardSubCol")

def LUkji(A: matrix) -> matrix:
    """
    Single-precision a*x plus y version of LU factorization of A

    Parameters
    ----------
    A : matrix
        n by n matrix

    Returns
    -------
    matrix
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

def LUjki(A: matrix) -> matrix:
    """
    Generalized single-precision a*x plus y version of LU factorization of A

    Parameters
    ----------
    A : matrix
        n by n matrix

    Returns
    -------
    matrix
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

def LUijk(A: matrix) -> matrix:
    """
    Doolittle method of LU factorization of A

    Parameters
    ----------
    A : matrix
        n by n matrix

    Returns
    -------
    matrix
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

def LUpartial(A: matrix, eps: float=1.) -> tuple[matrix, vector]:
    """
    LU factorization with partial pivoting followed by Doolittle Method

    Parameters
    ----------
    A : matrix
        n by n matrix
    eps : float, optional
        pivot threshold, by default 1

    Returns
    -------
    tuple[matrix, vector]
        LU factorization of matrix stored in A
        permutation of the rows of the matrix 
    """

def LUcomplete(A: matrix, eps: float=1.) -> tuple[matrix, vector, vector]:
    """
    LU factorization with complete pivoting followed by Doolittle Method

    Parameters
    ----------
    A : matrix
        n by n matrix
    eps : float, optional
        pivot threshold, by default 1.

    Returns
    -------
    tuple[matrix, vector, vector]
        LU factorization of matrix stored in A
        permutation of the rows of the matrix
        permutation of the columns of the matrix 
    """

def getLU(A: matrix) -> tuple[matrix, matrix]:
    """
    Returns the LU factorization of A as separate matrices from LU
        factorization method. Note that that argument must be the return of 
        any of the LU Factorization procedures

    Parameters
    ----------
    A : matrix
        LU factorization of A obtained from any LU factorization methods

    Returns
    -------
    tuple[matrix, matrix]
        L and U factorization of A 
    """
    n = len(A)
    L = matrix([[0] * n for _ in range (0, n)])
    U = matrix([[0] * n for _ in range (0, n)])
    for i in range(0, n):
        L[i][i] = 1
        for j in range(0, i):
            L[i][j] = A[i][j]
        for j in range(i, n):
            U[i][j] = A[i][j]
    return L, U


def solveLU(A: matrix, b: vector, method: str='ijk') -> direct:
    """
    Solves the equation Ax = b through LU factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : matrix
        LU factorization of A obtained from any LU factorization methods
    b : vector
        n-dimensional vector
    method : str, optional
        LU factorization to be used, by default 'ijk'

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = matrix(row[:] for row in A)
    if method == 'kji':
        A = LUkji(A)
        method_name = 'LUSolve - KJI'
    elif method == 'jki':
        A = LUjki(A)
        method_name = 'LUSolve - JKI'
    elif method == 'ijk':
        A = LUijk(A)
        method_name = 'LUSolve - IJK'
    L, U = getLU(A)
    y = forwardsubrow(L, b).x
    x = backwardsubrow(U, y).x
    stopwatch.stop()
    residual_max = (A_old*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, method_name)

def solveLUPivot(A: matrix, b: vector, eps: float=1., method: str="partial")\
        -> direct:
    """
    Solves the equation Ax = b through LU factorization with pivoting and
        forward then backward substitution.

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    eps : float, optional
        pivot threshold, by default 1.
    method : str, optional
        partial or complete pivot, by default "partial"

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = matrix([row[:] for row in A])
    if method == "partial":
        B, p = LUpartial(A, eps)
        method_name = "LUPivotSolve"
    if method == "complete":
        B, p, q = LUcomplete(A, eps)
        method_name = "LUCompletePivotSolve"
        
    L, U = getLU(A)
    # Permute b according to p
    
    y = forwardsubrow(L, b).x
    x = backwardsubrow(U, y).x
    # Permute x according to q if pivoting is complete
    
    stopwatch.stop()
    residual_max = (A_old*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, method_name)

def iterativeref(A: matrix, b: vector, tol: float=2.*eps, maxit: int=10,
                 method: str="partial")\
                 -> direct:
    """
    Refinement of solution y obtained by direct solvers by solving the system
    Az = b - Ay and taking x = y + z

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    tol : float, optional
        pivot threshold, by default 2.*eps
    maxit : int, optional
        maximum number of iterations for refinement, by default 10
    method : str, optional
        method of initially computing solution prior refinement

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = matrix([row[:] for row in A])
    err = tol + 1
    k = 0
    match method:
        case "ijk":
            x = solveLU(A, b)
        case "kji":
            x = solveLU(A, b, "kji")
        case "jki":
            x = solveLU(A, b, "jki")
        case "partial":
            x = solveLUPivot(A, b)
        case "complete":
            x = solveLUPivot(A, b, 1., "complete")
    while err > tol and k < maxit:
        r = b - A_old*x
        match method:
            case "ijk":
                z = solveLU(A, r)
            case "kji":
                z = solveLU(A, r, "kji")
            case "jki":
                z = solveLU(A, r, "jki")
            case "partial":
                z = solveLUPivot(A, r)
            case "complete":
                z = solveLUPivot(A, r, 1., "complete")
        x = x + z
        err = z.norm()/x.norm()
    stopwatch.stop()
    residual_max = (A_old*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, "IterativeRef")
    
    
def ModifiedGramSchmidt(A: matrix) -> tuple[matrix, matrix]:
    """
    QR factorization of A by the Gram-Schidmt Orthogonalization

    Parameters
    ----------
    A : matrix
        m by n matrix

    Returns
    -------
    tuple[matrix, matrix]
        orthogonal m by n matrix
        upper triangular n by n matrix
    """
    m = len(A)
    n = len(A[0])
    Q = matrix([[0] * n for _ in range(m)])
    R = matrix([row[:] for row in A])
    q = vector(A.transpose()[0]).norm()
    for i in range(m):
        Q[i][0] = A[i][0] / q
    for k in range(1, n):
        for j in range(k):
            s = 0
            for i in range(m):
                s = s + Q[i][j] * A[i][k]
            for i in range(m):
                A[i][k] = A[i][k] - s * Q[i][j]
        q = vector(A.transpose()[k]).norm()
        for i in range(m):
            Q[i][k] = A[i][k] / q
    R = Q.transpose()*R
    return Q, R

def ModifiedGramSchmidt2(A: matrix) -> tuple[matrix, matrix]:
    """
    QR factorization of A by the Gram-Schidmt Orthogonalization 
        with modification to guarantee linear independence        

    Parameters
    ----------
    A : matrix
        m by n matrix

    Returns
    -------
    tuple[matrix, matrix]
        orthogonal m by n matrix
        upper triangular n by n matrix
    """
    m, n = len(A), len(A[0])
    Q = matrix([[0] * n for _ in range(m)])
    R = matrix([[0] * n for _ in range(n)])
    for k in range(n):
        R[k][k] = vector(A.transpose()[k]).norm()
        for j in range(m):
            Q[j][k] = A[j][k] / R[k][k]
        for j in range(k + 1, n):
            for i in range(m):
                R[k][j] = R[k][j] + Q[i][k] * A[i][j]
            for i in range(m):
                A[i][j] = A[i][j] - R[k][j] * Q[i][k]
    return Q, R

def solveQR(A: matrix, b: vector, method: int=1) -> direct:
    """
    Solves the equation Ax = b through QR factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : matrix
        QR factorization of A obtained from any QR factorization methods
    b : vector
        n-dimensional vector
    method : int, optional
        1 for default second method and 0 for the first GramSchmidt method,
            by default 1

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    A_old = matrix([row[:] for row in A])
    match method:
        case 0:
            Q, R = ModifiedGramSchmidt(A)
            method_name = 'QRSolve - GSO'
        case 1:
            Q, R = ModifiedGramSchmidt2(A)
            method_name = 'QRSolve - GSO2'
    x = backwardsubrow(R, Q.transpose()*b).x
    stopwatch.stop()
    residual_max = (A_old*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, method_name)

def Cholesky(A: matrix) -> matrix:
    """
    LU factorization of a symmetric positive definite n by n matrix (SPD) A
        where U = L^T

    Parameters
    ----------
    A : matrix
        SPD n by n matrix

    Returns
    -------
    matrix
        lower triangular matrix L
    """
    n = len(A)
    L = matrix([[0] * n for i in range(n)])
    L[0][0] = np.sqrt(A[0][0])
    for i in range(1, n):
        for j in range(i):
            s = 0
            for k in range(j):
                s = s + L[i][k] * L[j][k]
            L[i][j] = (A[i][j] - s) / L[j][j]
        s = 0
        for j in range(n):
            s =  s + L[i][j] * L[i][j]
        L[i][i] = np.sqrt(A[i][i] - s)
    return L

def solveChol(A: matrix, b: vector) -> direct:
    """
    Solves the equation Ax = b through Cholesky factorization and forward
        then backward substitution.

    Parameters
    ----------
    A : matrix
        Cholesky factorization of A
    b : vector
        n-dimensional vector

    Returns
    -------
    direct
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    L = Cholesky(A)
    y = forwardsubrow(L, b).x
    x = backwardsubrow(L.transpose(), y).x
    stopwatch.stop()
    residual_max = (A*x - b).norm("max")
    return direct(x, residual_max, stopwatch.get_elapsed_time, 'CholSolve')

# Iterative Methods
class iterative:
    """
    Class for solution of linear systems by direct methods
    
    Attributes
    ----------
    x : vector
        approximate solution
    numit : int
        number of iterations
    maxit : int
        maximum number of iterations
    rel_err : float
        relative maximum error of Ax - b
    tol : float
        tolerance of the method
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the method
    termination_flag : str
        either 'Fail' or 'Success'
    """
    
    def __init__(self, x, numit, maxit, rel_err, tol, elapsed_time,
                 method_name, termination_flag):
        """
        Class initialization
        """
        self.x = x
        self.numit = numit
        self.maxit = maxit
        self.rel_err = rel_err
        self.tol = tol
        self.elapsed_time = elapsed_time
        self.method_name = method_name
        self.termination_flag = termination_flag

    def __str__(self):
        """
        Class string representation.
        """
        list_str_repr = [
            "METHOD:                  {}\n".format(self.method_name),
            "APPROXIMATE SOLUTION:    {}\n".format(self.x),
            "TERMINATION:             {}\n".format(self.termination_flag),
            "RESIDUAL MAX:            {:.16e}\n".format(self.rel_err),
            "TOLERANCE:               {:.16e}\n".format(self.tol),
            "NUM ITERATIONS:          {}\n".format(self.numit),
            "MAX ITERATIONS:          {}\n".format(self.maxit),
            "ELAPSED TIME:            {:.16e} seconds\n".format(self.elapsed_time)
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
        relaxation parameter, default is 0.5
    """

    def __init__(self, tol=1e-12, maxit=1e4, omega=0.5):
        """
        Class initialization.
        """
        self.tol = tol
        self.maxit = maxit
        self.omega = omega
        

def JOR(A: matrix, b: vector, x: vector, parameter: param) -> iterative:
    """
    Jacobi Over-relaxation method in solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    
    # Pivoting step
    
    b_norm = b.norm()
    err = (b - A*x).norm() / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(n):
            x_old = vector(x[:])
            s = 0
            for j in range(n):
                if j == i:
                    continue
                s = s + A[i][j] * x_old[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i]\
                + (1 - parameter.omega) * x_old[i]
        err = (b - A*x).norm() / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "JOR", term_flag)

def SOR(A: matrix, b: vector, x: vector, parameter: param) -> iterative:
    """
    Successive Over-relaxation method in solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    
    # Pivoting step
    
    b_norm = b.norm()
    err = (b - A*x).norm() / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(0, n):
            x_old = vector(x[:])
            s = 0
            for j in range(0, i):
                s = s + A[i][j] * x[j]
            for j in range(i + 1, n):
                s = s + A[i][j] * x_old[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i]\
                + (1 - parameter.omega) * x_old[i]
        err = (b - A*x).norm() / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "SOR", term_flag)

def BSOR(A: matrix, b: vector, x: vector, parameter: param) -> iterative:
    """
    Backward Symmetric Over-relaxation method in solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    
    # Pivoting step
    
    b_norm = b.norm()
    err = (b - A*x).norm() / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        for i in range(0, n):
            x_old = vector(x[:])
            s = 0
            for j in range(0, i):
                s = s + A[i][j] * x_old[j]
            for j in range(i + 1, n):
                s = s + A[i][j] * x[j]
            x[i] = parameter.omega * (b[i] - s) / A[i][i]\
                + (1 - parameter.omega) * x_old[i]
        err = (b - A*x).norm() / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "BSOR", term_flag)
    
def SSOR(A: matrix, b: vector, x: vector, parameter: param) -> iterative:
    """
    Successive Over-relaxation method in solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(b)
    
    # Pivoting step
    
    b_norm = b.norm()
    err = (b - A*x).norm() / b_norm
    k = 0
    while err > parameter.tol and k < parameter.maxit:
        y = vector(x[:])
        for i in range(0, n):
            x_old, y_old = vector(x[:]), vector(y[:])
            r, s = 0, 0      
            for j in range(0, i):
                r = r + A[i][j] * y[j]
                s = s + A[i][j] * y[j]
            for j in range(i + 1, n):
                r = r + A[i][j] * x_old[j]
                s = s + A[i][j] * x[j]
            y[i] = parameter.omega * (b[i] - r) / A[i][i]\
                + (1 - parameter.omega) * y_old[i]
            x[i] = parameter.omega * (b[i] - s) / A[i][i]\
                + (1 - parameter.omega) * x_old[i]
        err = (b - A*x).norm() / b_norm
        k += 1
    if err > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "SSOR", term_flag)

# Optimization-based methods for symmetric positive definite matrices

def steepestdescent(A: matrix, b: vector, x: vector, parameter: param)\
        -> iterative:
    """
    Steepest Descent method of solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = b - A*x
    rho = dot(r, r)
    k = 0
    while k < parameter.maxit:
        s = dot(r, A*r)
        alpha = rho / s
        x = x + alpha*r
        r = b - A*x
        rho = dot(r, r)
        k += 1
        if np.sqrt(rho) <= parameter.tol:
            break
    if np.sqrt(rho) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "SteepestDescent", term_flag)
    
def conjugategradient(A: matrix, b: vector, x: vector, parameter: param)\
        -> iterative:
    """
    Conjugate Gradient method of solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = b - A*x
    d = vector(r[:])
    rho = dot(r, r)
    k = 0
    while k < parameter.maxit:
        w = A*d
        alpha = rho / dot(d, w)
        x = x + alpha*d
        r = r - alpha*w
        rho_old = rho
        rho = dot(r, r)
        k += 1
        if np.sqrt(rho) <= parameter.tol:
            break
        beta = rho_old / rho
        d = r + beta*d
    if np.sqrt(rho) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "ConjugateGradient", term_flag)
    
def cgnormalresidual(A: matrix, b: vector, x: vector, parameter: param)\
        -> iterative:
    """
    Conjugate Gradient Normal Residual method of solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = b - A*x
    d = A.transpose()*r
    q = vector(d[:])
    sigma = dot(q, q)
    k = 0
    while k < parameter.maxit:
        w = A*d
        alpha = sigma / dot(w, w)
        x = x + alpha*d
        r = r - alpha*w
        k += 1
        if np.sqrt(dot(r, r)) <= parameter.tol:
            break
        sigma_old = sigma
        q = A.transpose()*r
        sigma = dot(q, q)
        beta = sigma_old / sigma
        d = q + beta*d
    if np.sqrt(dot(r, r)) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "CGNormalResidual", term_flag)
    

def cgresidual(A: matrix, b: vector, x: vector, parameter: param)\
        -> iterative:
    """
    Conjugate Gradient Residual method of solving linear system

    Parameters
    ----------
    A : matrix
        n by n matrix
    b : vector
        n-dimensional vector
    x : vector
        initial guess
    parameter : param
        Parameters of the method

    Returns
    -------
    iterative
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    r = b - A*x
    d = vector(r[:])
    k = 0
    while k < parameter.maxit:
        w = A*d
        omega = dot(w, w)
        alpha = dot(r, w) / omega
        x = x + alpha*d
        r = r - alpha*w
        k += 1
        if np.sqrt(dot(r, r)) <= parameter.tol:
            break
        beta = -dot(w, A*r) / omega
        d = r + beta*d
        k += 1
    if np.sqrt(dot(r, r)) > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    rel_err = (A*x - b).norm("max")
    return iterative(x, k, parameter.maxit, rel_err, parameter.tol,
                     stopwatch.get_elapsed_time, "CGResidual", term_flag)