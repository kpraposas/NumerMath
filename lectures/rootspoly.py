"""Module for evaluating and root-finding for polynomials. """

# Third-party import
import numpy as np

# Local import
from timer import timer # type: ignore

class Result:
    """
    Class for solution and evaluation of roots of a polynomial equations

    Attributes
    ----------
    numit : list
        number of iterations for each root
    maxit : int
        maximum number of iterations
    refnum : list
        number of iterations in refinement of each root
    refmax : int
        maximum number of refinements
    z : list
        approximate roots
    funval : list
        function value at each roots
    error : list
        error of each root without going thru refinement
    errror_ref : list
        error of each root after going thru refinement
    tol : float
        tolerance of the method
    reftol : float
        tolerance for the refinement step
    ref : bool
        if roots are to be refined or not
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the root-finding method
    """

    def __init__(self, numit, maxit, refnum, refmax, z, funval, error,
                 error_ref, tol, reftol, ref, elapsed_time, method_name):
        """
        Class initialization
        """
        self.numit = numit
        self.maxit = maxit
        self.refnum = refnum
        self.refmax = refmax
        self.z = z
        self.funval = funval
        self.error = error
        self.error_ref = error_ref
        self.tol = tol
        self.reftol = reftol
        self.ref = ref
        self.elapsed_time = elapsed_time
        self.method_name = method_name
    
    def print_table(self):
        if self.ref:
            pass
        else:
            pass
    
    def __str__(self):
        """
        Class string representation.
        """
        
        if self.ref:
           list_str_repr = [
               "POLYNOMIAL ROOT FINDER:          {}\n".format(self.method_name),
               "MAX ITERATIONS:                  {}\n".format(self.maxit),
               "TOLERANCE:                       {:.16e}\n".format(self.tol),
               "REFINEMENT:                      TRUE\n",
               "MAX REFINEMENT ITERATIONS:       {}\n".format(self.refmax),
               "REF TOLERANCE:                   {:.16e}\n".format(self.reftol)
           ]
        else:
            list_str_repr = [
                "POLYNOMIAL ROOT FINDER:         {}\n".format(self.method_name),
                "MAX ITERATIONS:                 {}\n".format(self.maxit),
                "TOLERANCE:                      {}\n".format(self.tol),
                "REFINEMENT:                     FALSE\n"
            ]
        return "".join(list_str_repr) + self.print_table()

class param():
    """
    Class for parameters in scalar root-finding algorithms.

    Attributes
    ----------
    tol : float
        tolerance of the method, default is machine epsilon
    reftol : float
        tolerance for the refinement subroutine, default is 1e-3
    maxit : int
        maximum number of iterations, default is 1000
    refmax : int
        maximum number of iterations for refinement, default = 100
    ref : bool
        refinement subroutine, default is true
    abstol : float
        weight of the absolute error of the root approximate
    funtol : float
        weight of the absolute function value of the root approximate
    """
    def __init__(self, maxit=1000, tol=np.finfo("float").eps, abstol=0.9, funtol=0.1,
                 refmax=100, reftol=1e-3, ref=True):
        """
        Class initialization.
        """
        self.abstol = abstol
        self.funtol = funtol
        self.tol = tol
        self.reftol = reftol
        self.maxit = maxit
        self.refmax = refmax
        self.ref = ref

# Constant for the machine epsilon
eps = np.finfo("float").eps

def usualpolyeval(p: list, z: complex) -> complex:
    """
    Usual polynomial evaluation of a polynomial p with coefficients
        [a0, a1, ..., an] evaluated at a complex number z.

    Parameters:
    -----------
    p : list
        coefficients of p
    z : complex
        valuation of p

    Returns
    -------
    complex
        function value of p at z
    """
    v = p[0]
    x = 1
    for j in range (1, len(p)):
        x = x * z
        v += p[j] * x
    return v

def horner(p: list, z: complex) -> {complex, list}:
    """
    Nested multiplication polynomial evaluation of a polynomial p with
        coefficients [a0, a1, ..., an] evaluated at a complex number z.
        Returns the function value p(z) and remainder q = [b1, b2, ..., bn]

    Parameters:
    -----------
    p : list
        coefficients of p
    z : complex
        valuation of p

    Returns
    -------
    b0 : complex
        function value of p at z
    [b1, b2, ..., bn] : list
        remainder of p when (x-z) is factored out
    """
    n = len(p) - 1
    q = [0] * (n)
    q[n - 1] = p[n]
    for k in range(n - 2, -1, -1):
        q[k] = p[k + 1] + q[k + 1] * z
    q0 = p[0] + q[0] * z
    return q0, q

def newtonhorner(p: list, z: complex, parameter: param) -> Result:
    """
    Newton-Horner method for approximating roots of a polynomial p
        with coefficients [a0, a1, ..., an]

    Parameters
    ----------
    p : list
        coefficients of the polynomial a0 + a1x + a2x^2 + ... + anx^2
    z : complex
        initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootspoly.Result
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(p) - 1
    p_aux = p[:]
    
    z_array = []
    pz_array = []
    numit_array = []
    refnum_array = []
    error_array = []
    error_ref_array = []
    
    # main loop
    for m in range(n):
        error_ref = parameter.tol * parameter.reftol
        k = 0
        z = complex(z, z)
        error = parameter.tol + 1.
        if m == n - 1:
            k += 1
            z = - p_aux[0] / p_aux[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                z_old = z
                pz, q = horner(p_aux, z)
                qz = horner(q, z)[0]
                if abs(qz) > eps:
                    z = z - pz/qz
                    error = max(np.absolute(z - z_old), np.absolute(pz))
                else:
                    error = 0.
        numit_array.append(k)
        error_array.append(error)
        
        # Refinement step
        if parameter.ref == True:
            k = 0
            z_ref = z
            error = parameter.tol + 1.
            while error > error_ref and k < parameter.refmax:
                k += 1
                pz, q = horner(p, z_ref)
                qz = horner(q, z_ref)[0]
                if np.abs(qz) > eps:
                    z_temp = z_ref
                    z_ref = z_ref - pz/qz
                    error = max(np.absolute(z_ref - z_temp), np.absolute(pz))
                else:
                    error = 0.
            z = z_ref
            refnum_array.append(k)
            error_ref_array.append(error)
        else:
            pz = horner(p, z)[0]
        
        z_array.append(z)
        pz_array.append(pz)
        p_aux = horner(p_aux, z)[1]
    stopwatch.stop()
    return Result(numit_array, parameter.maxit, refnum_array, parameter.refmax,
                  z_array, pz_array, error_array, error_ref_array, parameter.tol,
                  parameter.reftol, parameter.ref, stopwatch.get_elapsed_time,
                  "NEWTONHORNER")


def mullerhorner(p: list, x: list, parameter: param) -> Result:
    """
    Muller-Horner method for accelerating ix point method approximating
        a solution of a polynomial p with coefficients [a0, a1, ..., an]

    Parameters
    ----------
    p : list
        coefficients of the polynomial a0 + a1x + a2x^2 + ... + anx^2
    x : complex
        initial iterate containing three distinct points
    parameter : param
        parameters of the method

    Returns
    -------
    rootspoly.Result
        Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    n = len(p) - 1
    p_aux = p
    
    z_array = []
    pz_array = []
    numit_array = []
    refnum_array = []
    error_array = []
    error_ref_array = []
    
    # main loop
    for m in range(n):
        error_ref = parameter.tol * parameter.reftol
        k = 0
        error = parameter.tol + 1
        z0, z1, z2 = complex(x[0], 0.), complex(x[1], 0.), complex(x[2], 0.)
        if m == n - 1:
            k += 1
            z = - p_aux[0] / p_aux[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                f0, f1, f2 = horner(p_aux, z0)[0], horner(p_aux, z1)[0],\
                    horner(p_aux, z2)[0]
                f01 = (f1 - f0) / (z1 - z0)
                f12 = (f2 - f1) / (z2 - z1)
                f012 = (f12 - f01) / (z2 - z0)
                w = f12 + (z2 - z1)*f012
                alpha = w*w - 4.*f2*f012
                d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
                if np.abs(d) > eps:
                    z = z2 - 2.*f2/d
                    error = max(np.abs(z - z2), np.abs(f2))
                else:
                    error = 0.
                z0, z1, z2 = z1, z2, z
        numit_array.append(k)
        error_array.append(error)
        
        # Refinement step
        if parameter.ref == True:
            k = 0
            z_ref = z
            error = parameter.tol + 1
            while error > error_ref and k < parameter.refmax:
                k += 1
                pz, q = horner(p, z_ref)
                qz = horner(q, z_ref)[0]
                if np.abs(qz) > eps:
                    z_temp = z_ref
                    z_ref = z_ref - pz/qz
                    error = max(np.abs(z_ref - z_temp), np.abs(pz))
                else:
                    error = 0.
            z = z_ref
            refnum_array.append(k)
            error_ref_array.append(error)
        else:
            pz = horner(p, z)[0]
            
        z_array.append(z)
        pz_array.append(pz)
        p_aux = horner(p_aux, z)[1]
    stopwatch.stop()
    return Result(numit_array, parameter.maxit, refnum_array, parameter.refmax,
                  z_array, pz_array, error_array, error_ref_array, parameter.tol,
                  parameter.reftol, parameter.ref, stopwatch.get_elapsed_time,
                  "MULLERHORNER")