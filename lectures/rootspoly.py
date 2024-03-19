"""Module for evaluating and root-finding for polynomials. """

# Third-party import
import numpy as np

# Local import
from timer import timer

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
    x : list
        approximate roots
    funval : float
        function value at each roots
    tol : float
        tolerance of the method
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the root-finding method
    termination_flag : str
        either 'Fail' or 'Success'
    """

    def __init__(self, numit, maxit, refnum, refmax, x, funval, tol, elapsed_time,
        method_name, termination_flag):
        """
        Class initialization
        """
        self.numit = numit
        self.maxit = maxit
        self.refnum = refnum
        self.refmax = refmax
        self.x = x
        self.funval = funval
        self.tol = tol
        self.elapsed_time = elapsed_time
        self.method_name = method_name
        self.termination_flag = termination_flag
    
    def __str__(self):
        """
        Class string representation.
        """
        if self.termination_flag != None:
            term_flag = "{}\n".format(self.termination_flag)
        else: 
            term_flag = "\n"
        if self.tol != None:
            tol = "{:.16e}\n".format(self.tol)
        else: 
            tol = "\n"
        if self.maxit != None:
            maxit = "{}\n".format(self.maxit)
        else: 
            maxit = "\n"
        if self.refmax != None:
            refmax = "{}\n".format(self.refmax)
        else: 
            refmax = "\n"
        list_str_repr = [
            "ROOT FINDER:                    {}\n".format(self.method_name),
            "TERMINATION:                    ", term_flag,  
            "TOLERANCE:                      ", tol,
            "MAX ITERATIONS:                 ", maxit,
            "MAX REFINEMENTS:                ", refmax, 
            "ELAPSED TIME:                   {:.16e} seconds\n".format(self.elapsed_time),
            "-"*93,
            "\nREAL PART\t\tIMAG PART\t\t|FUNVAL|\t\t\t\tNUMIT\tREFNUM",
            "-"*93,
            "\n"
        ]
        for i in range(len(self.x)):
            if self.numit == None:
                self.numit = [""]*len(self.x)
            if self.refnum == None:
                self.numit = [""]*len(self.x)
            list_str_repr.append("{:+.16f}\t{:+.16f}\t{:.16f}\t{}\t{}\n".format(self.x[i].real, self.x[i].imag,
                    self.funval[i], self.numit[i], self.refnum[i]))
        list_str_repr.append("-"*93)
        return "".join(list_str_repr)


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

# Function evalution routines
# 1. Usual polynomial evaluation that takes 3n flops and returns the function value
# 2. Horner method that takes 2n flops and returns both the function value and the remainder polynomial
def usualpolyeval(p, z):
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
    v : complex
        function value of p at z
    """
    v = p[0]
    x = 1
    for j in range (1, len(p)):
        x = x * z
        v += p[j] * x
    return v


def horner(p, z):
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


# Options Dictionary Initialization
options = dict()
"""
    method : str
        method of Polynomial Deflation. Methods are Newton-Horner and Muller-Horner
"""
options = {
    "method" : "newtonhorner"
}

def rootspoly(p, z, options, parameter):
    """_summary_

    Args:
        p (_type_): _description_
        z (_type_): _description_
        options (_type_): _description_
        parameter (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(p) - 1
    if n == 2:
        return quadform(p)
    if n == 3:
        return cardano(p)
    if n == 4:
        return quartic(p)
    else:
        if options["method"] == "newtonhorner":
            return newtonhorner(p, z, parameter)
        if options["method"] == "mullerhorner":
            return mullerhorner(p, z, parameter)

def newtonhorner(p, z, parameter):
    """
    Newton-Horner method for approximating roots of a polynomial p
        with coefficients [a0, a1, ..., an]
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(p) - 1
    z_array = []
    p_array = []
    k_array = []
    k_ref_array = []
    eps = np.finfo(float).eps
    p_aux = p[:]
    error_ref = parameter.tol * parameter.reftol
    # main loop
    for m in range(n):
        k = 0
        x = complex(z, z)
        error = parameter.tol + 1
        if m == n - 1:
            k += 1
            x = - p_aux[0] / p_aux[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                z_old = x
                p_z, q_aux = horner(p_aux, x)
                q_z = horner(q_aux, x)[0]
                if abs(q_z) > eps:
                    x = z_old - p_z / q_z
                    error = max(abs(x - z_old), abs(p_z))
                else:
                    error = 0
                    x = z_old
        # refinement step
        if parameter.ref == True:
            k_ref = 0
            z_ref = x
            error = parameter.tol + 1
            while error > error_ref and k_ref < parameter.refmax:
                k_ref += 1
                p_z, q_aux = horner(p, z_ref)
                q_z = horner(q_aux, z_ref)[0]
                if abs(q_z) > eps:
                    z_ref2 = z_ref - p_z / q_z
                    error = max(abs(z_ref - z_ref2), abs(p_z))
                    z_ref = z_ref2
                else:
                    error = 0
            x = z_ref
            k_ref_array.append(k_ref)
        z_array.append(x)
        p_z, p_aux = horner(p_aux, x)
        p_array.append(p_z)
        k_array.append(k)
    stopwatch.stop()
    return Result(k_array, parameter.maxit, k_ref_array, parameter.refmax, z_array, p_array, 
            parameter.tol, stopwatch.get_elapsed_time, "NEWTON-HORNER", term_flag)


def mullerhorner(p, x, parameter):
    """
    Muller-Horner method for accelerating ix point method approximating a solution
        of a polynomial p with coefficients [a0, a1, ..., an]
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(p) - 1
    z_array = []
    p_array = []
    k_array = []
    k_ref_array = []
    eps = np.finfo(float).eps
    p_aux = p
    error_ref = parameter.tol * parameter.reftol
    # main loop
    for m in range(n):
        k = 0
        error = parameter.tol + 1
        z0, z1, z2 = complex(x[0], 0), complex(x[1], 0), complex(x[2], 0)
        if m == n - 1:
            k += 1
            z = - p_aux[0] / p_aux[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                f0, f1, f2 = horner(p_aux, z0)[0], horner(p_aux, z1)[0], horner(p_aux, z2)[0]
                f01 = (f1 - f0) / (z1 - z0)
                f12 = (f2 - f1) / (z2 - z1)
                f012 = (f12 - f01) / (z2 - z0)
                w = f12 + (z2 - z1) * f012
                alpha = w * w - 4. * f2 * f012
                if abs(alpha) >= 0:
                    d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
                    z = z2 - 2. * f2 / d
                    error = max(abs(z - z2), abs(f2))
                else:
                    z = z2 - f2 / f12
                    error = 0
                z0, z1, z2 = z1, z2, z
        # refinement step
        if parameter.ref == True:
            k_ref = 0
            z_ref = z
            error = parameter.tol + 1
            while error > error_ref and k_ref < parameter.refmax:
                k_ref +=  1
                p_z, q_aux = horner(p, z_ref)
                q_z = horner(q_aux, z_ref)[0]
                if abs(q_z) > eps:
                    z_ref2 = z_ref - p_z / q_z
                    error = max(abs(z_ref - z_ref2), abs(p_z))
                    z_ref = z_ref2
                else:
                    error = 0
            z = z_ref
            k_ref_array.append(k_ref)
        z_array.append(z)
        p_z, p_aux = horner(p_aux, z)
        p_array.append(p_z)
        k_array.append(k)
    stopwatch.stop()
    return Result(k_array, parameter.maxit, k_ref_array, parameter.refmax, z_array, p_array, parameter.tol,
        stopwatch.get_elapsed_time, "MULLER-HORNER", term_flag)