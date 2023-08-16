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
            "\nREAL PART\t\tIMAG PART\t\t|FUNVAL|\t\t\t\tNUMIT\t\REFNUM",
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


def quadform(p):
    """
    Computes the solutions of quadratic polynomial p using quadratic formula
    """
    stopwatch = timer()
    stopwatch.start()
    a, b, c = p[2], p[1], p[0]
    # Compute square root of discriminant
    sqrtDisc = np.sqrt(b*b - 4.*a*c)
    x = []
    x.append((-b + sqrtDisc) / (2.*a))
    x.append((-b - sqrtDisc) / (2.*a))
    stopwatch.stop()
    return Result(None, None, None, None, x, [horner(p, z)[0] for z in x], None, stopwatch.get_elapsed_time,
                  "Quadratic Formula", None)


def cardano(p):
    """
    Computes the solutions of cubic polynomial p using Cardano formula
    """
    stopwatch = timer()
    stopwatch.start()
    a, b, c, d = p[3], p[2], p[1], p[0]
    j = complex(0, 1.)
    q = (3.*a*c - b*b) / (9.*a*a)
    r = (9.*a*b*c - 27.*a*a*d - 2.*b**3) / (54.*a**3)
    sqrtD = np.sqrt(q**3 + r*r)
    u = r + sqrtD
    v = r - sqrtD
    if u.real >= 0.:
        s = np.power(u, 1./3.)
    else:
        s = -np.power(-u, 1./3.)
    if v.real >= 0.:
        t = np.power(v, 1./3.)
    else:
        t = -np.power(-v, 1./3.)
    x = []
    x.append(s + t - b/(3.*a))
    x.append(-0.5*(s + t) - b/(3.*a) + np.sqrt(3.)*j*(s - t)/2.)
    x.append(-0.5*(s + t) - b/(3.*a) - np.sqrt(3.)*j*(s - t)/2.)
    stopwatch.stop()
    return Result(None, None, None, None, x, [horner(p, z)[0] for z in x], None, stopwatch.get_elapsed_time,
                  "Cardano Formula", None)


def quartic(p):
    """
    Computes the solutions of quartic polynomial p using the formula
        by Lodovico Ferrari
    """
    stopwatch = timer()
    stopwatch.start()
    a, b, c, d, e = p[4], p[3], p[2], p[1], p[0]
    A, B, C, D = b/a, c/a, d/a, e/a
    k = -A/4.
    P = 3.*(2.*k + A)*k + B
    Q = ((2.*A)*k + 2.*B)*k + C
    R = (((k + A)*k + B)*k + C)*k + D
    x = []
    if Q == 0:
        sqrtDisc = np.sqrt(P*P - 4.*R)
        rootPos = np.sqrt((-P + sqrtDisc) / 2.0)
        rootNeg = np.sqrt((-P - sqrtDisc) / 2.0)
        temp = b / (4.*a)
        x.append(rootPos - temp)
        x.append(-rootPos - temp)
        x.append(rootNeg - temp)
        x.append(-rootNeg - temp)
    else:
        beta = cardano([-Q*Q, P*P - 4.*R, 2.*P, 1])
        beta = beta.x[0]
        root = np.sqrt(beta - 2.*(P + beta + Q / np.sqrt(beta)))
        term1 = (-np.sqrt(beta) + root) / 2.
        term2 = (-np.sqrt(beta) - root) / 2.
        temp = b / (4.*a)
        x.append(term1 - temp)
        x.append(-term2 - temp)
        x.append(term2 - temp)
        x.append(-term1 - temp)
    stopwatch.stop()
    return Result(None, None, None, None, x, [horner(p, z)[0] for z in x], None, stopwatch.get_elapsed_time,
                  "Quartic Formula", None)


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
        z = complex(z, z)
        error = parameter.tol + 1
        if m == n - 1:
            k += 1
            z = - p_aux[0] / p_aux[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                z_old = z
                p_z, q_aux = horner(p_aux, z)
                q_z = horner(q_aux, z)[0]
                if abs(q_z) > eps:
                    z = z_old - p_z / q_z
                    error = max(abs(z - z_old), abs(p_z))
                else:
                    error = 0
                    z = z_old
        # refinement step
        if parameter.ref == True:
            k_ref = 0
            z_ref = z
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
            z = z_ref
            k_ref_array.append(k_ref)
        z_array.append(z)
        p_z, p_aux = horner(p_aux, z)
        p_array.append(p_z)
        k_array.append(k)
    stopwatch.stop()
    return Result(k_array, parameter.maxit, k_ref_array, parameter.refmax, z_array, p_array, 
            parameter.tol, stopwatch.get_elapsed_time, "NEWTON-HORNER", term_flag)


def mullerhorner(p, x, parameter):
    """f
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