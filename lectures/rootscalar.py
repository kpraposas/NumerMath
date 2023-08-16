""" Module for scalar root-finding algorithms. """

# third party import
import numpy as np

# local import
from timer import timer


class Result:
    """
    Class for solution to scalar root-finding algorithms.

    Attributes
    ----------
    numit : int
        number of iterations
    maxit : int
        maximum number of iterations
    x : float
        approximate root or last iterate
    funval : float
        function value at x
    error : float
        last error computed by the method
    tol : float
        tolerance of the method
    elapsed_time : float
        number in seconds for the method to execute
    method_name : str
        name of the root-finding method
    termination_flag : str
        either 'Fail' or 'Success'
    """

    def __init__(self, numit, maxit, x, funval, error, tol, elapsed_time,
        method_name, termination_flag):
        """
        Class initialization
        """
        self.numit = numit
        self.maxit = maxit
        self.x = x
        self.funval = funval
        self.error = error
        self.tol = tol
        self.elapsed_time = elapsed_time
        self.method_name = method_name
        self.termination_flag = termination_flag

    def __str__(self):
        """
        Class string representation.
        """
        list_str_repr = [
            "ROOT FINDER:                    {}\n".format(self.method_name),
            "APPROXIMATE ROOT/ LAST ITERATE: {:.16f}\n".format(self.x),
            "TERMINATION:                    {}\n".format(self.termination_flag),
            "FUNCTION VALUE:                 {:.16e}\n".format(self.funval),
            "ERROR:                          {:.16e}\n".format(self.error),
            "TOLERANCE:                      {:.16e}\n".format(self.tol),
            "NUM ITERATIONS:                 {}\n".format(self.numit),
            "MAX ITERATIONS:                 {}\n".format(self.maxit),
            "ELAPSED TIME:                   {:.16e} seconds\n".format(self.elapsed_time)
            ]
        return "".join(list_str_repr)
    
    
    def print_settings(self):
        """
        Print options and parameters of an instance of the Result
        """


class param():
    """
    Class for parameters in scalar root-finding algorithms.

    Attributes
    ----------
    tol : float
        tolerance of the method, default is machine epsilon
    maxit : int
        maximum number of iterations, default is 1000
    abstol : float
        weight of the absolute error of the root approximate
    funtol : float
        weight of the absolute function value of the root approximate
    m0 : int
        initial guess of multiplicity of roots
    alpha : float
        allowable change in values of ratios of consecutive errors
        (see Aitken's Method in the Lecture Notes)
    Lambda : float
        lower bound of allowed ratio of consecutive error
    """
    

    def __init__(self, maxit=1000, tol=np.finfo("float").eps, abstol=0.9, funtol=0.1,
                 m0=1, alpha=1e-3, Lambda=1e-3):
        """
        Class initialization.
        """
        self.abstol = abstol
        self.funtol = funtol
        self.tol = tol
        self.maxit = maxit
        self.m0 = m0
        self.alpha = alpha
        self.Lambda = Lambda


# Options Dictionary Initialization
options = dict()
"""
    Dictionary for scalar root finding algorithms
        method      : bisection, chord, secant, regfalsi, newton, steffensen,
                        fixpoint, muller, rootpolyinterp, sidisecant,
                        dekker, aitken, modnewton, and adaptive newton
            methods for scalar root finding, newton method as default
        inexact     : forward, backward, center
            inexact approximation of derivative for inexact newton method
            center as default
            
"""
options = {
    "method" : "newton",
    "inexact" : "center"
}

def rootscalar(f, df, a, b, x, m, options, parameter):
    """
    Attributes
    ----------
        f : callable
            scalar-valued function
        df : callable
            derivative of f
        a : float
            left endpoint
        b : float
            right endpoint
        x : float or list of floats
            initial iterates or list of initial iterates
        m : int
            multiplicity of root 
        options : dict
            dictionary of methods to solve for scalar root
        parameters : class
            parameters to be passed in the specified method
            
    Returns
    -------
        rootscalar.Result : class
    """
    if options["method"] == "bisection":
        return bisection(f, a, b, parameter)
    if options["method"] == "chord":
        return chord(f, a, b, x, parameter)
    if options["method"] == "secant":
        return secant(f, x[0], x[1], parameter)
    if options["method"] == "regfalsi":
        return regfalsi(f, x[0], x[1], parameter)
    if options["method"] == "newton":
        return newtonraphson(f, df, x, options, parameter)
    if options["method"] == "steffensen":
        return steffensen(f, x, parameter)
    if options["method"] == "fixpoint":
        return fixpoint(f, x, parameter)
    if options["method"] == "muller":
        return muller(f, x[0], x[1], x[2], parameter)
    if options["method"] == "dekker":
        return dekkerbrent(f, a, b, parameter)
    if options["method"] == "aitken":
        return aitken(f, x, parameter)
    if options["method"] == "modnewton":
        return modnewton(f, df, x, m, parameter)
    if options["method"] == "adaptivenewton":
        return adaptivenewton(f, df, x, parameter)
    

def bisection(f, a, b, parameter):
    """
    Bisection method for approximating a solution of the scalar equation
        f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = b - a
    fa = f(a)
    fb = f(b)
    k = 0
    # Check for endpoints
    if fa == 0:
        c = a
        error = 0
    if fb == 0:
        c = b
        error = 0
    # Check if endpoints' sign differ
    if fa * fb > 0:
        print("Method Fails")
        return None
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        c = (a + b) / 2.
        fc = f(c)
        if fa * fc > 0:
            a = c
            fa = fc
        else:
            b = c
        error = abs(b-a)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, c, fc, error, parameter.tol,
        stopwatch.get_elapsed_time, "BISECTION", term_flag)


def chord(f, a, b, x, parameter):
    """
    Chord method for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start() 
    term_flag = "Success"
    error = parameter.tol + 1
    # Compute fixed slope q
    q = (f(b) - f(a)) / (b - a)
    fx = f(x)
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = x - fx / q
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "CHORD", term_flag)


def secant(f, x0, x1, parameter):
    """
    Secant method for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    f0 = f(x0)
    f1 = f(x1)
    k = 1
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        # Compute slope q
        q = (f1 - f0) / (x1 - x0)
        temp = x1
        x1 = x1 - f1 / q
        x0 = temp
        f0 = f1
        f1 = f(x1)
        error = parameter.abstol * abs(x1 - x0) + parameter.funtol * abs(f1)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x1, f1, error, parameter.tol,
        stopwatch.get_elapsed_time, "SECANT", term_flag)


def regfalsi(f, x0, x1, parameter):
    """
    Regula Falsi method for approximating a solution of the scalar equation
        f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    # Lists of iterates and their correponding function values
    x_array = [x0, x1]
    f_array = [f(x0), f(x1)]
    k = 1
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        xc = x_array[k]
        fc = f_array[k]
        j = k - 1
        xj = x_array[j]
        fj = f_array[j]
        while fj * fc >= 0 and j > 1:
            j -= 1
            xj = x_array[j]
            fj = f_array[j]
        q = (fc - fj) / (xc - xj)
        x = xc - fc / q
        x_array.append(x)
        f_array.append(f(x))
        error = parameter.abstol * abs(x - xc) + parameter.funtol * abs(f(x))
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, f(x), error, parameter.tol,
        stopwatch.get_elapsed_time, "REGULA FALSI", term_flag)


def newtonraphson(f, df, x, options, parameter):
    """
    Newton-Raphson method for approximating a solution of the scalar equation
        f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    fx = f(x)
    root_eps = np.sqrt(np.finfo(float).eps)
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        if df != None:
            dfx = df(x)
            methodname = "NEWTON-RAPHSON"
        elif options["inexact"] == "forward":
            dfx = (f(x + root_eps) - f(x)) / root_eps
            methodname = "INEXACT FORWARD"
        elif options["inexact"] == "backward":
            dfx = (f(x) - f(x - root_eps)) / root_eps
            methodname = "INEXACT BACKWARD"
        elif options["inexact"] == "center":
            dfx = (f(x + root_eps) - f(x - root_eps)) / (2.0*root_eps)
            methodname = "INEXACT CENTER"
        x = x - fx / dfx
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, methodname, term_flag)


def steffensen(f, x, parameter):
    """
    Steffensen method for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    fx = f(x)
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        q = (f(x + fx) - fx) / fx
        x = x - fx / q
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "STEFFENSEN", term_flag)


def fixpoint(g, x, parameter):
    """
    Fix-point method for approximating a solution of the scalar equation g(x) = x.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = g(x)
        error = abs(x - x_old)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, g(x), error, parameter.tol,
        stopwatch.get_elapsed_time, "FIX POINT", term_flag)


def muller(f, x0, x1, x2, parameter):
    """
    Muller method for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    f0 = f(x0)
    f1 = f(x1)
    f2 = f(x2)
    k = 2
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        f01 = (f1 - f0) / (x1 - x0)
        f12 = (f2 - f1) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + f012 * (x2 - x1)
        alpha = w * w - 4. * f2 * f012
        if alpha >= 0:
            d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
            x = x2 - 2. * f2 / d
        else:
            x = x2 - f2 / f12
        x0 = x1
        x1 = x2
        x2 = x
        f0 = f1
        f1 = f2
        f2 = f(x)
        error = parameter.abstol * abs(x2 - x1) + parameter.funtol * abs(f2)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x2, f2, error, parameter.tol,
        stopwatch.get_elapsed_time, "MULLER", term_flag)


def rootpolyinterp(f, x, parameter):
    """
    Generalization of Secant and Muller methods for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(x)
    error = parameter.tol + 1
    k = n - 1
    while error > parameter.tol and k < parameter.maxit:
        pass
        # Calculate polynomial p with args ((x0, f0), (x1, f1), ..., (xn-1, fn-1))
        # Compute real roots z1, z2, ..., zl of p, l <= n - 1
        # Choose s s.t. |z[s]| = min(z).index
        # x = z[s]
        # error = parameter.abstol * abs(x - x[n-1]) + parameter.funtol * abs(fx)
        # for j in range(n - 1):
        #     x[j] = x[j + 1]
        # x[n - 1] = x
        # k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return
        # Result(k, parameter.maxit, x2, f2, error, parameter.tol, stopwatch.get_elapsed_time, "MULLER", term_flag)


def sidisecant(f, x, parameter):
    """
    Generalization of Secant, Muller as well as Newton methods for approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(x)
    error = parameter.tol + 1
    k = n - 1
    while error > parameter.tol and k < parameter.maxit:
        pass
        # Calculate polynomial p with args ((x0, f0), (x1, f1), ..., (xn-1, fn-1))
        # x = x[n - 1] - f[n - 1] / dp(x[n - 1])
        # error = parameter.abstol * abs(x - x[n - 1]) + parameter.funtol * abs(fx)
        # for j in range(n - 2):
        #     x[j] = x[j + 1]
        # x[n - 1] = x
        # k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return
        # Result(k, parameter.maxit, x2, f2, error, parameter.tol, stopwatch.get_elapsed_time, "MULLER", term_flag)


def dekkerbrent(f, a, b, parameter):
    """
    Modified Dekker-Brent method for approximating a solution of the scalar
        equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    eps = np.finfo(float).eps
    delta = parameter.tol + 2. * eps * abs(b)
    fa = f(a)
    fb = f(b)
    k = 0
    if fa == 0:
        b = a
        error = 0
    if fb == 0:
        error = 0
    a_old = a
    f_a_old = fa
    # Interchange a and b as necessary
    if abs(fa) < abs(fb):
        a = b
        fa = fb
        b = a_old
        fb = f_a_old
    c = a
    fc = fa
    # main while loop
    while error > delta and k < parameter.maxit:
        b_old = b
        # Quadriatic Interpolation
        if a != b and a != c and b != c:
            f01 = (fa - fc) / (a - c)
            f12 = (fb - fa) / (b - a)
            f012 = (f12 - f01) / (b - c)
            w = f12 + f012 * (b - a)
            alpha = w * w - 4. * fb * f012
            if alpha >= 0:
                d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
                z = b - 2. * fb / d
            else:
                z = b - fb / f12
        # Linear Interpolation
        else:
            z = b - fb * (b - a) / (fb - fa)
        m = (a + b) / 2.
        if min(b, m) < z < max(b, m):
            b_temp = z
        else:
            b_temp = m
        if abs(b - b_temp) > delta:
            b = b_temp
        else:
            b = b + delta * np.sign(m)
        a = b_old
        fa = fb
        fb = f(b)
        if fa * fb < 0:
            c = a
            fc = fa
        if fc * fb < 0:
            a = b
            fa = fb
            b = c
            fb = fc
            c = a
            fc = fa
        delta = parameter.tol + 2. * eps * abs(b)
        error = abs(m - b)
        if abs(fb) <= parameter.tol:
            break
        k += 1
    if error > delta and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, b, fb, error, parameter.tol,
        stopwatch.get_elapsed_time, "MODIFIED DEKKER-BRENT", term_flag)


def aitken(g, x, parameter):
    """
    Aitken method for accelerating fix point method approximating a solution
        of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x1 = g(x)
        x2 = g(x1)
        x = x2 - (x2 - x1)**2 / (x2 - 2.*x1 + x)
        error = abs(x - x_old)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, g(x), error, parameter.tol,
        stopwatch.get_elapsed_time, "AITKEN", term_flag)


def modnewton(f, df, x, m, parameter):
    """
    Modified Newton-Raphson method for accelerating Newton-Raphson method
    approximating a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    fx = f(x)
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = x - m * fx / df(x)
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "MODIFIED NEWTON", term_flag)


def adaptivenewton(f, df, x, parameter):
    """
    Adaptive Newton method for accelerating fix point method approximating
        a solution of the scalar equation f(x) = 0.
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    error_abs = error
    fx = f(x)
    m = parameter.m0
    lmdb = 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = x - m * fx / df(x)
        fx = f(x)
        error_abs_old = error_abs
        error_abs = abs(x - x_old)
        error = parameter.abstol * error_abs + parameter.funtol * abs(fx)
        lmdb_old = lmdb
        lmdb = error_abs / error_abs_old
        if abs(lmdb - lmdb_old) < parameter.alpha and lmdb > parameter.Lambda:
            m_temp = 1 / abs(1 - lmdb)
            if m < m_temp: m = m_temp
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "ADAPTIVE NEWTON", term_flag)


def usualpolyeval(pn, z):
    """
    Usual polynomial evaluation of a polynomial pn with coefficients
        [a0, a1, ..., an] evaluated at a complex number z.

    Parameters:
    -----------
    pn : list
        coefficients of pn
    z : float
        valuation of pn

    Returns
    -------
    v : float
        function value of pn at z
    """
    v = pn[0]
    x = 1
    for j in range (1, len(pn)):
        x = x * z
        v += pn[j] * x
    return v


def horner(pn, z):
    """
    Nested multiplication polynomial evaluation of a polynomial pn with
        coefficients [a0, a1, ..., an] evaluated at a complex number z and
        remainder [b1, b2, .., bn].

    Parameters:
    -----------
    pn : list
        coefficients of pn
    z : float
        valuation of pn

    Returns
    -------
    b0 : float
        function value of pn at z
    [b1, b2, ..., bn] : list
        remainder of pn evaluated at z
    """
    n = len(pn) - 1
    q = [0] * (n)
    q[n - 1] = pn[n]
    for k in range(n - 2, -1, -1):
        q[k] = pn[k + 1] + q[k + 1] * z
    q0 = pn[0] + q[0] * z
    return q0, q