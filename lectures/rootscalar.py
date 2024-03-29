""" Module for scalar root-finding algorithms. """

# third party import
import numpy as np

# local import
from timer import timer # type: ignore
from rootspoly import newtonhorner, horner
from rootspoly import param as rootspoly_param
from polyinterp import undeterminedcoeff

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
    

    def __init__(self, maxit=1000, tol=np.finfo("float").eps, abstol=0.9,
                 funtol=0.1):
        """
        Class initialization.
        """
        self.abstol = abstol
        self.funtol = funtol
        self.tol = tol
        self.maxit = maxit

# Constant for Inexact Newton-Raphson Method
eps = np.finfo("float").eps
root_eps = np.sqrt(eps)

def bisection(f: callable, a: float, b: float, parameter: param) -> Result:
    """
    Bisection method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    a : float
        left endpoint
    b : float
        right endpoint
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = b - a
    fa, fb = f(a), f(b)
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

def chord(f: callable, a: float, b: float, x: float, parameter: param) -> Result:
    """
    Chord method for approximating a solution of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    a : float
        left endpoint
    b : float
        right endpoint
    x : float
        initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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

def secant(f: callable, x0: float, x1: float, parameter: param) -> Result:
    """
    Secant method for approximating a solution of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    x0 : float
        first of the initial iterates
    x1 : float
        second of the initial iterates
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    f0, f1 = f(x0), f(x1)
    k = 1
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        # Compute slope q
        q = (f1 - f0) / (x1 - x0)
        temp = x1
        x1 = x1 - f1 / q
        x0 = temp
        f0, f1 = f1, f(x1)
        error = parameter.abstol * abs(x1 - x0) + parameter.funtol * abs(f1)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x1, f1, error, parameter.tol,
        stopwatch.get_elapsed_time, "SECANT", term_flag)

def regfalsi(f: callable, x0: float, x1: float, parameter: param) -> Result:
    """
    Regula Falsi method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    x0 : float
        first of the initial iterates
    x1 : float
        second of the initial iterates
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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
        xc, fc = x_array[k], f_array[k]
        j = k - 1
        xj, fj = x_array[j], f_array[j]
        while fj * fc >= 0 and j > 1:
            j -= 1
            xj, fj = x_array[j], f_array[j]
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

def newtonraphson(f: callable, df: callable, x: float, options: str,
                  parameter: param) -> Result:
    """
    Newton-Raphson method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    df : callable
        derivative of f, None if df is to be approximated 
    x : float
        initial iterate
    options : str
        indicate which finite differences to be used to approximate
        the derivative. Possible args are "forward", "backward", and "center"
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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
        if df != None:
            dfx = df(x)
            methodname = "NEWTON-RAPHSON"
        else:
            match options:
                case "forward":
                    dfx = (f(x + root_eps) - f(x)) / root_eps
                    methodname = "INEXACT FORWARD"
                case "backward":
                    dfx = (f(x) - f(x - root_eps)) / root_eps
                    methodname = "INEXACT BACKWARD"
                case "center":
                    dfx = (f(x + root_eps) - f(x - root_eps)) / (2.*root_eps)
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

def steffensen(f: callable, x: float, parameter: param) -> Result:
    """
    Steffensen method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root is to be computed
    x : float
        initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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

def fixpoint(g: callable, x: float, parameter: param) -> Result:
    """
    Fix-point method for approximating a solution of the
        scalar equation g(x) = x.

    Parameters
    ----------
    g : callable
        function whose fix point is to be computed
    x : float
        initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old, x = x, g(x)
        error = abs(x - x_old)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, g(x), error, parameter.tol,
        stopwatch.get_elapsed_time, "FIX POINT", term_flag)

def muller(f: callable, x0: float, x1: float, x2: float, parameter: param)\
        -> Result:
    """
    Muller method for approximating a solution of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    x0 : float
        first point of the initial iterates
    x1 : float
        second point of the initial iterate
    x2 : float
        third point of the initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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

def rootpolyinterp(f: callable, x: list, parameter: param) -> Result:
    """
    Generalization of Secant and Muller methods for approximating a solution
        of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    x : list
        initial iterates
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(x)
    xc = x[n - 1]
    error = parameter.tol + 1
    k = n - 1
    while error > parameter.tol and k < parameter.maxit:
        p = undeterminedcoeff(f, x)
        z = newtonhorner(p, xc, rootspoly_param()).z
        z = [_.real for _ in z if abs(_.imag) <= eps]
        xc = min(z)
        fx = f(xc)
        error = parameter.abstol*abs(xc - x[n - 1]) + parameter.funtol*abs(fx)
        for j in range(n - 1):
            x[j] = x[j + 1]
        x[n - 1] = xc
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, xc, fx, error, parameter.tol,
                  stopwatch.get_elapsed_time, "ROOTPOLYINTERP", term_flag)

def sidisecant(f: callable, x: list, parameter: param) -> Result:
    """
    Generalization of Newton methods for approximating a solution of the
        scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    x : list
        initial iterates
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(x)
    xc = x[n - 1]
    error = parameter.tol + 1
    k = n - 1
    while error > parameter.tol and k < parameter.maxit:
        p = undeterminedcoeff(f, x)
        px, dp = horner(p, xc)
        dpx = horner(dp, xc)[0]
        xc = xc - px/dpx
        fx = f(xc.real)
        error = parameter.abstol*abs(xc - x[n - 1]) + parameter.funtol*abs(fx)
        for j in range(n - 1):
            x[j] = x[j + 1]
        x[n - 1] = xc
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, xc, fx, error, parameter.tol,
                  stopwatch.get_elapsed_time, "SIDISECANT", term_flag)


def dekkerbrent(f: callable, a: float, b: float, parameter: param) -> Result:
    """
    Modified Dekker-Brent method for approximating a solution of the scalar
        equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    a : float
        first point of the initial iterates
    b : float
        second point of the initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    delta = parameter.tol + 2. * eps * abs(b)
    fa, fb = f(a), f(b)
    k = 0
    if fa == 0:
        b = a
        error = 0
    if fb == 0:
        error = 0
    # Interchange a and b as necessary
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    # main while loop
    while error > delta and k < parameter.maxit:
        b_old = b
        
        # Linear Interpolation
        if a == b and b == c and a == c:
            z = b - fb * (b - a) / (fb - fa)
        # Quadratic Interpolation
        else:
            f01 = (fa - fc) / (a - c)
            f12 = (fb - fa) / (b - a)
            f012 = (f12 - f01) / (b - c)
            w = f12 + f012*(b - a)
            alpha = w*w - 4.*fb*f012
            if alpha >= 0:
                d = w + np.copysign(1., w)*np.sqrt(alpha)
                z = b - 2.*fb/d
            else:
                z = b - fb/f12
            
        m = (c - b) / 2.
        if min(b, b + m) < z < max(b, b + m):
            b_temp = z
        else:
            b_temp = b + m
        if abs(b - b_temp) > delta:
            b = b_temp
        else:
            b = b + delta*np.sign(m)
            
        a, fa = b_old, fb
        fb = f(b)
        if fa * fb < 0:
            c = a
            fc = fa
        if abs(fc) < abs(fb) < 0:
            a, b, c = b, c, a
            fa, fb, fc = fb, fc, fa
        delta = parameter.tol + 2. * eps * abs(b)
        error = abs(m - b)
        k += 1
        if abs(fb) <= parameter.tol:
            break
    if error > delta and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, b, fb, error, parameter.tol,
        stopwatch.get_elapsed_time, "MODIFIED DEKKER-BRENT", term_flag)

def aitken(g: callable, x: float, parameter: param) -> Result:
    """
    Aitken method for accelerating fix point method approximating a solution
        of the scalar equation f(x) = 0.

    Parameters
    ----------
    g : callable
        function whose fix point is to be computed
    x : float
        initial iterate
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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

def modnewton(f: callable, df: callable, x: float, m: int, parameter: param) -> Result:
    """
    Modified Newton-Raphson method for accelerating Newton-Raphson method
    approximating a solution of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    df : callable
        derivative of f
    x : float
        initial iterate
    m : int
        multiplicity of the root
    parameter : param
        parameters of the method

    Returns
    -------
    rootscalar::Result
        class Result of the method
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

def adaptivenewton(f: callable, df: callable, x: float, parameter: param,
                   m0: float, alpha: float, _lambda: float) -> Result:
    """
    Adaptive Newton method for accelerating fix point method approximating
        a solution of the scalar equation f(x) = 0.

    Parameters
    ----------
    f : callable
        function whose root point is to be computed
    df : callable
        derivative of f
    x : float
        initial iterate
    parameter : param
        parameters of the method
    m0 : float
        initial guess for the multiplicity
    alpha : float
        upper bound to the difference of successive multiplicity
    _lambda : float
        upper bound to alpha
        
    Returns
    -------
    Result
        class Result of the method
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    error_abs = error
    fx = f(x)
    m = m0
    LAMBDA = 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = x - m * fx / df(x)
        fx = f(x)
        error_temp = error_abs
        error_abs = abs(x - x_old)
        error = parameter.abstol*error_abs + parameter.funtol*abs(fx)
        LAMBDA_old = LAMBDA
        LAMBDA = error_abs / error_temp
        if abs(LAMBDA - LAMBDA_old) < alpha and LAMBDA > _lambda:
            m_temp = 1 / abs(1 - LAMBDA)
            if m < m_temp:
                m = m_temp
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "ADAPTIVE NEWTON", term_flag)