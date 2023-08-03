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
    

class Result_Multiple:
    """
    Class for mutiple solutions to scalar root-finding algorithms.

    Attributes
    ----------
    numit : list
        number of iterations for each root
    maxit : int
        maximum number of iterations
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

    def __init__(self, numit, maxit, x, funval, tol, elapsed_time,
        method_name, termination_flag):
        """
        Class initialization
        """
        self.numit = numit
        self.maxit = maxit
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
        list_str_repr = [
            "ROOT FINDER:                    {}\n".format(self.method_name),
            "TERMINATION:                    {}\n".format(self.termination_flag),
            "TOLERANCE:                      {:.16e}\n".format(self.tol),
            "MAX ITERATIONS:                 {}\n".format(self.maxit),
            "ELAPSED TIME:                   {:.16e} seconds\n".format(self.elapsed_time),
            "-"*85,
            "\nREAL PART\t\t\tIMAG PART\t\t\t|FUNVAL|\tNUMIT\n",
            "-"*85,
        ]
        for i in range(len(self.x)):
            list_str_repr.append("{:.16f}{:.16f}{:.16f}{}\n".format(self.x[i].real, self.x[i].imag,
                    self.funval[i], self.numit[i]))
            list_str_repr.append("-"*85)
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

    def __init__(self, maxit=1000, tol=np.finfo("float").eps, abstol=0.9,
                 funtol=0.1, refmax=100, reftol=1e-3, ref=1):
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


def bisection(f, a, b, parameter):
    """
    Bisection method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    a : float
        left endpoint
    b : float
        right endpoint
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = b - a
    fa = f(a)
    fb = f(b)
    k = 0
    if fa == 0:
        c = a
        error = 0
    if fb == 0:
        c = b
        error = 0
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

    Parameters:
    -----------
    f : callable
        scalar-valued function
    a : float
        left endpoint
    b : float
        right endpoint
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start() 
    term_flag = "Success"
    error = parameter.tol + 1
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

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x0 : float
        first iterate
    x1 : float
        second iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        q = (f1 - f0) / (x1 - x0)
        x_temp = x1
        x1 = x1 - f1 / q
        x0 = x_temp
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

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x0 : float
        first iterate
    x1 : float
        second iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    x_array = [x0, x1]
    f_array = [f(x0), f(x1)]
    k = 1
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        xc = x_array[k]
        fc = f_array[k]
        k_tilde = k - 1
        x_tilde = x_array[k_tilde]
        f_tilde = f_array[k_tilde]
        while f_tilde * fc >= 0 and k_tilde > 1:
            k_tilde -= 1
            x_tilde = x_array[k_tilde]
            f_tilde = f_array[k_tilde]
        q = (fc - f_tilde) / (xc - x_tilde)
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


def newtonraphson(f, df, x, parameter):
    """
    Newton-Raphson method for approximating a solution of the scalar equation
        f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    df : callable
        exact derivative of f
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        x = x - fx / df(x)
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "NEWTON-RAPHSON", term_flag)


def inexactforward(f, x, parameter):
    """
    Inexact Forward Newton-Raphson method for approximating a solution of
        the scalar equation f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        df = (f(x + root_eps) - f(x)) / root_eps
        x = x - fx / df
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "INEXACT FORWARD", term_flag)


def inexactbackward(f, x, parameter):
    """
    Inexact Backward Newton-Raphson method for approximating a solution of
        the scalar equation f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        df = (f(x) - f(x - root_eps)) / root_eps
        x = x - fx / df
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "INEXACT BACKWARD", term_flag)


def inexactcenter(f, x, parameter):
    """
    Inexact Center Newton-Raphson method for approximating a solution of
        the scalar equation f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        df = (f(x + root_eps) - f(x - root_eps)) / (2*root_eps)
        x = x - fx / df
        fx = f(x)
        error = parameter.abstol * abs(x - x_old) + parameter.funtol * abs(fx)
        k += 1
    if error > parameter.tol and k == parameter.maxit:
        term_flag = "Fail"
    stopwatch.stop()
    return Result(k, parameter.maxit, x, fx, error, parameter.tol,
        stopwatch.get_elapsed_time, "INEXACT CENTER", term_flag)


def steffensen(f, x, parameter):
    """
    Steffensen method for approximating a solution of the scalar equation f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
    Fix-point method for approximating a solution of the scalar equation f(x) = 0.

    Parameters:
    -----------
    g : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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

    Parameters:
    -----------
    f : callable
        scalar-valued function
    x1 : float
        second iterate
    x2 : float
        first iterate
    x3 : float
        third iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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
        alpha = w * w - 4.0 * f2 * f012
        if alpha >= 0:
            d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
            x = x2 - 2.0 * f2 / d
        else: x = x2 - f2 / f12
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


# rootpolyinterp, sidi secant


def moddekkerbrent(f, a, b, parameter):
    """
    Modified Dekker-Brent method for approximating a solution of the scalar
        equation f(x) = 0.

    Parameters:
    -----------
    g : callable
        scalar-valued function
    a : float
        left endpoint
    b : float
        right endpoint
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    eps = np.finfo(float).eps
    delta = parameter.tol + 2 * eps * abs(b)
    fa = f(a)
    fb = f(b)
    k = 0
    if fa == 0:
        b = a
        error = 0
    if fb == 0: error = 0
    a_old = a
    f_a_old = fa
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
        else:
            z = b - fb * (b - a) / (fb - fa)
        m = (c - b) / 2.
        if min(b, b + m) < z < max(b, b + m): b_temp = z
        else: b_temp = b + m
        if abs(b - b_temp) > delta: b = b_temp
        else: b = b + delta * np.sign(m)
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
        error = abs(m)
        if abs(fb) <= parameter.tol: break
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

    Parameters:
    -----------
    g : callable
        scalar-valued function
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x_1 = g(x)
        x_2 = g(x_1)
        x = x_2 - (x_2 - x_1)**2 / (x_2 - 2.0*x_1 + x)
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

    Parameters:
    -----------
    f : callable
        scalar-valued function
    df : callable
        exact derivative of f
    x : float
        first iterate
    m : int
        multiplicity of root
    parameter : class 'rootscalar.param'
        parameters to be passed in the method

    Returns
    -------
    class 'rootscalar.Result'
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


def adaptivenewton(f, df, x, parameter, m0, alpha, Lambda_):
    """
    Adaptive Newton method for accelerating fix point method approximating
        a solution of the scalar equation f(x) = 0.

    Parameters:
    -----------
    f : callable
        scalar-valued function
    df : callable
        scalar-valued exact derivative of f
    x : float
        first iterate
    parameter : class 'rootscalar.param'
        parameters to be passed in the method
    m0 : float
        parameter m0 >= 1
    alpha, lambda_ : float
        parameter 0 < alpha < Lambda_ < 1

    Returns
    -------
    class 'rootscalar.Result'
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    error = parameter.tol + 1
    error_abs = error
    fx = f(x)
    m = m0
    Lambda = 1
    k = 0
    # main loop
    while error > parameter.tol and k < parameter.maxit:
        x_old = x
        x = x - m * fx / df(x)
        fx = f(x)
        error_abs_old = error_abs
        error_abs = abs(x - x_old)
        error = parameter.abstol * error_abs + parameter.funtol * abs(fx)
        Lambda_old = Lambda
        Lambda = error_abs / error_abs_old
        if abs(Lambda - Lambda_old) < alpha and Lambda > Lambda_:
            m_temp = 1 / abs(1 - Lambda)
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
    q = [0] * (n + 1)
    q[n] = pn[n]
    for k in range(n - 1, -1, -1):
        q[k] = pn[k] + q[k + 1] * z
    q0 = q[0]
    q.pop(0)
    return q0, q


def newtonhorner(pn, z, parameter):
    """
    Newton-Horner method for accelerating fix point method approximating a solution
        of a polynomial pn with coefficients [a0, a1, ..., an]

    Parameters:
    -----------
    pn : list
        coefficients of pn
    z : float
        valuation of pn
    parameter : class 'rootscalar.param'
         parameters to be passed in the method

    Returns
    -------
    z_array : list
        approximate roots of pn
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(pn) - 1
    z_array = []
    p_array = []
    k_array = []
    eps = np.finfo(float).eps
    p_nm = pn
    q_nm1 = []
    error_ref = parameter.tol * parameter.reftol
    # main loop
    for m in range(n):
        k = 0
        z = z + z * 1j
        error = parameter.tol + 1
        if m == n - 1:
            k += 1
            z = - p_nm[0] / p_nm[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                z_old = z
                p_z, q_nm1 = horner(p_nm, z)
                q_z = horner(q_nm1, z)[0]
                if abs(q_z) > eps:
                    z = z_old - p_z / q_z
                    error = max(abs(z - z_old), abs(p_z))
                else:
                    error = 0
                    z = z_old
                    break
        # refinement step
        if parameter.ref == 1:
            k_ref = 0
            z_ref = z
            error = parameter.tol + 1
            while error > error_ref and k_ref < parameter.refmax:
                k_ref += 1
                p_z, q_n1 = horner(pn, z_ref)
                q_z = horner(q_n1, z_ref)[0]
                if abs(q_z) > eps:
                    z_ref2 = z_ref - p_z / q_z
                    error = max(abs(z_ref - z_ref2), abs(p_z))
                    z_ref = z_ref2
                else:
                    error = 0
                    break
            z = z_ref
        z_array.append(z)
        p_z = horner(p_nm, z)[0]
        p_array.append(p_z)
        p_nm = horner(p_nm, z)[1]
        k_array.append(k)
    stopwatch.stop()
    return Result_Multiple(k_array, parameter.maxit, z_array, p_array, parameter.tol,
        stopwatch.get_elapsed_time, "NEWTONHORNER", term_flag)


def mullerhorner(pn, x0, x1, x2, parameter):
    """
    Muller-Horner method for accelerating fix point method approximating a solution
        of a polynomial pn with coefficients [a0, a1, ..., an]

    Parameters:
    -----------
    pn : list
        coefficients of pn
    x0, x1, x2 : float
        initial iterates
    parameter : class 'rootscalar.param'
         parameters to be passed in the method

    Returns
    -------
    z_array : list
        approximate roots of pn
    """
    stopwatch = timer()
    stopwatch.start()
    term_flag = "Success"
    n = len(pn) - 1
    z_array = []
    p_array = []
    k_array = []
    eps = np.finfo(float).eps
    p_nm = pn
    error_ref = parameter.tol * parameter.reftol
    # main loop
    for m in range(0, n):
        k = 0
        error = parameter.tol + 1
        z0 = x0 + 0j
        z1 = x1 + 0j
        z2 = x2 + 0j
        if m == n - 1:
            k += 1
            z = - p_nm[0] / p_nm[1]
        else:
            while error > parameter.tol and k < parameter.maxit:
                k += 1
                f0, f1, f2 = horner(p_nm, z0)[0], horner(p_nm, z1)[0], horner(p_nm, z2)[0]
                f01 = (f1 - f0) / (z1 - z0)
                f12 = (f2 - f1) / (z2 - z1)
                f012 = (f12 - f01) / (z2 - z0)
                w = f12 + (z2 - z1) * f012
                alpha = w * w - 4. * f2 * f012
                d = max(w - np.sqrt(alpha), w + np.sqrt(alpha))
                if abs(d) > eps:
                    z = z2 - 2. * f2 / d
                    error = max(abs(z - z2), abs(f2))
                else:
                    z = z2 - f2 / f12
                    error = 0
                    break
                z0, z1, z2 = z1, z2, z
        # refinement step
        if parameter.ref == 1:
            k_ref = 0
            z_ref = z
            error = parameter.tol + 1
            while error > error_ref and k_ref < parameter.refmax:
                k_ref +=  1
                p_z, q_n1 = horner(pn, z_ref)
                q_z = horner(q_n1, z_ref)[0]
                if abs(q_z) > eps:
                    z_ref2 = z_ref - p_z / q_z
                    error = max(abs(z_ref - z_ref2), abs(p_z))
                    z_ref = z_ref2
                else:
                    error = 0
                    break
            z = z_ref
        z_array.append(z)
        p_z = horner(p_nm, z)[0]
        p_array.append(p_z)
        p_nm = horner(p_nm, z)[1]
        k_array.append(k)
    stopwatch.stop()
    return Result_Multiple(k_array, parameter.maxit, z_array, p_array, parameter.tol,
        stopwatch.get_elapsed_time, "MULLERHORNER", term_flag)