from cmath import sqrt

def eval(a: float, b: float, c: float, d: float, z: complex) -> complex:
    """
    Evaluates a cubic polynomial of the form ax^3 + bx^2 + cx + d by z

    Parameters
    ----------
    a : float
        leading coefficient
    b : float 
        coefficient of quadratic term
    c : float
        coefficient of linear term
    d : float
        coefficient of constant term
    z : complex
        number to evaluate the cubic equation
        
    Returns
    -------
    complex
        value of az^3 + bz^2 + cz + d
    """
    val = a*z + b
    val = val*z + c
    val = val*z + d
    return val

def cardano(a: float, b: float, c: float, d: float) -> list:
    """
    Computes for the roots of the cubic polynomial ax^3 + bx^2 + cx + d

    Parameters
    ----------
    a : float
        leading coefficient
    b : float 
        coefficient of quadratic term
    c : float
        coefficient of linear term
    d : float
        coefficient of constant term
    
    Returns
    -------
    list
        roots of the cubic polynomial
    """
    j = 1j
    q = (3.*a*c - b*b) / (9.*a*a)
    r = (9.*a*b*c - 27.*a*a*d - 2.*b*b*b) / (54.*a*a*a)
    D = q*q*q + r*r
    u = r + sqrt(D)
    v = r - sqrt(D)
    if u.real >= 0.:
        s = u**(1./3.)
    else:
        s = -(-u)**(1./3.)
    if v.real >= 0.:
        t = v**(1./3.)
    else:
        t = -(-v)**(1./3.)
    x = []
    x.append(s + t - b/(3.*a))
    x.append(-0.5*(s + t) - b/(3.*a) + sqrt(3.)*j*(s - t)/2.)
    x.append(-0.5*(s + t) - b/(3.*a) - sqrt(3.)*j*(s - t)/2.)
    return x
    

if __name__ == "__main__":
    a, b, c, d = 2., 3., 4., 5.
    x = cardano(a, b, c, d)
    print("The solutions of the cubic equation f(x) = ", a, "x^3 + ", b,
        "x^2 + ", c, "x + ", d, " = 0 are:")
    for i in range(len(x)):
        print("\tx = {:8e} \tf(x) = {}".format(x[i], eval(a, b, c, d, x[i])))