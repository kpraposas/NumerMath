from quadform import quadform
from cardano import cardano
from cmath import sqrt

def eval(a: float, b: float, c: float, d: float, e: float, z: complex) -> complex:
    """
    Evaluates a quartic polynomial of the form ax^4 + bx^3 + cx^2 + dx + e by z

    Parameters
    ----------
    a : float
        leading coefficient
    b : float
        coefficient of cubic term
    c : float 
        coefficient of quadratic term
    d : float
        coefficient of linear term
    e : float
        coefficient of constant term
    z : complex
        number to evaluate the quartic equation
        
    Returns
    -------
    complex
        value of az^4 + bz^3 + cz^2 + dz + e
    """
    val = a*z + b
    val = val*z + c
    val = val*z + d
    val = val*z + e
    return val

def quarform(a: float, b: float, c: float, d: float, e: float) -> list:
    """
    Computes for the roots of the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e

    Parameters
    ----------
    a : float
        leading coefficient
    b : float
        coefficient of cubic term
    c : float 
        coefficient of quadratic term
    d : float
        coefficient of linear term
    e : float
        coefficient of constant term
    
    Returns
    -------
    list
        roots of the quartic polynomial
    """
    A, B, C, D = b/a, c/a, d/a, e/a
    p = B - 3.*A*A/8.
    q = C - A*B/2. + A*A*A/8.
    r = D - A*C/4. + A*A*B/16. - 3.*A*A*A*A/256.
    t = A/4.
    
    x = []
    
    if D == 0.:
        x.append(0.)
        roots = cardano(1., A, B, C)
        x.append(roots[0])
        x.append(roots[1])
        x.append(roots[2])
        return x
    
    if q == 0.:
        roots = quadform(1., p, r)
        for i in range(2):
            sqrt_root = sqrt(roots[i])
            x.append(sqrt_root - t)
            x.append(-sqrt_root - t)
        return x
    
    beta = cardano(1., 2.*p, p*p - 4.*r, -q*q)
    beta0 = beta[1]
    sqrt_beta0 = sqrt(beta0)
    casePos = sqrt(beta0 - 2.*(p + beta0 + q/sqrt_beta0))
    caseNeg = sqrt(beta0 - 2.*(p + beta0 - q/sqrt_beta0))
    x.append(0.5*(sqrt_beta0 + casePos) - t)
    x.append(0.5*(-sqrt_beta0 + caseNeg) - t)
    x.append(0.5*(sqrt_beta0 - casePos) - t)
    x.append(0.5*(-sqrt_beta0 - casePos) - t)
    return x


if __name__ == "__main__":
    coeff1 = [2., 3., 4., 5., 6.]
    coeff2 = [1., -2., 3., 4., 0.]
    coeff3 = [4., 0., -9., 0., 2.]
    coeffmat = [coeff1, coeff2, coeff3]
    
    roots = []
    for i in range(3):
        roots.append(quarform(coeffmat[i][0], coeffmat[i][1], coeffmat[i][2],
                            coeffmat[i][3], coeffmat[i][4]))
        print("The solutions of the quartic equation f(x) = ", coeffmat[i][0],
              "x^4 + ", coeffmat[i][1], "x^3 + ", coeffmat[i][2], "x^2 + ",
                            coeffmat[i][3], "x + ", coeffmat[i][4], " = 0 are:")
        for j in range(4):
            print("\tx = {:8e}\tf(x) = {:8e}".format(roots[i][j],
                eval(coeffmat[i][0], coeffmat[i][1], coeffmat[i][2],
                    coeffmat[i][3], coeffmat[i][4], roots[i][j])))