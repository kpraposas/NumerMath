from cmath import sqrt

def eval(a:float, b:float, c:float, z:complex) -> complex:    
    """
    Evaluates a quadratic polynomial of the form ax^2+ bx + c by z

    Parameters
    ----------
    a : float
        leading coefficient
    b : float 
        coefficient of linear term
    c : float
        coefficient of constant term
    z : complex
        number to evaluate the quadratic equation
        
    Returns
    -------
    complex
        value of az^2+ bz + c
    """
    return (a*z + b)*z + c

def quadform(a:float, b:float, c:float) -> list:
    """
    Computes for the roots of the quadratic polynomial ax^2 + bx + c

    Parameters
    ----------
    a : float
        leading coefficient
    b : float 
        coefficient of linear term
    c : float
        coefficient of constant term
    x : list
        storage for roots
    
    Returns
    -------
    list
        roots of the quadratic polynomial
    """
    sqrt_disc = sqrt(b*b - 4.*a*c)
    x = []
    x.append((-b + sqrt_disc) / (2.*a))
    x.append((-b - sqrt_disc) / (2.*a))
    return x

if __name__ == "__main__":
    a, b, c = 2., 3., 4.
    x = quadform(2, 3, 4)
    print("The solutions of the quadratic equation f(x) = ", a, "x^2 + ", b,
            "x + ", c, " = 0 are:")
    for i in range(len(x)):
        print("\tx = {:8e} \tf(x) = {}".format(x[i], eval(a, b, c, x[i])))
    