from linalg import vector, matrix, param, eps, newton, broyden

# Define system of equations f
def f(x: vector) -> vector:
    ans = vector()
    ans.append(x[0]**2. + x[1]**2. - 4.)
    ans.append((x[0] - 1.)**2. + (x[1] + 1.)**2. - 1.)
    return ans

# Define Jacobian of f
def Df(x: vector) -> matrix:
    ans = matrix([[0]*2 for _ in range(2)])
    ans[0][0] = 2.*x[0]
    ans[0][1] = 2.*x[1]
    ans[1][0] = 2.*x[0] - 2.
    ans[1][1] = 2.*x[1] + 2.
    return ans

if __name__ == "__main__":
    parameter = param()
    parameter.tol = eps
    parameter.maxit = 100
    
    # Newton methods
    x = vector([3., 1.5])
    print("\n", newton(f, Df, x, parameter))
    x = vector([-1.5, -3.])
    print("\n", newton(f, Df, x, parameter))
    
    # Broyden methods
    x = vector([3., 1.5])
    Q = Df(x)
    print("\n", broyden(f, Q, x, parameter))
    x = vector([-1.5, -3.])
    Q = Df(x)
    print("\n", broyden(f, Q, x, parameter))