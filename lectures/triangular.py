""" Sample implementation of solving triangular systems algorithms. """

from linalg import vector, matrix, forwardsubrow, forwardsubcol,\
    backwardsubrow, backwardsubcol

n = 1000
# Construct lower triangular matrix L and upper triangular matrix U
L, U = [[0]*n for _ in range(n)], [[0]*n for _ in range(n)]
k = 1
for i in range(n):
    for j in range(i + 1):
        L[i][j] = ((k+1.)/n)**2
        k += 1
L = matrix(L)
        
for i in range(n - 1, -1, -1):
    for j in range(n - 1, i - 1, -1):
        U[i][j] = ((k+1.)/n)**2
        k += 1
U = matrix(U)
        
x = vector([1]*n)

result = []
b = L*x
forwardsubrow = forwardsubrow(L, b)
forwardsubcol = forwardsubcol(L, b)
result.append(forwardsubrow)
result.append(forwardsubcol)

b = U*x
backwardsubrow = backwardsubrow(U, b)
backwardsubcol = backwardsubcol(U, b)
result.append(backwardsubrow)
result.append(backwardsubcol)

print('-'*76)
print("{}\t\t{}\t\t{}\t{}".format("METHOD", "MAX ABS ERROR", "RESIDUAL MAX NORM",
                              "ELAPSED TIME"))
print('-'*76)
for i in range(4):
    error = (x - result[i].x).norm("max")
    print("{}\t{:.15e}\t{:.15e}\t{:.6f} sec".format(result[i].method_name,
                                                    error,
                                                    result[i].residual_max,
                                                    result[i].elapsed_time))
print('-'*76)