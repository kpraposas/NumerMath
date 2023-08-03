""" Sample implementation of solving triangular systems algorithms. """

import linearalgebra

n = 1000
L, U = [[0]*n for i in range(n)], [[0]*n for i in range(n)]
k = 1
for i in range(n):
    for j in range(i + 1):
        L[i][j] = (k/n)**2
        k += 1
        
for i in range(n - 1, -1, -1):
    for j in range(n - 1, i - 1, -1):
        U[i][j] = (k/n)**2
        k += 1
x = [1]*n

result = []
b = linearalgebra.matrixvectorSAXPY(L, x)
forwardsubrow = linearalgebra.forwardsubrow(L, b).x
forwardsubcol = linearalgebra.forwardsubcol(L, b).x
result.append(forwardsubrow)
result.append(forwardsubcol)
b = linearalgebra.matrixvectorSAXPY(U, x)
backwardsubrow = linearalgebra.backwardsubrow(U, b).x
backwardsubcol = linearalgebra.backwardsubcol(U, b).x
result.append(backwardsubrow)
result.append(backwardsubcol)

print('-'*77)
print("{}\t\t\t{}\t\t\t{}".format("METHOD", "MAX ABS ERROR", "RESIDUAL MAX NORM"))
print('-'*77)
for i in range(4):
    method_name = ["ForwardSubRow", "ForwardSubCol", "BackwardSubRow", "BackwardSubCol"]
    max_abs = max(linearalgebra.vectordiff(result[i], x))
    if i < 2:
        b = linearalgebra.matrixvectorSAXPY(L, x)
        residual_max = max(linearalgebra.vectordiff(b, linearalgebra.matrixvectorSAXPY(L, result[i])))
    else:
        b = linearalgebra.matrixvectorSAXPY(U, x)
        residual_max = max(linearalgebra.vectordiff(b, linearalgebra.matrixvectorSAXPY(U, result[i])))
    print("{}\t\t{:.15e}\t\t{:.15e}".format(method_name[i], max_abs, residual_max))
print('-'*77)



