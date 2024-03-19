# Sample implementations of solving linear systems numerically

import numpy as np
from linalg import *

def An(n):
    An = [[0]*n for i in range(n)]
    for i in range(n):
        An[i][i] = 2
    for i in range(n):
        for j in range(n):
            if i == j + 1 or j == i + 1:
                An[i][j] = -1
    for i in range(n):
        for j in range(n):
            An[i][j] *= (n + 1)**2
    return Matrix(An)

print(len(An(10).matrix[0][0]))

def bn(n):
    bn = []
    for i in range(1, n+1):
        bn.append(np.sin(i*np.pi/(n + 1)) * np.pi**2)
    return Matrix(bn)
    
# # Print results
# print("> Direct Methods")
# print('-'*69)
# print("{}\t\t{}\t{}\t{}".format('METHOD', 'n', 'RESIDUAL MAX NORM', 'ERROR NORM'))
# print('-'*69)
# for n in [10]:
#     result = []
#     max_norm = []
#     error_norm = []
#     result.append(LU_Solve(An(n), bn(n)))
#     result.append(LU_Solve(An(n), bn(n), 'kji'))
#     result.append(LU_Solve(An(n), bn(n), 'ijk'))
#     result.append(QRSolve(An(n), bn(n)))
#     result.append(QRSolve(An(n), bn(n), 2))
#     result.append(LL_TSolve(An(n), bn(n)))
#     for i in range(len(result)):
#         max = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result[i].x)))
#         max_norm.append(max)
#         error = (n + 1)**-1 * Maxnorm(vectordiff(result[i].x, scalarvectorprod(bn(n), np.pi**-2)))
#         error_norm.append(error)
#     for i in range(len(result)):
#         print("{:<15}\t{}\t{:.15e}\t{:.15e}".format(result[i].method_name, n, max_norm[i], error_norm[i]))
# print('-'*69)

# print("\n> Over-Relaxation Methods")
# print('-'*77)
# print("{}\t{}\t{}\t{}\t{}\t{}".format('METHOD', 'n', 'OMEGA', 'NUMIT', 'RESIDUAL MAX NORM', 'ERROR NORM'))
# print('-'*77)
# parameter = param()
# for parameter.omega in [0.5, 1.0, 1.5]:
#     for n in [10]:
#         result = []
#         max_norm = []
#         error_norm = []
#         x = [0] * n
#         result.append(JOR(An(n), bn(n), x, parameter))
#         x = [0] * n
#         result.append(SOR(An(n), bn(n), x, parameter))
#         x = [0] * n
#         result.append(BSOR(An(n), bn(n), x, parameter))
#         x = [0] * n
#         result.append(SSOR(An(n), bn(n), x,parameter))
#         for i in range(len(result)):
#             max = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result[i].x)))
#             max_norm.append(max)
#             error = (n + 1)**-1 * Maxnorm(vectordiff(result[i].x, scalarvectorprod(bn(n), np.pi**-2)))
#             error_norm.append(error)
#         for i in range(len(result)):
#             print("{}\t{}\t{}\t{}\t{:.15e}\t{:.15e}".format(result[i].method_name, n, parameter.omega,
#                 result[i].numit, max_norm[i], error_norm[i]))
# print('-'*77)

# print("\n> Gradient Methods")
# print('-'*85)
# print("{}\t\t\t{}\t{}\t{}\t{}".format('METHOD', 'n', 'NUMIT', 'RESIDUAL MAX NORM', 'ERROR NORM'))
# print('-'*85)
# parameter = param()
# for n in [10]:
#     result = []
#     max_norm = []
#     error_norm = []
#     x = [0] * n
#     result.append(steepestdescent(An(n), bn(n), x, parameter))
#     x = [0] * n
#     result.append(conjugategradient(An(n), bn(n), x, parameter))
#     x = [0] * n
#     result.append(cgnormalresidual(An(n), bn(n), x, parameter))
#     x = [0] * n
#     result.append(cgresidual(An(n), bn(n), x, parameter))
#     for i in range(len(result)):
#         max = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result[i].x)))
#         max_norm.append(max)
#         error = (n + 1)**-1 * Maxnorm(vectordiff(result[i].x, scalarvectorprod(bn(n), np.pi**-2)))
#         error_norm.append(error)
#     for i in range(len(result)):
#         tabspace = "\t\t"
#         if len(result[i].method_name) > 15:
#             tabspace = "\t"
#         print("{}{}{}\t{}\t{:.15e}\t{:.15e}".format(result[i].method_name, tabspace, n,
#             result[i].numit, max_norm[i], error_norm[i]))
# print('-'*85)