import numpy as np
from rootscalar import rootscalar, param, options

def f(x):
    return x**5 - x**4 - x**3 - x**2 - x - 1

def df(x):
    return 5*x**4 - 4*x**3 - 3*x**2 - 2*x - 1

parameter = param()
parameter.maxit = 1000
parameter.tol = np.finfo(float).eps

options = options
options["inexact"] = ""

method_result = []

newton_result = []
secant_result = []
muller_result = []
for k in range(0, 10):
    newton_result.append(rootscalar(f, df, None, None, 0.2*k, None,
                                    options, parameter=parameter))
    secant_result.append(rootscalar(f, None, None, None, [0.2*k + _ for _ in [0, 0.5]], None,
                                    options=dict({"method" : "secant"}), parameter=parameter))
    muller_result.append(rootscalar(f, None, None, None, [0.2*k + _ for _ in [0, 0.5, 1.]], None,
                                    options=dict({"method" : "muller"}), parameter=parameter))
    
method_result.append(newton_result)
method_result.append(secant_result)
method_result.append(muller_result)

for i in range(0, 3):
    print('\n>', method_result[i][0].method_name, 'METHOD')
    print('-'*63)
    print("{:<8}{:<24}{:<8}{:<16}{:<7}".format('K', 'APPROXIMATE ROOT', 'NUMIT',
        '|FUNVAL|', 'ERROR'))
    print('-'*63)
    method_ave_root = 0
    method_ave_numit = 0
    method_ave_funval = 0
    method_ave_error = 0
    method_success_iterate = 0
    for k in range(0, 10):
        if method_result[i][k].termination_flag == 'Success':
            method_ave_root += method_result[i][k].x
            method_ave_numit += method_result[i][k].numit
            method_ave_funval += abs(method_result[i][k].funval)/np.finfo(float).eps
            method_ave_error += method_result[i][k].error/np.finfo(float).eps
            method_success_iterate += 1
            print("{:<8}{:.16}\t{}\t{:.1f}eps\t\t{:.1f}eps".format(k,
                method_result[i][k].x, method_result[i][k].numit,
                abs(method_result[i][k].funval)/np.finfo(float).eps,
                method_result[i][k].error/np.finfo(float).eps))
        else:
            print("{:<8}{:.16}\t{}\t{:.1f}eps\t\t{:.1f}eps".format(k,
                method_result[i][k].x, method_result[i][k].numit,
                abs(method_result[i][k].funval)/np.finfo(float).eps,
                method_result[i][k].error/np.finfo(float).eps))
    print('-'*63)
    print("{:<8}{:.16}\t{:.1f}\t{:.1f}eps\t\t{:.1f}eps".format(
        'AVE', method_ave_root / method_success_iterate,
        method_ave_numit / method_success_iterate,
        method_ave_funval / method_success_iterate,
        method_ave_error / method_success_iterate))
    print('-'*63)
