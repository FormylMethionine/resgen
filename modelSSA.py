import numpy as np
import random
from numba import jit
import time


def timeit(func):
    def wrapper(*args, **kargs):
        import time
        t1 = time.time()
        result = func(*args, **kargs)
        t2 = time.time()
        print("execution time", t2-t1)
        return result
    return wrapper


def rates(X, M, K):
    R = np.zeros(M.shape[0])
    for i, r in enumerate(M):
        R[i] = K[i]*np.prod(X[np.where(r < 0)])
    R[R < 0] = 0
    return R


def Gillespie_ts(t, X, M, R, Rtot):

        tau = np.random.exponential(1/Rtot)
        t += tau

        reac = M[np.random.choice(range(M.shape[0]),
                                  p=R/Rtot)]
        X += reac

        return t, X


@timeit
def Gillespie(tmax, t, Xini, M, K):

    X = np.array(Xini, ndmin=2)
    t = np.array(t)
    K = np.array(K)
    M = np.array(M)

    while t[-1] < tmax:

        R = rates(X[-1], M, K)
        Rtot = np.sum(R)
        if Rtot == 0:
            break

        t_new, X_new = Gillespie_ts(t[-1], X[-1], M, R, Rtot)

        #X = np.concatenate((X, X_new))
        X = np.vstack((X, X_new))
        t = np.append(t, t_new)

    return t, X


if __name__ == "__main__":
    init = [290, 10, 0]
    stoch = [[-1, 1, 0],
             [0, -1, 1]]
    const = [2, 0.5]
    t = [0.0]
    t, X = Gillespie(10, t, init, stoch, const)
    print(X[-1])
    print(len(t))
