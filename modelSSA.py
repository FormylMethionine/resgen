import numpy as np
import random
from numba import jit


def timeit(func):
    def wrapper(*args, **kargs):
        import time
        t1 = time.time()
        result = func(*args, **kargs)
        t2 = time.time()
        print("execution time", t2-t1)
        return result
    return wrapper


class modelSSA:

    def __init__(self, ini, stoch, const):
        self.ini = np.array(ini)
        self.stoch = np.array(stoch)
        self.const = np.array(const)
        self.times = np.array([0.0])

        self.states = np.array([self.ini], ndmin=2)

    @timeit
    def Gillespie(self, dur):

        while self.times[-1] < dur:

            state = self.states[-1]

            a = np.zeros(self.stoch.shape[0])
            for i, r in enumerate(self.stoch):
                a[i] = self.const[i]*np.prod(state[np.where(r < 0)])
            a[a < 0] = 0
            atot = np.sum(a)

            if atot == 0:
                break

            tau = np.random.exponential(1/atot)
            self.times = np.append(self.times, self.times[-1]+tau)

            reac = random.choices(self.stoch, weights=a)
            self.states = np.concatenate((self.states, state+reac))


if __name__ == "__main__":
    init = [290, 10, 0]
    stoch = [[-1, 1, 0],
             [0, -1, 1]]
    const = [2, 0.5]
    model = modelSSA(init, stoch, const)
    print(model.states[-1])
    model.run(100)
    print(model.states[-1])
    print(model.times[-1])
