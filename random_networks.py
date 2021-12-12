import numpy as np
import os

path = "./networks"

if not os.path.exists(path):
    os.makedirs(path)

Nspecies = np.array([10, 20, 50, 100, 250, 500, 750, 1000])

for i, Nsp in enumerate(Nspecies):
    print(i)
    X = np.random.randint(0, 1000, size=Nsp)
    Nreacs = int(.8*Nsp)
    K = 10**(-1*(np.random.rand(Nreacs)*5))
    M = np.zeros((Nreacs, Nsp))
    # Generating stoechiometry matrix
    for i in range(Nreacs):
        # Species consumed
        if np.random.randint(1, 3) == 1:
            M[i][np.random.randint(1, Nsp)] = -2
        else:
            choices = np.random.choice(np.arange(Nsp), 2)
            for j in choices:
                M[i][j] = -1
        # Species produced
        if np.random.randint(1, 3) == 1:
            M[i][np.random.randint(1, Nsp)] = 2
        else:
            choices = np.random.choice(np.arange(Nsp), 2)
            for j in choices:
                M[i][j] = 1
    M = np.unique(M, axis=0)
    Nreacs = M.shape[0]
    # writing network
    with open(f"{path}/network{Nsp}.txt", "w") as f:
        # writing species
        f.write(f"{Nsp} ")
        for i in range(Nsp):
            f.write(f"{X[i]} ")
        f.write('\n')
        # writing reaction rates
        f.write(f"{Nreacs} ")
        for i in range(Nreacs):
            f.write(f"{K[i]} ")
        f.write('\n')
        # writing stoechiometry matrix
        for i in range(Nreacs):
            for j in range(Nsp):
                f.write(f"{M[i][j]} ")
        f.write('\n')
