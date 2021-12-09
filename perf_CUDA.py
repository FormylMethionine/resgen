import os
import numpy as np
import pandas as pd

os.system('nvcc ./GSSA.cu -O3 -o GSSA_cu')

sizes = [100, 500, 1000]
Nspecies = []
Nreacs = []
N = []
times = []
method = []
n = 20
reps = [1000, 5000, 10000, 50000, 100000, 200000]

for i in sizes:
    for j in reps:
        for _ in range(n):
            os.system(f"nvprof ./GSSA_cu networks/network{i}.txt {j}")
            for line in open('./time.txt', 'r'):
                line = line[:-1].split(',')
                times.append(int(line[-1]))
                Nspecies.append(int(line[0]))
                Nreacs.append(int(line[1]))
                N.append(int(line[2]))
                method.append('CUDA')

df = pd.DataFrame(data={'Nspecies':Nspecies,
                        'Nreacs':Nreacs,
                        'N':N,
                        't':times,
                        'method':method})
df.to_csv('results_CUDA.csv')
