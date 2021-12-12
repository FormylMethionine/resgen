import os
import numpy as np
import pandas as pd

os.system('nvcc ./GSSA.cu -O3 -o GSSA_cu')

sizes = [10, 20, 50, 100, 250, 500, 750]
Nspecies = []
Nreacs = []
N = []
times = []
method = []
n = 5
reps = [10, 20, 50, 100, 250, 500, 750, 1000]

for i in sizes:
    for j in reps:
        os.system(f"./GSSA_cu networks/network{i}.txt {j} 0 0.05")
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
