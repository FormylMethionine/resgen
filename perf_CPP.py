import os
import numpy as np
import pandas as pd

os.system('g++ ./GSSA.cpp -O3 -o GSSA')

sizes = [10, 20, 50, 100, 250, 500, 750]
Nspecies = []
Nreacs = []
N = []
times = []
method = []
n = 100

for i in sizes:
    os.system(f"./GSSA networks/network{i}.txt {n} 0 0.05")
    for line in open('./time.txt', 'r'):
        line = line[:-1].split(',')
        times.append(int(line[-1]))
        Nspecies.append(int(line[0]))
        Nreacs.append(int(line[1]))
        N.append(int(line[2]))
        method.append('C++')

df = pd.DataFrame(data={'Nspecies':Nspecies,
                        'Nreacs':Nreacs,
                        'N':N,
                        't':times,
                        'method':method})
df.to_csv('results_C++.csv')
