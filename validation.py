import os
import numpy as np
import pandas as pd
import stochpy

N = 256;

model = stochpy.SSA()
model.Model('validation_model.psc', dir='./')
model.DoStochSim(method='direct', trajectories=N, mode='time', end=1.0)

distrib = np.zeros((N, 3))
for i in range(N):
    model.GetTrajectoryData(i+1)
    distrib[i, :] = model.data_stochsim.species[-1]

df = pd.DataFrame(data={'S1': distrib[:, 0],
                        'S2': distrib[:, 1],
                        'S3': distrib[:, 2],
                        'method': 'stochpy'})

os.system('g++ ./GSSA.cpp -O3 -o ./GSSA')
os.system(f'./GSSA networks/test.txt {N} 0 1')

for i, line in enumerate(open('./results.txt', 'r')):
    line = line[:-1].split(',')
    for j in range(3):
        distrib[i, j] = int(line[j])

df2 = pd.DataFrame(data={'S1': distrib[:, 0],
                         'S2': distrib[:, 1],
                         'S3': distrib[:, 2],
                         'method': 'C++'})

os.system('nvcc ./GSSA.cu -O3 -o GSSA_cu')
os.system(f'nvprof ./GSSA_cu networks/test.txt {N} 0 1')

for i, line in enumerate(open('./results.txt', 'r')):
    line = line[:-1].split(',')
    for j in range(3):
        distrib[i, j] = int(line[j])

df3 = pd.DataFrame(data={'S1': distrib[:, 0],
                         'S2': distrib[:, 1],
                         'S3': distrib[:, 2],
                         'method': 'CUDA'})

df = df.append(df2)
df = df.append(df3)

df.to_csv('validation.csv')
