#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <curand_mtgp32_host.h>

#define NB 200
#define TPB 256

__global__
void Gillespie(int* X, int nSpecies, float* K, int nReacs, int* M, float
        tstart, float tmax, int N, curandStateMtgp32* states) {

    float t = tstart;
    float* R = (float*)malloc(nReacs*sizeof(float));
    float Rsum; // sum of reaction rates
    float partialRsum; // partial sum of reaction rates
    int choice; // index of chosen reaction
    float r1, r2; // random numbers
    float tau; // increment of time
    bool exit; // flag to exit the loop
    
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    //setting up random generator
    //curand_init(clock64(), threadIdx.x, 0, &states[threadIdx.x]);

    while (t < tmax) {

        Rsum = 0;

        //Calculte reaction rates
        for (int i=0; i<nReacs; i++) {
            R[i] = K[i];
            for (int j=0; j<nSpecies; j++) 
                if (M[i*nSpecies + j] < 0) R[i] *= pow(X[id + i*N], -M[i*nSpecies + j]);
            Rsum += R[i];
        }

        exit = true;
        for (int i=0; i<nReacs; i++) if (R[i] != 0) exit=false; 
        if (exit) break;

        // Draw two random numbers
        r1 = curand_uniform(&states[blockIdx.x]);
        r2 = curand_uniform(&states[blockIdx.x]);
        
        // Select reaction to fire
        choice = 0;
        partialRsum = R[choice];
        while (partialRsum < r2*Rsum) {
            choice++;
            partialRsum += R[choice];
        }
        
        // Pass time
        tau = -log(r1)/Rsum;
        t += tau;
        
        // update X
        for (int i=0; i<nSpecies; i++) 
            X[id + i*N] += M[choice*nSpecies + i];

    }

}

int main() {

    int N = NB*TPB;

    int* X;
    float* K;
    int* M;

    cudaMallocManaged(&X, 3*N*sizeof(int));
    cudaMallocManaged(&K, 2*sizeof(float));
    cudaMallocManaged(&M, 3*2*sizeof(float));

    for (int i=0; i<N; i++) X[i] = 300;
    for (int i=N; i<2*N; i++) X[i] = 10;
    for (int i=2*N; i<3*N; i++) X[i] = 0;

    K[0] = 2;
    K[1] = .5;

    M[0] = -1;
    M[1] = 1;
    M[2] = 0;

    M[3] = 0;
    M[4] = -1;
    M[5] = 1;

    curandStateMtgp32* states;
    cudaMalloc(&states, N*sizeof(curandStateMtgp32));

    mtgp32_kernel_params* kernelParams;
    cudaMalloc(&kernelParams, sizeof(kernelParams));
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernelParams);
    curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213,
            kernelParams, NB, time(NULL));

    auto start = std::chrono::high_resolution_clock::now();
    Gillespie <<<NB, TPB>>> (X, 3, K, 2, M, 0.0, 10.0, N, states);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto time = end - start;

    float X_mean[3] = {0, 0, 0};
    for (int i=0; i<3; i++) for (int j=i*N; j<(i+1)*N; j++) X_mean[i] += X[j];
    for (int i=0; i<3; i++) X_mean[i] /= N;
    for (int i=0; i<3; i++) std::cout << X_mean[i] << " ";
    std::cout << std::endl;
    std::cout << "Time taken: " << time/std::chrono::milliseconds(1) << "ms"
        << std::endl;
       
}
