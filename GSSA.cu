#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <curand_mtgp32_host.h>

#define NB 200
#define TPB 256

__global__
void Gillespie(int* X, int nSpecies, double* K, int nReacs, int* M, double
        tstart, double tmax, int N, curandStateMtgp32* states) {

    double t = tstart;
    double* R = (double*)malloc(nReacs*sizeof(double));
    double Rsum; // sum of reaction rates
    double partialRsum; // partial sum of reaction rates
    int choice; // index of chosen reaction
    double r1, r2; // random numbers
    double tau; // increment of time
    bool exit; // flag to exit the loop
    
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    while (t < tmax) {

        Rsum = 0;

        //Calculte reaction rates
        for (int i=0; i<nReacs; i++) {
            R[i] = K[i];
            for (int j=0; j<nSpecies; j++) {
                //printf("%d ", M[i*nSpecies+j]);
                if (M[i*nSpecies + j] < 0) {
                    //printf("(%d) ", X[id + j*N]);
                    R[i] *= pow((double)X[j*N + id], 
                                (double)(-M[i*nSpecies + j]));
                }
            }
            //printf("\n");
            Rsum += R[i];
        }
        //printf("\n");

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
        X[i*N + id] += M[choice*nSpecies + i];

    }

}

int main() {

    int N = NB*TPB;

    int* X;
    double* K;
    int* M;

    cudaMallocManaged(&X, 3*N*sizeof(int));
    cudaMallocManaged(&K, 4*sizeof(double));
    cudaMallocManaged(&M, 3*4*sizeof(int));

    for (int i=0; i<N; i++) X[i] = 10000;
    for (int i=N; i<2*N; i++) X[i] = 0;
    for (int i=2*N; i<3*N; i++) X[i] = 0;

    K[0] = 1;
    K[1] = .002;
    K[2] = .5;
    K[3] = .04;

    M[0] = -1;
    M[1] = 0;
    M[2] = 0;

    M[3] = -2;
    M[4] = 1;
    M[5] = 0;

    M[6] = 2;
    M[7] = -1;
    M[8] = 0;

    M[9] = 0;
    M[10] = -1;
    M[11] = 1;

    curandStateMtgp32* states;
    cudaMalloc(&states, N*sizeof(curandStateMtgp32));

    mtgp32_kernel_params* kernelParams;
    cudaMalloc(&kernelParams, sizeof(kernelParams));
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernelParams);
    curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213,
            kernelParams, NB, time(NULL));

    Gillespie <<<NB, TPB>>> (X, 3, K, 4, M, 0.0, 6.0, N, states);
    cudaDeviceSynchronize();


    float X_mean[3] = {0, 0, 0};
    for (int i=0; i<3; i++) for (int j=i*N; j<(i+1)*N; j++) X_mean[i] += X[j];
    for (int i=0; i<3; i++) X_mean[i] /= N;
    for (int i=0; i<3; i++) std::cout << X_mean[i] << " ";
    std::cout << std::endl;
       
}
