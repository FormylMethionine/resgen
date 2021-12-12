#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>

#define MAX_BLOCKS 1024
#define TPB 1024

__global__
void Gillespie(int* X, 
        const int nSpecies, 
        const double* K, 
        const int nReacs, 
        const int* M, 
        const double tstart,
        const double tmax,
        const int N,
        unsigned long int seed) {

    double t;
    //double* R = (double*)malloc(nReacs*sizeof(double));
    double* R = new double[nSpecies];
    double Rsum; // sum of reaction rates
    double partialRsum; // partial sum of reaction rates
    int choice; // index of chosen reaction
    double r1, r2; // random numbers
    double tau; // increment of time
    bool exit; // flag to exit the loop
    
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState_t state;
    curand_init(seed, tid, 0, &state);

    for (int id=tid; id<N; id += stride) {

        t = tstart;

        while (t < tmax) {

            __syncthreads();

            Rsum = 0;

            //Calculte reaction rates
            for (int i=0; i<nReacs; i++) {

                R[i] = K[i];

                for (int j=0; j<nSpecies; j++) {

                    if (M[i*nSpecies + j] < 0) {

                        if (X[j*N + id] >= -M[i*nSpecies + j]) {
                            // Reaction rate can only be non zero if there is
                            // enough reactants to permit reactions
                            // this a safeguard to prevent X from going negative
                            R[i] *= pow((double)X[j*N + id], 
                                        (double)(-M[i*nSpecies + j]));
                        } else R[i] *= 0;
                    }
                }

                Rsum += R[i];
            }

            exit = true;
            for (int i=0; i<nReacs; i++) if (R[i] != 0) exit=false; 
            if (exit) break;

            // Draw two random numbers
            r1 = curand_uniform(&state);
            r2 = curand_uniform(&state);
            
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
            for (int i=0; i<nSpecies; i++) {
                X[i*N + id] += M[choice*nSpecies + i];
            }

        }

    }

    delete[] R;

}

int main(int argc, char** argv) {

    if (argc != 5) {
        std::cout << "Argument!" << std::endl;
        return 1;
    }

    // Reading network file
    
    std::cout << argv[1] << std::endl;

    std::ifstream network(argv[1]);

    int N = std::stoi(argv[2]);

    int NB = std::min((N + TPB - 1) / TPB, MAX_BLOCKS);

    double tstart = std::stod(argv[3]);
    double tend = std::stod(argv[4]);

    int nSpecies;
    int nReacs;
    int* Xini;
    double* K;
    int* M;

    if (network.is_open()) {

        std::string line;
        std::string delimiter = " ";
        int line_nb = 0;
        std::vector<std::string> words{};
        size_t pos = 0;

        while (std::getline(network, line)) {
            
            // splitting line into numbers
            while ((pos = line.find(delimiter)) != std::string::npos) {
                words.push_back(line.substr(0, pos));
                line.erase(0, pos + delimiter.length());
            } 
            if (pos == std::string::npos) {
                words.push_back(line.substr(0, std::string::npos));
            }

            switch(line_nb) {

                case 0 : 
                {
                    nSpecies = std::stoi(words[0]);
                    Xini = new int[nSpecies];
                    for (int i=0; i<nSpecies; i++)
                        Xini[i] = std::stoi(words[i+1]);
                    break;
                }

                case 1:
                {
                    nReacs = std::stoi(words[0]);
                    cudaMallocManaged(&K, nReacs*sizeof(double));
                    for (int i=0; i<nReacs; i++)
                        K[i] = std::stod(words[i+1]);
                    break;
                }

                case 2:
                {
                    cudaMallocManaged(&M, nReacs*nSpecies*sizeof(int));
                    for (int i=0; i<(nReacs*nSpecies); i++)
                        M[i] = std::stoi(words[i]);
                    break;
                }                  

            }

            line_nb++;
            words.clear();

        }
    } else {
        std::cout << "File not opening you dumbfuck!" << std::endl;
        return 1;
    }

    int* X;
    cudaMallocManaged(&X, nSpecies*N*sizeof(int));

    for (int i=0; i<nSpecies; i++) {
        for (int j=0; j<N; j++) X[i*N + j] = Xini[i];
    }

    std::cout << "file: " << argv[1] << " "
        << "N: " << N << " "
        << "Nspecies: " << nSpecies << " "
        << "Nreacs: " << nReacs << " "
        << "NB: " << NB << " "
        << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Gillespie <<<NB, TPB>>> (X, nSpecies, K, nReacs, M, tstart, tend, N, time(NULL));
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto time = end - start;

    std::cout << "Time taken: " << time/std::chrono::milliseconds(1) << "ms"
        << std::endl;

    double Xavg[nSpecies];

    for (int i=0; i<nSpecies; i++) {
        Xavg[i] = 0;
        for (int j=0; j<N; j++) {
            Xavg[i] += X[i*N + j];
        }
        Xavg[i] /= N;
    }

    std::ofstream results;
    results.open("results.txt");

    if (results.is_open()) {
        for (int i=0; i<N; i++) {
            for (int j=0; j<nSpecies; j++) {
                results << X[j*N + i] << ",";
            }
            results << "\n";
        }
    } else {
        std::cout << "Unable to open result file" << std::endl;
        return 1;
    }

    results.close();

    results.open("time.txt");

    if (results.is_open()) {
        results << nSpecies 
            << "," 
            << nReacs 
            << "," 
            << N 
            << "," 
            << time/std::chrono::milliseconds(1)
            << "\n";
    } else {
        std::cout << "Unable to open result file" << std::endl;
        return 1;
    }

    results.close();

    delete Xini;
    cudaFree(&X);
    cudaFree(&K);
    cudaFree(&M);

    return 0;
       
}
