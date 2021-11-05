#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <random>

template <size_t nReacs, size_t nSpecies>
int* Gillespie(int (&X)[nSpecies], double (&K)[nReacs], int
        (&M)[nReacs][nSpecies], double tstart, double tmax) {

    double t = tstart;
    double R[nReacs]; // Array of reaction rates
    double Rsum; // sum of reaction rates
    double partialRsum; // partial sum of reaction rates
    int choice; // index of chosen reaction
    double r1, r2; // random numbers
    double tau; // increment of time
    bool exit; // flag to exit the loop

    //setting up random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    while (t < tmax) {

        Rsum = 0;

        //Calculte reaction rates
        for (int i=0; i<nReacs; i++) {
            R[i] = K[i];
            for (int j=0; j<nSpecies; j++) 
                if (M[i][j] < 0) R[i] *= pow(X[j], -M[i][j]);
            Rsum += R[i];
        }

        exit = true;
        for (int i=0; i<nReacs; i++) if (R[i] != 0) exit=false; 
        if (exit) break;

        // Draw two random numbers
        r1 = dis(gen);
        r2 = dis(gen);
        
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
        for (int i=0; i<nSpecies; i++) X[i] += M[choice][i];

    }

    return X;

}

int main() {

    int X[3] = {300, 10, 0};
    int M[2][3] = {{-1, 1, 0}, {0, -1, 1}};
    double K[2] = {2, .5};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<30000; i++) {
        X[0] = 300;
        X[1] = 10;
        X[2] = 0;
        Gillespie(X, K, M, 0.0, 10);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto time = end - start;

    for (int i=0; i<3; i++) std::cout << X[i] << " ";
    std::cout << std::endl;
    std::cout << "Time taken: " << time/std::chrono::milliseconds(1) << "ms"
        << std::endl;

    return 0;
}
