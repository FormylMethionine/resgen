#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <random>

int* Gillespie(int* X, 
        int nSpecies,
        double* K,
        int nReacs,
        int* M,
        double tstart, double tmax) {

    double t = tstart;
    double t_new;
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

        //for (int i=0; i<nReacs; i++) std::cout << K[i] << " ";
        //std::cout << std::endl;

        Rsum = 0;

        //Calculte reaction rates
        for (int i=0; i<nReacs; i++) {
            R[i] = K[i];
            for (int j=0; j<nSpecies; j++) 
                if (M[i*nSpecies + j] < 0) 
                    R[i] *= pow(X[j], -M[i*nSpecies + j]);
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
        //std::cout << t << " " << tau << std::endl;
        t_new = t + tau;
        if (t_new == t) break;
        t = t_new;

        // update X
        for (int i=0; i<nSpecies; i++) X[i] += M[choice*nSpecies + i];

    }

    return X;

}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cout << "Argument!" << std::endl;
        return 1;
    }

    // Reading network file
    
    std::cout << argv[1] << std::endl;

    std::ifstream network(argv[1]);

    int N = std::stoi(argv[2]);

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
                    K = new double[nReacs];
                    for (int i=0; i<nReacs; i++)
                        K[i] = std::stod(words[i+1]);
                    break;
                }

                case 2:
                {
                    M = new int[nSpecies*nReacs];
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

    int* Xs[N];

    for (int i=0; i<N; i++) {
        Xs[i] = new int[nSpecies];
        for (int j=0; j<nSpecies; j++) (Xs[i])[j] = Xini[j];
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<N; i++)
        Gillespie(Xs[i], nSpecies, K, nReacs, M, 0.0, 1.0);
    auto end = std::chrono::high_resolution_clock::now();

    auto time = end - start;

    //for (int i=0; i<3; i++) std::cout << X[i] << " ";
    std::cout << std::endl;
    std::cout << "Time taken: " << time/std::chrono::milliseconds(1) << "ms"
        << std::endl;

    int Xavg[nSpecies];

    for (int i=0; i<nSpecies; i++) {
        for (int j=0; j<N; j++) {
            Xavg[i] += (Xs[j])[i];
        }
        Xavg[i] /= N;
    }
    for (int i=0; i<N; i++) delete Xs[i];

    std::ofstream results;
    results.open("results.txt");

    if (results.is_open()) {
        for (int i=0; i<nSpecies; i++) {
            results << Xavg[i] << ",";
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
            << time/std::chrono::milliseconds(1);
    } else {
        std::cout << "Unable to open result file" << std::endl;
        return 1;
    }

    results.close();

    delete Xini;
    delete K;
    delete M;

    return 0;
}
