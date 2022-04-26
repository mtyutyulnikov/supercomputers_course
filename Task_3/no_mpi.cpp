//#include<mpi.h> // Подключение библиотеки MPI
#include<cstdio>
#include "time.h"

int main(int argc, char *argv[]) {

    clock_t tStart = clock();
    const int matrixSize = 3500;


    double *A = new double [matrixSize*matrixSize];


    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            if (i == j)
                A[i*matrixSize + j] = 2;
            else
                A[i*matrixSize + j] = 1;
        }
    }

    double * x = new double[matrixSize];
    for (int i = 0; i < matrixSize; i++)
        x[i] = 0;

    double * newX = new double[matrixSize];
    for (int i = 0; i < matrixSize; i++)
        newX[i] = 0;

    double * b = new double[matrixSize];
    for (int i = 0; i < matrixSize; i++)
        b[i] = matrixSize + 1;

    const double eps = 1E-10;
    const double tau = 0.00001;
    double error = 10;

    double denominator = 0;
    for (int i = 0; i < matrixSize; i++) {
        denominator += b[i] * b[i];
    }
//    denominator = sqrt(denominator);

    while (error >= eps) {

        for (int i = 0; i < matrixSize; i++) {
            double delta=0;
            for (int j = 0; j < matrixSize; j++) {
                delta += A[i*matrixSize + j] * x[j];
            }
            delta -= b[i];
            newX[i] = x[i] - tau * delta;
        }

        double nominator = 0;

        for (int i = 0; i < matrixSize; i++) {
            double value = 0;
            for (int j = 0; j < matrixSize; j++) {
                value += A[i*matrixSize + j] * newX[j];
            }
            value -= b[i];
            nominator += value*value;
        }

        error = nominator/denominator;
//        error = sqrt(error);

//        nominator = sqrt(nominator);

        for (int i = 0; i<matrixSize; i++){
            x[i] = newX[i];
        }
    }

    for (int i=0; i<matrixSize; i++){
        printf("%f ", newX[i]);
    }

    printf("\nTime taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}