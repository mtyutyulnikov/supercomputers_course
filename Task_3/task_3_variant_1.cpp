#include <mpi.h>
#include <cstdio>
#include <stdexcept>
#include "time.h"


void getXValues(const int matrixSize, int workersNum, double *x) {
    int xId = 0;
    double xPart[matrixSize];

    for (int workerId = 0; workerId < workersNum; workerId++) {
        if (workerId < matrixSize % workersNum) {
            MPI_Recv(&xPart, (matrixSize / workersNum + 1), MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < (matrixSize / workersNum + 1); i++) {
                x[xId] = xPart[i];
                xId++;
            }
        } else {
            MPI_Recv(&xPart, (matrixSize / workersNum), MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            for (int i = 0; i < (matrixSize / workersNum); i++) {
                x[xId] = xPart[i];
                xId++;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const int matrixSize = 10;

    clock_t tStart = clock();

    int numProcs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numProcs < 2) {
        throw std::invalid_argument("can't work with one thread");
    }

    int thisProcRowsNum = 0;
    int workersNum = numProcs - 1;

    if (rank > 0) {
        if (rank < matrixSize % workersNum + 1)
            thisProcRowsNum = matrixSize / workersNum + 1;
        else
            thisProcRowsNum = matrixSize / workersNum;
    }

    printf("Rank #%d will process %d rows\n", rank, thisProcRowsNum);

    double *A_part;
    double *b = new double[matrixSize];
    double *x = new double[matrixSize];
    double *newX = new double[matrixSize];

    for (int i = 0; i < matrixSize; i++)
        x[i] = 0;

    for (int i = 0; i < matrixSize; i++)
        b[i] = matrixSize + 1;

    for (int i = 0; i < matrixSize; i++)
        newX[i] = 0;

    int startRow = 0;
    if (rank == 0) {
        double *A = new double[matrixSize * matrixSize];

        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                if (i == j)
                    A[i * matrixSize + j] = 2;
                else
                    A[i * matrixSize + j] = 1;
            }
        }

        int matrixIdx = 0;
        for (int workerId = 0; workerId < workersNum; workerId++) {
            if (workerId < matrixSize % workersNum) {
                MPI_Ssend(A + matrixIdx, (matrixSize / workersNum + 1) * matrixSize, MPI_DOUBLE, workerId + 1, 1,
                          MPI_COMM_WORLD);
                int matrixRow = matrixIdx / matrixSize;
                MPI_Ssend(&matrixRow, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);
                matrixIdx += (matrixSize / workersNum + 1) * matrixSize;
            } else {
                MPI_Ssend(A + matrixIdx, (matrixSize / workersNum) * matrixSize, MPI_DOUBLE, workerId + 1, 1,
                          MPI_COMM_WORLD);
                int matrixRow = matrixIdx / matrixSize;
                MPI_Ssend(&matrixRow, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);

                matrixIdx += (matrixSize / workersNum) * matrixSize;
            }
        }
    } else {
        A_part = new double[thisProcRowsNum * matrixSize];
        MPI_Recv(A_part, thisProcRowsNum * matrixSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Rank %d received %d items from row %d  \n", rank, thisProcRowsNum * matrixSize, startRow);

    }


    const double eps = 1E-10;
    const double tau = 0.00001;
    double error = 10;


    if (rank == 0) { //Main machine
        double denominator = 0;
        for (int i = 0; i < matrixSize; i++) {
            denominator += b[i] * b[i];
        }

        double errors[workersNum];
        bool needToCalc = true;
        while (error >= eps) {
            MPI_Request request;

            for (int workerId = 0; workerId < workersNum; workerId++) {
                MPI_Ssend(&needToCalc, 1, MPI_C_BOOL, workerId + 1, 1, MPI_COMM_WORLD);
            }

            getXValues(matrixSize, workersNum, x);
//            MPI_Bcast(x, matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (int workerId = 0; workerId < workersNum; workerId++) {
                MPI_Ssend(x, matrixSize, MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD);
            }


            double nominator = 0;
            for (int workerId = 0; workerId < workersNum; workerId++) {
                double procError = 0;
                MPI_Recv(&procError, 1, MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nominator += procError;
            }

            error = nominator / denominator;
        }


        needToCalc = false;
        for (int workerId = 0; workerId < workersNum; workerId++) {
            MPI_Ssend(&needToCalc, 1, MPI_C_BOOL, workerId + 1, 1, MPI_COMM_WORLD);
        }

        getXValues(matrixSize, workersNum, x);

        for (int i = 0; i < matrixSize; i++) {
            printf("%f ", x[i]);
        }

        printf("\nTime taken: %.2fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);


    } else {//Worker machine
        bool needToCalc = true;
        MPI_Request request;

        while (true) {
            MPI_Recv(&needToCalc, 1, MPI_C_BOOL, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!needToCalc)
                break;

            for (int row = 0; row < thisProcRowsNum; row++) {
                double delta = 0;
                for (int j = 0; j < matrixSize; j++) {
                    delta += A_part[row * matrixSize + j] * x[j];
                }
                delta -= b[startRow + row];
                x[startRow + row] = x[startRow + row] - tau * delta;
            }

            MPI_Ssend(x + startRow, thisProcRowsNum, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(x, matrixSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double sendValue = 0;

            for (int row = 0; row < thisProcRowsNum; row++) {
                double value = 0;

                for (int j = 0; j < matrixSize; j++) {
                    value += A_part[row * matrixSize + j] * x[j];
                }
                value -= b[startRow + row];
                value = value * value;

                sendValue += value;
            }

            MPI_Ssend(&sendValue, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        }
        MPI_Ssend(x + startRow, thisProcRowsNum, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

