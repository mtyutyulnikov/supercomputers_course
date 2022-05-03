#include <mpi.h>
#include <malloc.h>
#include <stdio.h>

//#include <mpe.h>

void getXValues(const int matrixSize, int workersNum, double *x) {
    int xId = 0;
    double xPart[matrixSize];
    int workerId = 0;
    for (workerId = 0; workerId < workersNum; workerId++) {
        if (workerId < matrixSize % workersNum) {
            MPI_Recv(&xPart, (matrixSize / workersNum + 1), MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            int i;
            for (i = 0; i < (matrixSize / workersNum + 1); i++) {
                x[xId] = xPart[i];
                xId++;
            }
        } else {
            MPI_Recv(&xPart, (matrixSize / workersNum), MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            int i = 0;
            for (i = 0; i < (matrixSize / workersNum); i++) {
                x[xId] = xPart[i];
                xId++;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const int matrixSize = 3500;
    const double eps = 1E-10;
    const double tau = 0.00001;
    double error = 10;

    int numProcs, rank;
    MPI_Init(&argc, &argv);
//    MPE_Init_log();

    MPI_Barrier(MPI_COMM_WORLD);
    double starttime = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    if (numProcs < 2) {
//        throw std::invalid_argument("can't work with one thread");
//    }

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
    double *b_part;

//    double *x = new double[matrixSize];
    double *x = (double *) malloc(matrixSize * sizeof(double));
    int i, j;
    for (i = 0; i < matrixSize; i++)
        x[i] = 0;

    double *b = (double *) malloc(matrixSize * sizeof(double));
    for (i = 0; i < matrixSize; i++)
        b[i] = matrixSize + 1;
    

    int startRow = 0;
    if (rank == 0) { //MAIN
//        double *A = new double[matrixSize * matrixSize];
        double *A = (double *) malloc(matrixSize * matrixSize * sizeof(double));
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                if (i == j)
                    A[i * matrixSize + j] = 2;
                else
                    A[i * matrixSize + j] = 1;
            }
        }

        int matrixIdx = 0;
        int workerId;
        for (workerId = 0; workerId < workersNum; workerId++) {
            if (workerId < matrixSize % workersNum) {
                MPI_Send(A + matrixIdx, (matrixSize / workersNum + 1) * matrixSize, MPI_DOUBLE, workerId + 1, 1,
                         MPI_COMM_WORLD);
                int matrixRow = matrixIdx / matrixSize;
                MPI_Send(&matrixRow, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);

                matrixIdx += (matrixSize / workersNum + 1) * matrixSize;
            } else {
                MPI_Send(A + matrixIdx, (matrixSize / workersNum) * matrixSize, MPI_DOUBLE, workerId + 1, 1,
                         MPI_COMM_WORLD);
                int matrixRow = matrixIdx / matrixSize;
                MPI_Send(&matrixRow, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);

                matrixIdx += (matrixSize / workersNum) * matrixSize;
            }
        }

        //CALC
        double denominator = 0;
        for (i = 0; i < matrixSize; i++) {
            denominator += b[i] * b[i];
        }

        double errors[workersNum];
        int needToCalc = 1;

        while (error >= eps) {
            for (workerId = 0; workerId < workersNum; workerId++) {
                MPI_Send(&needToCalc, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);
            }

            getXValues(matrixSize, workersNum, x);
//            MPI_Bcast(x, matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (workerId = 0; workerId < workersNum; workerId++) {
                MPI_Send(x, matrixSize, MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD);
            }


            double nominator = 0;
            for (workerId = 0; workerId < workersNum; workerId++) {
                double procError = 0;
                MPI_Recv(&procError, 1, MPI_DOUBLE, workerId + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nominator += procError;
            }

            error = nominator / denominator;
        }

        needToCalc = 0;
        for (workerId = 0; workerId < workersNum; workerId++) {
            MPI_Send(&needToCalc, 1, MPI_INT, workerId + 1, 1, MPI_COMM_WORLD);
        }

        getXValues(matrixSize, workersNum, x);

        //PRINT X
//        for (int i = 0; i < matrixSize; i++) {
//            printf("%f ", x[i]);
//        }

        MPI_Barrier(MPI_COMM_WORLD);
        double endtime = MPI_Wtime();

        printf("\nMPI Time: %.2f sec\n", endtime - starttime);


    } else { //WORKER
        //INIT
//        A_part = new double[thisProcRowsNum * matrixSize];
        A_part = (double *) malloc(thisProcRowsNum * matrixSize * sizeof(double));

        MPI_Recv(A_part, thisProcRowsNum * matrixSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Rank %d received %d items starting from row %d  \n", rank, thisProcRowsNum * matrixSize, startRow);


        //CALC
        int needToCalc = 1;

        while (1) {
            MPI_Recv(&needToCalc, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!needToCalc)
                break;

            int row;
            for (row = 0; row < thisProcRowsNum; row++) {
                double delta = 0;
                for (j = 0; j < matrixSize; j++) {
                    delta += A_part[row * matrixSize + j] * x[j];
                }
                delta -= b[startRow+row];
                x[startRow + row] = x[startRow + row] - tau * delta;
            }

            MPI_Send(x + startRow, thisProcRowsNum, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(x, matrixSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double sendValue = 0;

            for (row = 0; row < thisProcRowsNum; row++) {
                double value = 0;

                for (j = 0; j < matrixSize; j++) {
                    value += A_part[row * matrixSize + j] * x[j];
                }
                value -= b[startRow+row];
                value = value * value;

                sendValue += value;
            }

            MPI_Send(&sendValue, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        }
        MPI_Send(x + startRow, thisProcRowsNum, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}


