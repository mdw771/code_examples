#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void print_array(float *x, int size){
    for (int i = 0; i < size; i++)
        printf("%f ", x[i]);
    printf("\n");
}

int main(){
    
    MPI_Init(NULL, NULL);

    int n = 10;
    float a[n];
    float res[n];
    int n_ranks, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    for (int i = 0; i < n; i++){
       a[i] = rank;
    }

    print_array(a, n);
    
    MPI_Allreduce(a, res, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    print_array(res, n);

    MPI_Finalize();

    return 0;
}