#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size; // elements per process

    /* Only root allocates the full array */
    int *full_array = NULL;
    if (rank == 0) {
        full_array = (int *)malloc(N * sizeof(int));
        /* Root fills the array with values 1 to N */
        for (int i = 0; i < N; i++)
            full_array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Each process allocates its own chunk */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /*
     * SCATTER: Root sends chunks of the array to all processes.
     * Each process receives its chunk into local_chunk.
     */
    MPI_Scatter(full_array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process computes the sum of its own chunk */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    /* Allreduce: every process gets the total sum */
    long long total_sum;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    /* Every process prints its local_sum, the total_sum, and its percentage contribution */
    double percentage = (double)local_sum / total_sum * 100.0;
    printf("  Rank %d: local_sum = %lld, total_sum = %lld, percentage = %.2f%%\n",
           rank, local_sum, total_sum, percentage);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Allreduce] Total sum   = %lld\n", total_sum);
        printf("[Allreduce] Expected    = %lld\n", expected);
        printf("[Allreduce] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time        = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0)
        free(full_array);

    MPI_Finalize();
    return 0;
}